import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from txnclass.pipeline import (
    accuracy_at_coverage,
    build_features,
    clean_labels,
    clean_text,
    client_split,
    coverage_curve,
    decision_confidence,
    random_split,
    validate_columns,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


# ---------------- text and labels ----------------

def test_clean_text_strips_codes_numbers_and_short_tokens():
    assert clean_text("POS PURCHASE 0412 STARBUCKS #1234 SEATTLE WA 06/14") == "pos purchase starbucks seattle"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == "none"
    assert clean_text(float("nan")) == "nan"


def test_clean_labels_merges_duplicates_and_drops_unlabelled():
    df = pd.DataFrame({
        "client_id": [1, 1, 2, 2],
        "txn_date": ["2023-06-01"] * 4,
        "description": ["ATM FEE 12", "ATM FEE 13", "KROGER 9", "MYSTERY"],
        "amount": [3.0, 3.0, 40.0, 1.0],
        "category": [" Bank Fee", "Bank Fees", None, "Uncategorized"],
    })
    out = clean_labels(df)
    assert list(out["category"]) == ["Bank Fees", "Bank Fees"]
    assert list(out["clean_desc"]) == ["atm fee", "atm fee"]


def test_validate_columns_names_what_is_missing():
    with pytest.raises(ValueError, match="description, amount"):
        validate_columns(pd.DataFrame(columns=["client_id", "txn_date", "category"]))


# ---------------- splits ----------------

def test_client_split_never_shares_a_customer(sample):
    train, test = client_split(clean_labels(sample))
    assert set(train["client_id"]).isdisjoint(test["client_id"])
    assert len(train) + len(test) == len(clean_labels(sample))


def test_random_split_is_stratified(sample):
    df = clean_labels(sample)
    _, test = random_split(df)
    share = df["category"].value_counts(normalize=True)
    test_share = test["category"].value_counts(normalize=True).reindex(share.index, fill_value=0)
    assert np.abs(share - test_share).max() < 0.02


# ---------------- features ----------------

def test_features_add_one_amount_column(sample):
    train, test = random_split(clean_labels(sample))
    feats = build_features(train, test)
    assert feats.X_train.shape == (len(train), len(feats.tfidf.vocabulary_) + 1)
    assert feats.X_test.shape[1] == feats.X_train.shape[1]


def test_vectoriser_is_fitted_on_training_rows_only(sample):
    train, test = random_split(clean_labels(sample))
    vocab = build_features(train, test).tfidf.vocabulary_
    marker = "zzunseenmerchant"
    test = test.assign(clean_desc=test["clean_desc"] + f" {marker}")
    assert marker not in build_features(train, test).tfidf.vocabulary_
    assert marker not in vocab


# ---------------- confidence routing ----------------

def test_decision_confidence_is_gap_between_top_two():
    margins = np.array([[0.1, 0.9, 0.3], [2.0, -1.0, 1.5]])
    np.testing.assert_allclose(decision_confidence(margins), [0.6, 0.5])


def test_decision_confidence_rejects_single_class():
    with pytest.raises(ValueError):
        decision_confidence(np.array([[1.0], [2.0]]))


def test_accuracy_at_coverage_uses_most_confident_first():
    confidence = np.array([0.9, 0.1, 0.8, 0.2])
    correct = np.array([True, False, True, False])
    assert accuracy_at_coverage(confidence, correct, 0.5) == 1.0
    assert accuracy_at_coverage(confidence, correct, 1.0) == 0.5
    with pytest.raises(ValueError):
        accuracy_at_coverage(confidence, correct, 0)


def test_coverage_curve_ends_at_overall_accuracy():
    rng = np.random.default_rng(0)
    confidence, correct = rng.random(200), rng.random(200) > 0.3
    cov, acc = coverage_curve(confidence, correct)
    assert cov[-1] == 1.0
    assert acc[-1] == pytest.approx(correct.mean())


# ---------------- end to end ----------------

def test_pipeline_beats_baseline_on_unseen_customers(sample):
    from sklearn.metrics import accuracy_score
    from sklearn.svm import LinearSVC

    train, test = client_split(clean_labels(sample))
    feats = build_features(train, test)
    pred = LinearSVC(class_weight="balanced").fit(feats.X_train, feats.y_train).predict(feats.X_test)
    baseline = pd.Series(feats.y_train).value_counts(normalize=True).iloc[0]
    assert accuracy_score(feats.y_test, pred) > baseline + 0.5


def test_analysis_script_runs_on_the_sample(tmp_path):
    import analysis

    out = tmp_path / "metrics.json"
    assert analysis.main(["--data", str(SAMPLE), "--figures", str(tmp_path / "figs"), "--results", str(out)]) == 0
    assert out.exists()
    assert len(list((tmp_path / "figs").glob("*.png"))) == 7


def test_analysis_script_explains_missing_data(tmp_path, capsys):
    import analysis

    assert analysis.main(["--data", str(tmp_path / "nope.csv")]) == 2
    assert "sample_transactions.csv" in capsys.readouterr().err


# ---------------- data hygiene ----------------

def test_sample_contains_no_personal_data_patterns(sample):
    text = "\n".join(sample["description"])
    assert not re.search(r"INDN:", text)                       # individual-name fields
    assert not re.search(r"[xX*]{4,}\d{4}", text)               # masked card numbers
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)      # email addresses
    assert not re.search(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", text)  # phone numbers


def test_macro_f1_ignores_categories_absent_from_the_test_set():
    import analysis

    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "c", "b", "b"]  # "c" is predicted but has no true examples
    s = analysis.scores(y_true, y_pred)
    assert s["accuracy"] == 0.75
    assert s["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-4)  # averaged over a and b only
