"""Categorise bank-transaction descriptions and evaluate the result.

Usage:
    python analysis.py --data data/sample_transactions.csv            # synthetic sample, runs in seconds
    python analysis.py --data data/bank_transactions.csv --figures figures   # full dataset (not redistributed)

Writes figures to --figures and a metrics file to --results.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from txnclass.pipeline import (
    SEED,
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

COVERAGES = (0.7, 0.8, 0.9)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/bank_transactions.csv", help="input CSV")
    p.add_argument("--figures", default="out/figures", help="directory for the charts")
    p.add_argument("--results", default="out/metrics.json", help="path for the metrics JSON")
    return p.parse_args(argv)


def scores(y_true, y_pred) -> dict:
    """Accuracy and macro-F1, averaging only over categories present in y_true.

    sklearn's default averages over every label seen in y_true or y_pred, so a category with no
    test examples that the model sometimes predicts would add a zero. With a random split every
    category is in the test set and both definitions agree.
    """
    present = sorted(set(y_true))
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=present, average="macro")), 4)}


def fit_svm(X, y):
    return LinearSVC(class_weight="balanced", C=1.0).fit(X, y)


def save(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / name)
    plt.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}\n"
              "The original dataset is not redistributed. Run on the synthetic sample with:\n"
              "  python analysis.py --data data/sample_transactions.csv", file=sys.stderr)
        return 2
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120
    metrics: dict = {"data": data_path.name}

    # ---------------- LOAD + LABEL CLEANING ----------------
    raw = pd.read_csv(data_path, low_memory=False)
    validate_columns(raw)
    print(f"Raw transactions: {len(raw):,} | clients: {raw['client_id'].nunique()} | "
          f"dates {str(raw['txn_date'].min())[:10]} -> {str(raw['txn_date'].max())[:10]}")
    before = raw["category"].str.strip().replace({"Bank Fee": "Bank Fees"}).nunique()
    df = clean_labels(raw)
    print(f"Categories: {before} -> {df['category'].nunique()} after merging duplicates & dropping 'Uncategorized'")
    print(f"Labelled transactions: {len(df):,}")
    metrics.update(raw_transactions=len(raw), labelled_transactions=len(df),
                   clients=int(raw["client_id"].nunique()), categories=int(df["category"].nunique()))

    example = "POS PURCHASE 0412 STARBUCKS #1234 SEATTLE WA 06/14"
    print(f"\nExample cleaning: '{example}' -> '{clean_text(example)}'")

    # ---------------- FIG 1: class distribution ----------------
    counts = df["category"].value_counts()
    plt.figure(figsize=(9, 6.5))
    counts.plot(kind="barh", color="#4c72b0")
    plt.gca().invert_yaxis()
    plt.xlabel("transactions")
    plt.title(f"Real-world category imbalance ({df['category'].nunique()} categories)")
    save(fig_dir, "01_class_distribution.png")
    print(f"\nMost common: {', '.join(f'{k} ({v:,})' for k, v in counts.head(4).items())}")
    print(f"Rarest: {', '.join(f'{k} ({v})' for k, v in counts.tail(4).items())}")

    # ---------------- FIG 2: amount by category ----------------
    top = counts.head(12).index
    med_amt = df[df["category"].isin(top)].groupby("category")["amount"].median().reindex(top)
    plt.figure(figsize=(9, 5))
    med_amt.sort_values().plot(kind="barh", color="#55a868")
    plt.xlabel("median transaction amount")
    plt.title("Amount is a useful signal: median value by category")
    save(fig_dir, "02_amount_by_category.png")

    # ---------------- MAIN MODEL: random split (the original evaluation) ----------------
    train, test = random_split(df)
    feats = build_features(train, test)
    print(f"\nVocabulary: {len(feats.tfidf.vocabulary_):,} terms | train {feats.X_train.shape[0]:,} "
          f"| test {feats.X_test.shape[0]:,}")
    p_dummy = DummyClassifier(strategy="most_frequent").fit(feats.X_train, feats.y_train).predict(feats.X_test)
    clf = fit_svm(feats.X_train, feats.y_train)
    pred = clf.predict(feats.X_test)
    yte = feats.y_test
    metrics["random_split"] = {"baseline": scores(yte, p_dummy), "svm_text_amount": scores(yte, pred)}
    print("\nPerformance (test, random split by transaction):")
    print(f"  Baseline (most-frequent)  accuracy {accuracy_score(yte, p_dummy):.3f} | "
          f"macro-F1 {f1_score(yte, p_dummy, average='macro'):.3f}")
    print(f"  TF-IDF + LinearSVM        accuracy {accuracy_score(yte, pred):.3f} | "
          f"macro-F1 {f1_score(yte, pred, average='macro'):.3f} | "
          f"weighted-F1 {f1_score(yte, pred, average='weighted'):.3f}")

    # ---------------- FIG 3: per-class F1 ----------------
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    f1s = pd.Series({k: v["f1-score"] for k, v in rep.items()
                     if k not in ("accuracy", "macro avg", "weighted avg")}).sort_values()
    plt.figure(figsize=(9, 7))
    f1s.plot(kind="barh", color=["#c44e52" if v < 0.5 else "#55a868" for v in f1s])
    plt.xlabel("F1 score")
    plt.title("Per-category performance (red = the hard ones)")
    plt.axvline(f1_score(yte, pred, average="macro"), ls="--", color="black", label="macro-F1")
    plt.legend()
    save(fig_dir, "03_per_class_f1.png")
    print(f"\nEasiest: {', '.join(f'{k} ({v:.2f})' for k, v in f1s.tail(4).items())}")
    print(f"Hardest: {', '.join(f'{k} ({v:.2f})' for k, v in f1s.head(4).items())}")

    # ---------------- FIG 4: confusion among top categories ----------------
    labels = counts.head(12).index.tolist()
    cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
    plt.figure(figsize=(9.5, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "row-normalised"}, annot_kws={"size": 7})
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("Where the model confuses categories (top 12)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save(fig_dir, "04_confusion_matrix.png")

    # ---------------- FIG 6: model comparison on text only ----------------
    models = {"Complement NB": ComplementNB(),
              "Logistic (SGD)": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=30,
                                              random_state=SEED),
              "Linear SVM": LinearSVC(class_weight="balanced")}
    comp = {}
    for name, model in models.items():
        model.fit(feats.X_train_text, feats.y_train)
        comp[name] = scores(yte, model.predict(feats.X_test_text))
    comp_df = pd.DataFrame(comp).T.rename(columns={"macro_f1": "macro_F1"})
    metrics["random_split"]["text_only_comparison"] = comp
    print("\nModel comparison (description text only):")
    print(comp_df.round(3).to_string())
    plt.figure(figsize=(7, 4.5))
    comp_df["macro_F1"].plot(kind="bar", color=["#8172b3", "#4c72b0", "#55a868"])
    plt.ylabel("macro-F1")
    plt.title("Three linear classifiers on transaction text")
    plt.xticks(rotation=0)
    plt.ylim(0, comp_df["macro_F1"].max() * 1.15)
    for i, v in enumerate(comp_df["macro_F1"]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    save(fig_dir, "06_model_comparison.png")

    # ---------------- FIG 7: confidence routing ----------------
    conf = decision_confidence(clf.decision_function(feats.X_test))
    correct = pred == yte
    cov, acc = coverage_curve(conf, correct)
    plt.figure(figsize=(8, 5))
    plt.plot(cov * 100, acc * 100, color="#55a868", lw=2)
    plt.axhline(accuracy_score(yte, pred) * 100, ls="--", color="black",
                label=f"classify everything ({accuracy_score(yte, pred):.0%})")
    plt.xlabel("Coverage: % auto-classified (most confident first)")
    plt.ylabel("Accuracy on auto-classified (%)")
    plt.title("Auto-classify the confident majority, route the rest to review")
    plt.legend()
    save(fig_dir, "07_confidence_coverage.png")
    metrics["random_split"]["routing"] = {}
    print("\nConfidence-based routing (auto-classify most-confident first):")
    for c in COVERAGES:
        a = accuracy_at_coverage(conf, correct, c)
        metrics["random_split"]["routing"][f"{c:.0%}"] = round(a, 4)
        print(f"  auto-classify {c:.0%} -> accuracy {a:.1%} (route remaining {1 - c:.0%} to review)")

    # ---------------- UNSEEN CUSTOMERS: split by client ----------------
    # The random split lets one customer's recurring transactions (and the names in them)
    # appear in both train and test, so the model can score well by memorising customers.
    # Splitting by customer measures what matters for a categoriser: new customers.
    g_train, g_test = client_split(df)
    assert set(g_train["client_id"]).isdisjoint(g_test["client_id"])
    g = build_features(g_train, g_test)
    g_clf = fit_svm(g.X_train, g.y_train)
    g_pred = g_clf.predict(g.X_test)
    g_conf = decision_confidence(g_clf.decision_function(g.X_test))
    g_correct = g_pred == g.y_test
    g_scores = scores(g.y_test, g_pred)
    in_test = sorted(set(g.y_test))
    g_rep = classification_report(g.y_test, g_pred, labels=in_test, output_dict=True, zero_division=0)
    test_support = pd.Series(g.y_test).value_counts()
    per_class = pd.DataFrame({
        "random_split": {k: rep[k]["f1-score"] for k in in_test if k in rep},
        "unseen_customers": {k: g_rep[k]["f1-score"] for k in in_test},
        "unseen_test_rows": test_support.reindex(in_test),
    })
    per_class["drop"] = per_class["random_split"] - per_class["unseen_customers"]
    customers_per_category = df.groupby("category")["client_id"].nunique().sort_values()
    single_customer = customers_per_category[customers_per_category == 1]
    metrics["client_split"] = {
        "train_customers": int(g_train["client_id"].nunique()),
        "test_customers": int(g_test["client_id"].nunique()),
        "test_transactions": int(len(g_test)),
        "categories_in_test": len(in_test),
        "categories_without_test_customers": sorted(set(counts.index) - set(in_test)),
        "svm_text_amount": g_scores,
        "routing": {f"{c:.0%}": round(accuracy_at_coverage(g_conf, g_correct, c), 4) for c in COVERAGES},
    }
    metrics["per_class_f1_on_test_categories"] = per_class.round(3).to_dict(orient="index")
    metrics["single_customer_categories"] = {k: int(counts[k]) for k in single_customer.index}
    print(f"\nUnseen-customer split: {g_train['client_id'].nunique()} training customers, "
          f"{g_test['client_id'].nunique()} test customers ({len(g_test):,} transactions, "
          f"{len(in_test)} of {len(counts)} categories present)")
    print(f"  TF-IDF + LinearSVM        accuracy {g_scores['accuracy']:.3f} | "
          f"macro-F1 {g_scores['macro_f1']:.3f} (over categories present in the test customers)")
    for c in COVERAGES:
        print(f"  auto-classify {c:.0%} -> accuracy {accuracy_at_coverage(g_conf, g_correct, c):.1%}")
    print("Largest per-category drops for new customers (categories with 50+ test rows):")
    big = per_class[per_class["unseen_test_rows"] >= 50].sort_values("drop", ascending=False)
    for k, r in big.head(6).iterrows():
        print(f"  {k:28s} {r['random_split']:.2f} -> {r['unseen_customers']:.2f} "
              f"({int(r['unseen_test_rows']):,} test rows)")
    if len(single_customer):
        print("Categories whose every row comes from a single customer: "
              + ", ".join(f"{k} ({counts[k]:,} rows)" for k in single_customer.index))

    # ---------------- FIG 5: random vs unseen-customer evaluation ----------------
    comparison = pd.DataFrame({
        "Random split\n(by transaction)": metrics["random_split"]["svm_text_amount"],
        "Unseen customers\n(by customer)": metrics["client_split"]["svm_text_amount"],
    }).T.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"})
    ax = comparison.plot(kind="bar", figsize=(7, 4.8), color=["#4c72b0", "#55a868"], rot=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Same model, two evaluations: new customers are harder")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)
    save(fig_dir, "05_split_comparison.png")

    results_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nFigures saved to {fig_dir}/, metrics to {results_path}. DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
