"""Core steps of the transaction-categorisation pipeline.

Kept free of file I/O and plotting so every step can be unit-tested on small inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = ["client_id", "txn_date", "description", "amount", "category"]
LABEL_MERGES = {"Bank Fee": "Bank Fees"}  # exact duplicate in the source taxonomy
DROP_LABELS = {"Uncategorized"}
SEED = 42

# TF-IDF settings used for every model in the analysis.
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_FEATURES = 30_000


def validate_columns(df: pd.DataFrame) -> None:
    """Fail early, with a clear message, if the input file has the wrong schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {', '.join(missing)}")


def clean_text(s: object) -> str:
    """Lower-case and strip numbers, punctuation and 1-2 character tokens.

    Auth codes, dates and embedded amounts are noise for categorisation; merchant words
    are the signal.
    """
    s = str(s).lower()
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\b\w{1,2}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate labels, drop unlabelled rows, and add the cleaned description."""
    out = df.copy()
    out["category"] = out["category"].str.strip().replace(LABEL_MERGES)
    out = out[out["category"].notna() & ~out["category"].isin(DROP_LABELS)].copy()
    out["clean_desc"] = out["description"].map(clean_text)
    return out


def random_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = SEED):
    """Stratified split by transaction. The same customer can appear on both sides."""
    return train_test_split(df, test_size=test_size, stratify=df["category"], random_state=seed)


def client_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = SEED):
    """Split by customer, so every test transaction belongs to a customer never seen in training.

    This is the honest estimate for a categoriser that will meet new customers.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["client_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


@dataclass
class Features:
    tfidf: TfidfVectorizer
    X_train_text: spmatrix
    X_test_text: spmatrix
    X_train: spmatrix
    X_test: spmatrix
    y_train: np.ndarray
    y_test: np.ndarray


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> Features:
    """TF-IDF over cleaned descriptions plus a scaled log-amount column.

    The vectoriser is fitted with fit_transform on the training rows. Fitting and transforming
    in two steps gives values that differ by about 1e-16, which is enough to flip a few
    borderline predictions, so fit_transform is used to keep published results reproducible.
    """
    tfidf = TfidfVectorizer(
        ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_features=MAX_FEATURES, sublinear_tf=True
    )
    X_train_text = tfidf.fit_transform(train["clean_desc"])
    X_test_text = tfidf.transform(test["clean_desc"])
    scaler = StandardScaler()
    amt_train = scaler.fit_transform(np.log1p(train["amount"].abs().to_numpy()).reshape(-1, 1))
    amt_test = scaler.transform(np.log1p(test["amount"].abs().to_numpy()).reshape(-1, 1))
    return Features(
        tfidf=tfidf,
        X_train_text=X_train_text,
        X_test_text=X_test_text,
        X_train=hstack([X_train_text, csr_matrix(amt_train)]),
        X_test=hstack([X_test_text, csr_matrix(amt_test)]),
        y_train=train["category"].to_numpy(),
        y_test=test["category"].to_numpy(),
    )


def decision_confidence(margins: np.ndarray) -> np.ndarray:
    """Gap between the top two class scores for each row: larger means more confident."""
    margins = np.asarray(margins, dtype=float)
    if margins.ndim != 2 or margins.shape[1] < 2:
        raise ValueError("margins must be a 2-D array with at least two classes")
    top_two = np.partition(margins, -2, axis=1)
    return top_two[:, -1] - top_two[:, -2]


def accuracy_at_coverage(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident `coverage` share of predictions."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    order = np.argsort(-np.asarray(confidence))
    k = int(coverage * len(order))
    if k == 0:
        raise ValueError("coverage selects no predictions")
    return float(np.asarray(correct)[order][:k].mean())


def coverage_curve(confidence: np.ndarray, correct: np.ndarray):
    """Coverage (0-1] against accuracy on the auto-classified share, most confident first."""
    order = np.argsort(-np.asarray(confidence))
    ordered = np.asarray(correct)[order]
    n = np.arange(1, len(order) + 1)
    return n / len(order), np.cumsum(ordered) / n
