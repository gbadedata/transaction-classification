"""Generate data/sample_transactions.csv: a small, fully synthetic dataset with the same schema
as the original. No real transactions or people are involved; every row comes from the
templates below, so the pipeline and tests can run without the original file.

    python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_CLIENTS = 60
TXNS_PER_CLIENT = (30, 70)
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"

# category -> (description templates, median amount)
CATEGORIES = {
    "Restaurants": (["POS PURCHASE {n} MCDONALD'S F{n} {city}", "STARBUCKS STORE {n} {city}",
                     "DUNKIN #{n} {city}", "CHIPOTLE {n} ONLINE", "WENDYS #{n} {city}"], 14),
    "Supermarkets and Groceries": (["KROGER #{n} {city}", "PUBLIX SUPER MAR {n} {city}",
                                    "SAFEWAY STORE {n}", "ALDI {n} {city}", "WAL-MART GROCERY {n}"], 62),
    "Gas Stations": (["SHELL OIL {n} {city}", "KROGER FUEL #{n}", "EXXONMOBIL {n} {city}",
                      "COSTCO GAS #{n}", "CHEVRON {n} {city}"], 41),
    "Payroll": (["ACME CORP DES:PAYROLL ID:{n} PPD", "DIRECT DEP {n} PAYROLL",
                 "UBER TECHNOLOGIES PAYROLL {n}", "LYFT INC DIRECT DEP {n}", "PAYCHECK TRANSFER {n}"], 1450),
    "Utilities": (["CITY WATER UTILITY {n}", "DOMINION ENERGY BILL PAY {n}", "NATIONAL GRID ELECTRIC {n}",
                   "GAS UTILITIES ONLINE PMT {n}"], 120),
    "Travel": (["DELTA AIR {n} {city}", "UBER TRIP {n} HELP.UBER.COM", "MARRIOTT HOTEL {n} {city}",
                "AMTRAK {n} ONLINE", "UBER TECHNOLOGIES {n} {city}"], 210),
    "Loans": (["EARNIN PAYMENT {n}", "DAVE INC LOAN {n}", "BRIGIT ADVANCE REPAY {n}",
               "AUTO LOAN PMT {n}"], 180),
    "Transfer Credit": (["ONLINE TRANSFER FROM CHECKING {n}", "ZELLE TRANSFER FROM {n}",
                         "MOBILE TRANSFER CREDIT {n}", "PAYPAL TRANSFER FROM {n}"], 150),
    "ATM": (["ATM WITHDRAWAL {n} {city}", "NON-BANK ATM {n} {city}", "ATM CASH {n}"], 80),
    "Bank Fees": (["MONTHLY SERVICE FEE", "OVERDRAFT ITEM FEE {n}", "ATM FEE {n}"], 12),
    "Third Party": (["PAYPAL INST XFER {n}", "VENMO PAYMENT {n}", "CASH APP {n} PAYMENT"], 45),
    "Healthcare": (["CVS PHARMACY {n} {city}", "WALGREENS #{n}", "DENTAL CLINIC {n} {city}"], 55),
}
CITIES = ["SEATTLE WA", "AUSTIN TX", "DENVER CO", "ATLANTA GA", "COLUMBUS OH", "TAMPA FL"]


def make_sample(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(CATEGORIES)
    weights = np.array([10, 8, 6, 5, 4, 3, 5, 6, 3, 2, 6, 2], dtype=float)
    weights /= weights.sum()
    dates = pd.date_range("2023-06-01", "2023-09-30", freq="D")
    rows, txn_id = [], 0
    for client in range(1, N_CLIENTS + 1):
        for _ in range(rng.integers(*TXNS_PER_CLIENT)):
            txn_id += 1
            cat = names[rng.choice(len(names), p=weights)]
            templates, median = CATEGORIES[cat]
            desc = templates[rng.integers(len(templates))].format(
                n=rng.integers(100, 9999), city=CITIES[rng.integers(len(CITIES))])
            amount = round(float(median * rng.lognormal(0, 0.35)), 2)
            rows.append({"client_id": client, "bank_id": 1, "account_id": client, "txn_id": txn_id,
                         "txn_date": f"{dates[rng.integers(len(dates))]:%Y-%m-%d} 00:00:00",
                         "description": desc, "amount": amount, "category": cat})
    df = pd.DataFrame(rows)
    # Real feeds are noisy: aggregator labels are sometimes wrong, and some merchants
    # (Uber, Kroger, transfers) legitimately span categories.
    noisy = df.sample(frac=0.06, random_state=seed).index
    df.loc[noisy, "category"] = [names[i] for i in rng.integers(len(names), size=len(noisy))]
    # Exercise the label-cleaning step: a duplicate label and unlabelled rows.
    fee = df.index[df["category"] == "Bank Fees"]
    df.loc[fee[: len(fee) // 3], "category"] = "Bank Fee"
    df.loc[df.sample(frac=0.03, random_state=seed).index, "category"] = "Uncategorized"
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample = make_sample()
    sample.to_csv(OUT, index=False)
    print(f"Wrote {len(sample):,} synthetic transactions for {sample['client_id'].nunique()} clients to {OUT}")
