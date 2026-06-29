import pandas as pd, numpy as np, re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import warnings; warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 120
RND = 42

# ---------------- LOAD (real transactions) ----------------
df = pd.read_csv("data/bank_transactions.csv", low_memory=False)
print(f"Raw transactions: {len(df):,} | clients: {df['client_id'].nunique()} | "
      f"dates {df['txn_date'].min()[:10]} -> {df['txn_date'].max()[:10]}")

# ---------------- LABEL CLEANING (real-world taxonomy is messy) ----------------
df["category"] = df["category"].str.strip()
df["category"] = df["category"].replace({"Bank Fee": "Bank Fees"})       # merge exact duplicate
before = df["category"].nunique()
df = df[(df["category"].notna()) & (df["category"] != "Uncategorized")].copy()                         # drop unlabelled rows
print(f"Categories: {before} -> {df['category'].nunique()} after merging duplicates & dropping 'Uncategorized'")
print(f"Labelled transactions: {len(df):,}")

# ---------------- TEXT CLEANING (strip auth codes, dates, numbers) ----------------
def clean(s):
    s = str(s).lower()
    s = re.sub(r"\d+", " ", s)            # remove numbers (auth codes, dates, embedded amounts)
    s = re.sub(r"[^a-z\s]", " ", s)       # keep letters only
    s = re.sub(r"\b\w{1,2}\b", " ", s)    # drop 1-2 char tokens
    return re.sub(r"\s+", " ", s).strip()
df["clean_desc"] = df["description"].map(clean)
print("\nExample cleaning:")
for d in df["description"].iloc[:3]:
    print(f"  '{d[:55].strip()}...'  ->  '{clean(d)}'")

# ---------------- FIG 1: class distribution (imbalanced) ----------------
counts = df["category"].value_counts()
plt.figure(figsize=(9, 6.5))
counts.plot(kind="barh", color="#4c72b0"); plt.gca().invert_yaxis()
plt.xlabel("transactions"); plt.title(f"Real-world category imbalance ({df['category'].nunique()} categories)")
plt.tight_layout(); plt.savefig("figures/01_class_distribution.png"); plt.close()
print(f"\nMost common: {', '.join(f'{k} ({v:,})' for k,v in counts.head(4).items())}")
print(f"Rarest: {', '.join(f'{k} ({v})' for k,v in counts.tail(4).items())}")

# ---------------- FIG 2: amount signature by category ----------------
df["is_credit"] = (df["amount"] > 0).astype(int)   # note: this dataset stores magnitude; sign via category context
top = counts.head(12).index
med_amt = df[df["category"].isin(top)].groupby("category")["amount"].median().reindex(top)
plt.figure(figsize=(9, 5))
med_amt.sort_values().plot(kind="barh", color="#55a868")
plt.xlabel("median transaction amount"); plt.title("Amount is a useful signal: median value by category")
plt.tight_layout(); plt.savefig("figures/02_amount_by_category.png"); plt.close()

# ---------------- FEATURES: TF-IDF(text) + amount ----------------
Xtr_df, Xte_df = train_test_split(df, test_size=0.25, stratify=df["category"], random_state=RND)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
Xtr_txt = tfidf.fit_transform(Xtr_df["clean_desc"]); Xte_txt = tfidf.transform(Xte_df["clean_desc"])
amt_scaler = StandardScaler()
Xtr_amt = amt_scaler.fit_transform(np.log1p(Xtr_df["amount"].abs().values).reshape(-1, 1))
Xte_amt = amt_scaler.transform(np.log1p(Xte_df["amount"].abs().values).reshape(-1, 1))
Xtr = hstack([Xtr_txt, csr_matrix(Xtr_amt)]); Xte = hstack([Xte_txt, csr_matrix(Xte_amt)])
ytr, yte = Xtr_df["category"].values, Xte_df["category"].values
print(f"\nVocabulary: {len(tfidf.vocabulary_):,} terms | train {Xtr.shape[0]:,} | test {Xte.shape[0]:,}")

# ---------------- MODELS ----------------
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
p_dummy = dummy.predict(Xte)
clf = LinearSVC(class_weight="balanced", C=1.0)
clf.fit(Xtr, ytr); pred = clf.predict(Xte)
print("\nPerformance (test):")
print(f"  Baseline (most-frequent)  accuracy {accuracy_score(yte,p_dummy):.3f} | macro-F1 {f1_score(yte,p_dummy,average='macro'):.3f}")
print(f"  TF-IDF + LinearSVM        accuracy {accuracy_score(yte,pred):.3f} | macro-F1 {f1_score(yte,pred,average='macro'):.3f} | weighted-F1 {f1_score(yte,pred,average='weighted'):.3f}")

# ---------------- FIG 3: per-class F1 ----------------
rep = classification_report(yte, pred, output_dict=True, zero_division=0)
f1s = pd.Series({k: v["f1-score"] for k, v in rep.items()
                if k not in ("accuracy", "macro avg", "weighted avg")}).sort_values()
plt.figure(figsize=(9, 7))
f1s.plot(kind="barh", color=["#c44e52" if v < 0.5 else "#55a868" for v in f1s])
plt.xlabel("F1 score"); plt.title("Per-category performance (red = the hard ones)")
plt.axvline(f1_score(yte, pred, average="macro"), ls="--", color="black", label="macro-F1")
plt.legend(); plt.tight_layout(); plt.savefig("figures/03_per_class_f1.png"); plt.close()
print(f"\nEasiest: {', '.join(f'{k} ({v:.2f})' for k,v in f1s.tail(4).items())}")
print(f"Hardest: {', '.join(f'{k} ({v:.2f})' for k,v in f1s.head(4).items())}")

# ---------------- FIG 4: confusion among top categories ----------------
labels = counts.head(12).index.tolist()
cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
plt.figure(figsize=(9.5, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "row-normalised"}, annot_kws={"size": 7})
plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Where the model confuses categories (top 12)")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("figures/04_confusion_matrix.png"); plt.close()

# ---------------- FIG 5: what the model learned (top tokens per category) ----------------
terms = np.array(tfidf.get_feature_names_out())
show = ["Restaurants", "Gas Stations", "Payroll", "Travel", "Supermarkets and Groceries", "Utilities"]
show = [c for c in show if c in list(clf.classes_)]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, cat in zip(axes.ravel(), show):
    i = list(clf.classes_).index(cat)
    coef = clf.coef_[i][:len(terms)]                 # text coefficients only
    topk = np.argsort(coef)[-8:]
    ax.barh(terms[topk], coef[topk], color="#4c72b0")
    ax.set_title(cat, fontsize=10); ax.tick_params(labelsize=8)
fig.suptitle("The model learned sensible merchant signals (top terms per category)", fontsize=12)
plt.tight_layout(); plt.savefig("figures/05_top_terms.png"); plt.close()
print("\nTop term per shown category:")
for cat in show:
    i = list(clf.classes_).index(cat); coef = clf.coef_[i][:len(terms)]
    print(f"  {cat:28s} -> {', '.join(terms[np.argsort(coef)[-4:]][::-1])}")

# ---------------- FIG 6: model comparison (justifying the classifier) ----------------
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
models = {"Complement NB": ComplementNB(),
          "Logistic (SGD)": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=30, random_state=RND),
          "Linear SVM": LinearSVC(class_weight="balanced")}
comp = {}
for name, m in models.items():
    m.fit(Xtr_txt, ytr); pr = m.predict(Xte_txt)
    comp[name] = (accuracy_score(yte, pr), f1_score(yte, pr, average="macro"))
comp_df = pd.DataFrame(comp, index=["accuracy", "macro_F1"]).T
print("\nModel comparison (description text only):"); print(comp_df.round(3).to_string())
plt.figure(figsize=(7, 4.5))
comp_df["macro_F1"].plot(kind="bar", color=["#8172b3", "#4c72b0", "#55a868"])
plt.ylabel("macro-F1"); plt.title("Three linear classifiers on transaction text"); plt.xticks(rotation=0)
plt.ylim(0, comp_df["macro_F1"].max()*1.15)
for i, v in enumerate(comp_df["macro_F1"]): plt.text(i, v+0.01, f"{v:.2f}", ha="center")
plt.tight_layout(); plt.savefig("figures/06_model_comparison.png"); plt.close()

# ---------------- FIG 7: confidence-coverage (auto-classify vs route to review) ----------------
margins = clf.decision_function(Xte)
part = np.partition(margins, -2, axis=1)
conf = part[:, -1] - part[:, -2]                       # top-1 minus top-2 score = decision confidence
order = np.argsort(-conf); correct = (pred == yte)[order]
coverage = np.arange(1, len(order)+1) / len(order)
acc_at_cov = np.cumsum(correct) / np.arange(1, len(order)+1)
plt.figure(figsize=(8, 5))
plt.plot(coverage*100, acc_at_cov*100, color="#55a868", lw=2)
plt.axhline(accuracy_score(yte, pred)*100, ls="--", color="black", label=f"classify everything ({accuracy_score(yte,pred):.0%})")
plt.xlabel("Coverage: % auto-classified (most confident first)"); plt.ylabel("Accuracy on auto-classified (%)")
plt.title("Auto-classify the confident majority, route the rest to review")
plt.legend(); plt.tight_layout(); plt.savefig("figures/07_confidence_coverage.png"); plt.close()
print("\nConfidence-based routing (auto-classify most-confident first):")
for cov in [0.7, 0.8, 0.9]:
    k = int(cov*len(order)); print(f"  auto-classify {cov:.0%} -> accuracy {correct[:k].mean():.1%} (route remaining {1-cov:.0%} to review)")

print("\nAll figures saved. DONE.")
