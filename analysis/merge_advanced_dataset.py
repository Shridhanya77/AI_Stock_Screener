import pandas as pd


ADVANCED_PATH = "data/advanced_ml_dataset.csv"
OUTCOME_PATH = "data/historical_ml_dataset.csv"

OUTPUT_PATH = "data/final_ml_dataset.csv"


print("=" * 70)
print("BUILDING FINAL ML DATASET")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

advanced = pd.read_csv(
    ADVANCED_PATH
)

outcome = pd.read_csv(
    OUTCOME_PATH
)


print(
    "\nAdvanced dataset:",
    advanced.shape
)

print(
    "Outcome dataset:",
    outcome.shape
)


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

advanced["timestamp"] = pd.to_numeric(
    advanced["timestamp"],
    errors="coerce"
)

outcome["timestamp"] = pd.to_numeric(
    outcome["timestamp"],
    errors="coerce"
)


# ============================================================
# CREATE MERGE KEY
# ============================================================

advanced["timestamp"] = (
    advanced["timestamp"]
    .astype("int64")
)

outcome["timestamp"] = (
    outcome["timestamp"]
    .astype("int64")
)


# ============================================================
# SELECT OUTCOME COLUMNS
# ============================================================

outcome_columns = [
    "symbol",
    "timestamp",
    "entry_ltp",
    "exit_ltp",
    "pnl",
    "target"
]

outcome = outcome[
    outcome_columns
]


# ============================================================
# MERGE
# ============================================================

print(
    "\nMerging datasets..."
)

final = pd.merge(
    advanced,
    outcome,
    on=[
        "symbol",
        "timestamp"
    ],
    how="inner"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

final = final.drop_duplicates(
    subset=[
        "symbol",
        "timestamp"
    ]
)


# ============================================================
# SORT
# ============================================================

final = final.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# CHECK RESULT
# ============================================================

print(
    "\nFinal dataset:",
    final.shape
)

print(
    "\nTarget distribution:"
)

print(
    final["target"]
    .value_counts()
)


print(
    "\nSignal distribution:"
)

print(
    final["signal"]
    .value_counts()
)


print(
    "\nP&L:"
)

print(
    "Total:",
    final["pnl"].sum()
)

print(
    "Average:",
    final["pnl"].mean()
)


# ============================================================
# CHECK MATCHING
# ============================================================

print(
    "\nMatched rows:",
    len(final)
)

print(
    "Expected rows:",
    len(outcome)
)


# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 70)
print("FINAL DATASET CREATED")
print("=" * 70)

print(
    "\nSaved to:",
    OUTPUT_PATH
)