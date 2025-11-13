import sys, types, gzip, joblib, pandas as pd
from ase.io import write

# --- 1️⃣ Patch for old pandas pickles ---
import pandas.core.indexes
sys.modules['pandas.core.indexes.numeric'] = types.ModuleType('pandas.core.indexes.numeric')
setattr(sys.modules['pandas.core.indexes.numeric'], 'Int64Index', pd.Index)

# --- 2️⃣ Load the file ---
filename = "df_Al_train.pckl.gzip"  # or "df_Al_train.pickl" if not gzipped

try:
    # try gzip first
    with gzip.open(filename, "rb") as f:
        train = joblib.load(f)
except OSError:
    # fallback if not gzipped
    train = joblib.load(filename)

# --- 3️⃣ Inspect what's inside ---
print("\n=== Type and basic info ===")
print(type(train))

if isinstance(train, pd.DataFrame):
    print("\nDataFrame shape:", train.shape)
    print("Columns:", train.columns.tolist())
    print("\nFirst few rows:\n", train.head())
elif isinstance(train, list):
    print(f"\nList of length {len(train)}")
    print("First element type:", type(train[0]))
else:
    print("\nObject attributes:", dir(train))

# --- 4️⃣ Save as .extxyz ---
try:
    # Case 1: DataFrame with 'atoms' column
    if isinstance(train, pd.DataFrame) and "atoms" in train.columns:
        atoms_list = train["atoms"].to_list()
        write("df_train_Al.extxyz", atoms_list)
        print("\n✅ Saved ASE Atoms objects from DataFrame['atoms'] to df_train_Al.extxyz")

    # Case 2: List of Atoms objects directly
    elif isinstance(train, list) and hasattr(train[0], "positions"):
        write("df_train_Al.extxyz", train)
        print("\n✅ Saved ASE Atoms list to df_train_Al.extxyz")

    else:
        print("\n⚠️ Could not detect ASE Atoms automatically. "
              "Inspect 'train' to locate your atomic structures.")
except Exception as e:
    print("\n❌ Error while writing extxyz:", e)
