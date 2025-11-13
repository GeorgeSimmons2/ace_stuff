
import sys, types
import pandas as pd
import joblib

# --- patch missing old pandas module ---
import pandas.core.indexes
sys.modules['pandas.core.indexes.numeric'] = types.ModuleType('pandas.core.indexes.numeric')
# old pickles expect this class here
setattr(sys.modules['pandas.core.indexes.numeric'], 'Int64Index', pd.Index)

# --- now load safely ---
train = joblib.load("df_Al_train.pickl")

print("Loaded successfully:", type(train))

