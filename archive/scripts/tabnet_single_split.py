# %%
import pandas as pd
import numpy  as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# %%
DATA_PATH = r'C:\Users\aweie\OneDrive - University of Houston Downtown\Documents\GitHub\CapstoneClass\entraineddropletfraction_VerticalFlow.csv'

# %%
import warnings, math, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())

TARGET_COL = None
for c in df.columns:
    lc = c.lower()
    if any(k in lc for k in ["entrain","fraction"]) or lc in ["e","entrained","entrainment","target","y"]:
        if pd.api.types.is_numeric_dtype(df[c]):
            TARGET_COL = c
            break
if TARGET_COL is None:
    for c in reversed(df.columns):
        if pd.api.types.is_numeric_dtype(df[c]):
            TARGET_COL = c
            break

print("Detected target column:", TARGET_COL)

# %%
# Define features and target
features = ['D',	'Usg',	'UsL',	'rhol',	'rhog',	'st',	'MuL',	'Mug']
target = 'e'

# %%
X = df[features]
y = df[target]

# %%
# --- Split the data ---

#Split data into training and testing sets
X = pd.DataFrame(df, columns=df.columns.difference(['e']))
y = pd.Series(df['e'], name='e')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# --- Prepare Data for TabNet ---
X_train_np, y_train_np = X_train.values, y_train.values.reshape(-1, 1)
X_test_np, y_test_np = X_test.values, y_test.values.reshape(-1, 1)


# %%
import torch
tabnet = TabNetRegressor(
        n_d=16,             # decision layer width
        n_a=16,             # attention layer width
        n_steps=5,          # number of steps
        gamma=1.5,          # relaxation parameter
        n_independent=2,    # independent layers
        n_shared=2,         # shared layers
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', # sparsemax or entmax
        verbose=1,
        seed=42
    )

    # -----------------------------
    # 3. TRAIN TABNET
    # -----------------------------
tabnet.fit(
        X_train_np, y_train_np,
        eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)],
        eval_name=['train','valid'],
        max_epochs=500,
        patience=50,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0
    )

# %%
##Evaluate the performance of the predictions using the following metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared, and Mean Absolute Percentage Error (MAPE).
preds = tabnet.predict(X_test_np).flatten()

print("R²:", r2_score(y_test_np, preds))
print("MAE:", mean_absolute_error(y_test_np, preds))
print("RMSE:", root_mean_squared_error(y_test_np, preds))
print("MAPE:", np.mean(np.abs((y_test_np - preds) / y_test_np)) * 100)

# %%



