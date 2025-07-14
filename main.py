from env_setup import *
from imports import *
from data_loader import X, y
from objective import mse_objective
from esca import ESCA
from store import storedata, lb, ub, dim

global X_train, y_train



# define mse_objective *after* X_train, y_train come into scope
def mse_objective(params):
    learning_rate = max(0.001, min(0.35, params[0]))
    num_leaves = max(10, int(round(params[1])))
    max_depth = max(3, int(round(params[2])))
    n_estimators = int(params[3])

    model = lgb.LGBMRegressor(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        objective='regression',
        n_estimators=n_estimators,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train.ravel())
    preds = model.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    return mse


Name = 'ESCA-LightGBM'
rand_state = 4
kf = KFold(n_splits=5, shuffle=True, random_state=rand_state)
fold = 1
print(f"\n=== K fold Fold ===")
print("\nMode\tR2_train\tR2_test\tRMSE_train\tRMSE_test\tMSE_train\tMSE_test\tME_train\tME_test\tRAE_train\tRAE_test")
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    result = ESCA(mse_objective, lb, ub, dim, 30, 50)
    storedata(result)
    best_params = result.bestIndividual

    # Safely decode best parameters
    best_learning_rate = max(0.001, min(0.35, best_params[0]))
    best_num_leaves = max(10, int(round(best_params[1])))
    best_max_depth = max(3, int(round(best_params[2])))
    best_n_estimators = int(best_params[3])

    # Final model training
    final_model = lgb.LGBMRegressor(
        learning_rate=best_learning_rate,
        num_leaves=best_num_leaves,
        max_depth=best_max_depth,
        objective='regression',
        n_estimators=best_n_estimators,
        random_state=2,
        verbosity=-1
    )

    final_model.fit(X_train, y_train.ravel())
    train_preds = final_model.predict(X_train)
    test_preds = final_model.predict(X_test)

    e_train = RegressionMetric(y_train, train_preds, decimal=6)
    e_test  = RegressionMetric(y_test,  test_preds,  decimal=6)

    print(f"{Name}",
          f"{e_train.R2():.6f}",
          f"{e_test.R2():.6f}",
          f"{e_train.RMSE():.6f}",
          f"{e_test.RMSE():.6f}",
          f"{e_train.MSE():.6f}",
          f"{e_test.MSE():.6f}",
          f"{e_train.ME():.6f}",
          f"{e_test.ME():.6f}",
          f"{e_train.RAE():.6f}",
          f"{e_test.RAE():.6f}",
          sep="\t"
    )
