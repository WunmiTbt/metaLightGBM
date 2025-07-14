import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Objective function with enforced safety
def mse_objective(params):
    learning_rate = max(0.001, min(0.2, params[0]))
    num_leaves = max(2, int(round(params[1])))
    max_depth = max(1, int(round(params[2])))
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
