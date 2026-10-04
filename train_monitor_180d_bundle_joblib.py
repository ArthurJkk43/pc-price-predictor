import argparse
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class Monitor180dForecastPipeline:
    def __init__(self, model, feature_cols, cat_cols, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--out_joblib", required=True)
    ap.add_argument("--test_weeks", type=int, default=26)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.train_parquet)
    df["date_dt"] = pd.to_datetime(df["date_dt"])

    df = df.dropna(subset=["price_now", "target_price_t_plus"]).copy()
    df = df[(df["price_now"] > 0) & (df["target_price_t_plus"] > 0)].copy()
    df["y_log_ratio"] = np.log(df["target_price_t_plus"] / df["price_now"])

    feature_cols = [
        "price_now", "ma_4w", "ma_12w", "vol_12w", "mom_4w",
        "screen_size_in", "res_w", "res_h", "refresh_hz", "response_ms",
        "is_ultrawide", "is_oled", "is_mini_led",
        "resolution_class", "panel_type", "brand",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    cat_cols = [c for c in ["resolution_class", "panel_type", "brand"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    max_date = df["date_dt"].max()
    cutoff = max_date - pd.Timedelta(weeks=args.test_weeks)
    train = df[df["date_dt"] < cutoff].copy()
    test = df[df["date_dt"] >= cutoff].copy()

    X_train = train[feature_cols].copy()
    y_train = train["y_log_ratio"].to_numpy(float)
    X_test = test[feature_cols].copy()
    y_test = test["y_log_ratio"].to_numpy(float)

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("UNKNOWN")
        X_test[c] = X_test[c].astype(str).fillna("UNKNOWN")
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    train_mask = X_train[num_cols].notna().all(axis=1) & np.isfinite(y_train)
    test_mask = X_test[num_cols].notna().all(axis=1) & np.isfinite(y_test)
    X_train = X_train.loc[train_mask]
    y_train = y_train[train_mask.to_numpy()]
    X_test = X_test.loc[test_mask]
    y_test = y_test[test_mask.to_numpy()]

    cat_idx = [feature_cols.index(c) for c in cat_cols]

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=args.seed,
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=8.0,
        verbose=200,
    )
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True)

    pred_log_ratio = model.predict(X_test).astype(float)
    pred_price = X_test["price_now"].to_numpy(float) * np.exp(pred_log_ratio)
    true_price = test.loc[X_test.index, "target_price_t_plus"].to_numpy(float)

    mae = float(np.mean(np.abs(true_price - pred_price)))
    rmse = float(np.sqrt(np.mean((true_price - pred_price) ** 2)))
    mape = float(np.mean(np.abs(true_price - pred_price) / np.maximum(true_price, 1e-9)) * 100.0)

    metrics = {"mae": mae, "rmse": rmse, "mape_pct": mape, "test_weeks": int(args.test_weeks)}
    print("[METRICS]", metrics)

    pipeline = Monitor180dForecastPipeline(model=model, feature_cols=feature_cols, cat_cols=cat_cols, price_col="price_now")
    bundle = {"pipeline": pipeline, "feature_cols": feature_cols, "metrics": metrics, "log_target": False}
    joblib.dump(bundle, args.out_joblib)
    print(f"[OK] Saved -> {args.out_joblib}")

if __name__ == "__main__":
    main()
    