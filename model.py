import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def prepare_lap_data(laps):
    df = laps.copy()
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    df = df[
        [
            "LapNumber",
            "TyreLife",
            "Stint",
            "Compound",
            "LapTimeSeconds"
        ]
    ]

    df = df.dropna()

    if df.empty:
        return None, None, df

    q_low = df["LapTimeSeconds"].quantile(0.05)
    q_high = df["LapTimeSeconds"].quantile(0.95)

    df = df[
        (df["LapTimeSeconds"] >= q_low) &
        (df["LapTimeSeconds"] <= q_high)
    ]

    df["TyreLifeSquared"] = df["TyreLife"] ** 2
    df["LapProgress"] = df["LapNumber"] / df["LapNumber"].max()
    df["IsEarlyStint"] = (df["TyreLife"] <= 5).astype(int)
    df["IsOldTyre"] = (df["TyreLife"] >= 15).astype(int)

    feature_cols = [
        "LapNumber",
        "TyreLife",
        "TyreLifeSquared",
        "LapProgress",
        "Stint",
        "IsEarlyStint",
        "IsOldTyre",
        "Compound"
    ]

    X = pd.get_dummies(
        df[feature_cols],
        columns=["Compound"]
    )

    y = df["LapTimeSeconds"]

    return X, y, df


def train_tyre_model(laps):
    X, y, df = prepare_lap_data(laps)

    if X is None or len(df) < 8:
        return None, None, None, df

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = Ridge(
        alpha=1.0
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return model, X.columns, mae, df