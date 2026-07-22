import pandas as pd
import numpy as np

def build_features(input_path: str) -> pd.DataFrame:
    """
    Load the riders_ml dataset, engineer features, and return a new DataFrame.
    input_path: path to the CSV file containing riders_ml data.
    """

    # 1. LOAD THE DATA
    # We read the CSV file into a DataFrame.
    # This replaces 'riders_ml' from the notebook.
    df = pd.read_csv(input_path)

    # 2. RECENCY (Days since last trip)
    # Convert last_trip_time to datetime so we can do date calculations.
    df["last_trip_time"] = pd.to_datetime(df["last_trip_time"])

    # Use the most recent trip date in the dataset as a reference point.
    reference_date = df["last_trip_time"].max()

    # Recency = how many days since the rider's last trip.
    df["recency_days"] = (reference_date - df["last_trip_time"]).dt.days

    # 3. FREQUENCY (Total trips)
    # Frequency is simply the total number of trips the rider has taken.
    df["frequency_trips"] = df["total_trips"]

    # 4. MONETARY (Average fare)
    # Monetary value is the average fare per trip.
    df["monetary_avg_fare"] = df["avg_fare"]

    # 5. ENGAGEMENT SCORE
    # We combine total_sessions, avg_pages_visited, and conversion_rate
    # into a single engagement score using simple weights.
    df["engagement_score"] = (
        df["total_sessions"].fillna(0) * 0.4 +
        df["avg_pages_visited"].fillna(0) * 0.3 +
        df["conversion_rate"].fillna(0) * 0.3
    )

    # 6. LOYALTY ENCODING
    # Convert loyalty tiers into numbers so the model can use them.
    loyalty_map = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3}
    df["loyalty_encoded"] = df["loyalty_status"].map(loyalty_map)

    # 7. REFERRAL FLAG
    # 1 if the rider was referred by someone, 0 if 'Unknown'.
    df["referred_flag"] = np.where(df["referred_by"] == "Unknown", 0, 1)

    # 8. CITY ENCODING
    # Convert city names into numeric codes.
    city_map = {city: idx for idx, city in enumerate(df["city"].unique())}
    df["city_encoded"] = df["city"].map(city_map)

    # 9. AGE BUCKETS
    # Group ages into meaningful categories
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[18, 25, 35, 50, 80],
        labels=["young", "early_adult", "mid_adult", "senior"]
    )

    # 10. MISSING SESSION FLAG
    # 1 if total_sessions is missing, 0 otherwise.
    df["missing_sessions_flag"] = df["total_sessions"].isna().astype(int)

    # 11. SELECT FINAL FEATURE COLUMNS
    feature_cols = [
        "recency_days",
        "frequency_trips",
        "monetary_avg_fare",
        "engagement_score",
        "loyalty_encoded",
        "referred_flag",
        "city_encoded",
        "age",
        "age_bucket",
        "missing_sessions_flag",
        "churn_prob" # keep target in the same DataFrame for now
    ]

    df_features = df[feature_cols].copy()

    return df_features


if __name__ == "__main__":
    # Example usage:
    # Adjust the path to where your riders_ml CSV is stored.
    input_csv_path = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\riders_ml.csv"

    features_df = build_features(input_csv_path)

    # Save engineered features to a new CSV for modelling.
    output_csv_path = r"Data\riders_ml_features.csv"
    features_df.to_csv(output_csv_path, index=False)

    print(f"Feature engineering complete. Saved to {output_csv_path}")
