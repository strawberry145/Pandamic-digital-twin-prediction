"""Pandemic Digital Twin prototype

This script builds a shared preprocessing and modeling pipeline for
public health surveillance data. It trains:

- outbreak status classification
- infection risk level classification
- individual daily new cases regression
- region-level next-day and next-week case forecasts

Usage:
  python pandemic_digital_twin.py
  python pandemic_digital_twin.py --csv public_health_surveillance_dataset.csv
"""

import argparse
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

ORDINAL_MAPPINGS = {
    "SES": ["Low", "Medium", "High"],
    "Immunity_Level": ["Low", "Medium", "High"],
    "Reported_Symptoms": ["Mild", "Moderate", "Severe"],
    "Population_Density": ["Low", "Medium", "High"],
    "Travel_History": ["No Travel", "Domestic", "International"],
    "Social_Activity": ["Low", "Medium", "High"],
    "Vaccination_Hesitancy": ["No", "Yes"],
    "Hospitalization_Rate": ["Low", "Medium", "High"],
    "Hospital_Capacity": ["Full", "Limited", "Available"],
    "Healthcare_Personnel_Availability": ["Scarce", "Adequate"],
    "Disease_Severity": ["Mild", "Moderate", "Severe"],
    "Hospitalization_Requirement": ["No Hospitalization", "Requires Hospitalization", "Requires ICU"],
}

NOMINAL_FEATURES = [
    "Gender", "Location", "Ethnicity", "Medical_History",
    "Diagnosis", "Testing_Results"
]

ORDINAL_FEATURES = [
    "SES", "Immunity_Level", "Reported_Symptoms", "Population_Density",
    "Travel_History", "Social_Activity", "Vaccination_Hesitancy",
    "Hospitalization_Rate", "Hospital_Capacity", "Healthcare_Personnel_Availability"
]

NUMERIC_FEATURES = [
    "Age", "Chronic_Conditions", "Vaccination_Status", "Temperature",
    "AQI", "Humidity", "Transmission_Rate", "Mortality_Rate",
    "Case_Fatality_Ratio", "Resource_Utilization", "Compliance_with_Health_Guidelines",
    "days_since_onset", "days_since_campaign", "day_of_week", "month"
]

TARGET_LABEL_MAPS = {
    "Outbreak_Status": {"No Outbreak": 0, "Emerging Outbreak": 1, "Ongoing Outbreak": 2},
    "Infection_Risk_Level": {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2},
}


def load_data(csv_path=None):
    if csv_path is None:
        csv_path = "public_health_surveillance_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        parse_dates=["Date_of_Onset", "Date_of_Data_Collection", "Vaccination_Campaign_Dates"],
        dayfirst=False,
        infer_datetime_format=True,
    )
    return df


def feature_engineering(df):
    df = df.copy()
    df["days_since_onset"] = (df["Date_of_Data_Collection"] - df["Date_of_Onset"]).dt.days.clip(lower=0)
    df["days_since_campaign"] = (
        df["Date_of_Data_Collection"] - df["Vaccination_Campaign_Dates"]
    ).dt.days.fillna(0).clip(lower=0)
    df["day_of_week"] = df["Date_of_Data_Collection"].dt.dayofweek
    df["month"] = df["Date_of_Data_Collection"].dt.month
    df["Vaccination_Status"] = df["Vaccination_Status"].fillna(0).astype(int)
    df["Chronic_Conditions"] = df["Chronic_Conditions"].fillna(0).astype(int)
    df["Compliance_with_Health_Guidelines"] = df["Compliance_with_Health_Guidelines"].fillna(0).astype(int)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())
    df["Humidity"] = df["Humidity"].fillna(df["Humidity"].median())
    df["AQI"] = df["AQI"].fillna(df["AQI"].median())
    df["Transmission_Rate"] = df["Transmission_Rate"].fillna(df["Transmission_Rate"].median())
    df["Mortality_Rate"] = df["Mortality_Rate"].fillna(df["Mortality_Rate"].median())
    df["Case_Fatality_Ratio"] = df["Case_Fatality_Ratio"].fillna(df["Case_Fatality_Ratio"].median())
    df["Resource_Utilization"] = df["Resource_Utilization"].fillna(df["Resource_Utilization"].median())
    return df


def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ordinal_transformer = Pipeline([
        (
            "encoder",
            OrdinalEncoder(
                categories=[ORDINAL_MAPPINGS[c] for c in ORDINAL_FEATURES],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        )
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("cat", categorical_transformer, NOMINAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def model_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def train_classifier(X, y, label_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    pipe = model_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[{label_name}] accuracy: {accuracy:.4f}")
    print(report)
    return pipe, X_test, y_test, y_pred


def train_regressor(X, y, label_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = model_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[{label_name}] MAE={mae:.3f}, R²={r2:.4f}")
    return pipe, X_test, y_test, y_pred


def get_individual_features(df):
    return df[NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES]


def build_region_aggregates(df):
    numeric_cols = [
        "Age", "Chronic_Conditions", "Vaccination_Status", "Temperature",
        "AQI", "Humidity", "Transmission_Rate", "Mortality_Rate",
        "Case_Fatality_Ratio", "Resource_Utilization", "Compliance_with_Health_Guidelines",
        "days_since_onset", "days_since_campaign",
    ]
    ordinal_cols = [
        "SES", "Immunity_Level", "Reported_Symptoms", "Population_Density",
        "Travel_History", "Social_Activity", "Vaccination_Hesitancy",
        "Hospitalization_Rate", "Hospital_Capacity", "Healthcare_Personnel_Availability",
    ]
    mode_cols = ordinal_cols + ["Gender", "Ethnicity", "Medical_History", "Diagnosis", "Testing_Results", "SES", "Immunity_Level"]

    agg_dict = {col: "median" for col in numeric_cols}
    agg_dict.update({"Daily_New_Cases": "median"})

    grouped = df.groupby(["Location", "Date_of_Data_Collection"]).agg(agg_dict)
    for col in mode_cols:
        grouped[col] = df.groupby(["Location", "Date_of_Data_Collection"])[col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]
        )

    grouped = grouped.sort_index()
    for lag in [1, 7]:
        grouped[f"cases_lag_{lag}"] = grouped.groupby(level=0)["Daily_New_Cases"].shift(lag)
    grouped["cases_ma_7"] = grouped.groupby(level=0)["Daily_New_Cases"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    grouped["target_next_day"] = grouped.groupby(level=0)["Daily_New_Cases"].shift(-1)
    grouped["target_next_week"] = grouped.groupby(level=0)["Daily_New_Cases"].shift(-7)

    grouped["day_of_week"] = grouped.index.get_level_values("Date_of_Data_Collection").dayofweek
    grouped["month"] = grouped.index.get_level_values("Date_of_Data_Collection").month

    grouped = grouped.dropna(subset=["target_next_day", "target_next_week"])
    return grouped.reset_index()


def get_region_features(df):
    features = [
        "Age", "Chronic_Conditions", "Vaccination_Status", "Temperature",
        "AQI", "Humidity", "Transmission_Rate", "Mortality_Rate", "Case_Fatality_Ratio",
        "Resource_Utilization", "Compliance_with_Health_Guidelines", "days_since_onset",
        "days_since_campaign", "day_of_week", "month",
        "cases_lag_1", "cases_lag_7", "cases_ma_7",
    ]
    features += ORDINAL_FEATURES + NOMINAL_FEATURES
    features = list(dict.fromkeys(features))
    return df[features]


def main():
    parser = argparse.ArgumentParser(description="Pandemic Digital Twin prototype")
    parser.add_argument("--csv", type=str, default="public_health_surveillance_dataset.csv")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_data(args.csv)
    df = feature_engineering(df)
    print(f"Dataset rows: {len(df):,}, columns: {len(df.columns)}")

    print("\nTraining individual-level models...")
    individual_df = df.dropna(subset=["Outbreak_Status", "Infection_Risk_Level", "Daily_New_Cases"]).copy()
    X_individual = get_individual_features(individual_df)
    y_outbreak = individual_df["Outbreak_Status"].map(TARGET_LABEL_MAPS["Outbreak_Status"])
    y_risk = individual_df["Infection_Risk_Level"].map(TARGET_LABEL_MAPS["Infection_Risk_Level"])
    y_cases = individual_df["Daily_New_Cases"]

    outbreak_model, _, _, _ = train_classifier(X_individual, y_outbreak, "Outbreak_Status")
    risk_model, _, _, _ = train_classifier(X_individual, y_risk, "Infection_Risk_Level")
    cases_model, _, _, _ = train_regressor(X_individual, y_cases, "Daily_New_Cases")

    print("\nPreparing region-level time-series dataset...")
    region_df = build_region_aggregates(df)
    X_region = get_region_features(region_df)

    print("\nTraining region next-day forecast...")
    y_region_next_day = region_df["target_next_day"]
    next_day_model, _, _, _ = train_regressor(X_region, y_region_next_day, "Region Next-Day Cases")

    print("\nTraining region next-week forecast...")
    y_region_next_week = region_df["target_next_week"]
    next_week_model, _, _, _ = train_regressor(X_region, y_region_next_week, "Region Next-Week Cases")

    predictions_path = "pandemic_digital_twin_predictions.csv"
    print(f"\nSaving example predictions to {predictions_path}")
    sample = individual_df.sample(min(200, len(individual_df)), random_state=42)
    sample_preds = sample.copy()
    sample_preds["predicted_outbreak"] = outbreak_model.predict(get_individual_features(sample))
    sample_preds["predicted_risk"] = risk_model.predict(get_individual_features(sample))
    sample_preds["predicted_daily_cases"] = cases_model.predict(get_individual_features(sample))
    sample_preds.to_csv(predictions_path, index=False)

    print("Done. The following models were trained:")
    print(" - Outbreak status classifier")
    print(" - Infection risk classifier")
    print(" - Individual daily cases regressor")
    print(" - Region next-day cases forecast")
    print(" - Region next-week cases forecast")
    print("Saved sample predictions to", predictions_path)


if __name__ == "__main__":
    main()
