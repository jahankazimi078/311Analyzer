from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.exceptions import ConvergenceWarning

from geo import clean_zip_series


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYTIC_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_analytic.parquet"
NLP_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_issue_subtypes.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "analytics"
MODEL_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_resolution_bucket_model_metrics.parquet"
PREDICTIONS_PATH = OUTPUT_DIR / "requests_2025_2026_resolution_bucket_predictions.parquet"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "requests_2025_2026_resolution_bucket_feature_importance.parquet"
ERROR_SLICES_PATH = OUTPUT_DIR / "requests_2025_2026_resolution_bucket_error_slices.parquet"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "requests_2025_2026_resolution_bucket_confusion_matrix.parquet"

TARGET_COLUMN = "resolution_bucket"
TRAIN_YEAR = 2025
TEST_YEAR = 2026
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 350_000
TEST_SAMPLE_SIZE = 200_000
TOP_ERROR_SLICE_VALUES = 25
MIN_ERROR_SLICE_ROWS = 1_000
ENCODER_MIN_FREQUENCY = 100
MODEL_NAME_ORDER = [
    "most_frequent_baseline",
    "multinomial_logistic",
    "hist_gradient_boosting",
]
FEATURE_SET_ORDER = ["shared", "post_routing", "intake_only"]
RESOLUTION_BUCKET_ORDER = ["<1 day", "1-3 days", "3-7 days", "7-30 days", "30+ days"]
LOGISTIC_KWARGS = {
    "max_iter": 800,
    "solver": "saga",
    "C": 0.35,
    "class_weight": "balanced",
    "tol": 3e-3,
    "random_state": RANDOM_STATE,
}
TREE_KWARGS = {
    "learning_rate": 0.08,
    "max_iter": 200,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 200,
    "random_state": RANDOM_STATE,
}

POST_ROUTING_FEATURE_COLUMNS = [
    "complaint_type",
    "descriptor",
    "borough",
    "agency",
    "incident_zip_clean",
    "community_board",
    "council_district",
    "created_month",
    "created_weekday",
    "created_hour",
    "created_season",
    "issue_family",
    "issue_subtype",
    "subtype_modeled_flag",
]
INTAKE_ONLY_FEATURE_COLUMNS = [
    column for column in POST_ROUTING_FEATURE_COLUMNS if column != "agency"
]
FEATURE_SET_COLUMNS = {
    "post_routing": POST_ROUTING_FEATURE_COLUMNS,
    "intake_only": INTAKE_ONLY_FEATURE_COLUMNS,
}
TREE_MAX_LEVELS = {
    "descriptor": 250,
    "incident_zip_clean": 250,
}


def load_predictive_analysis_frame() -> pd.DataFrame:
    if not ANALYTIC_PATH.exists():
        raise FileNotFoundError(f"Missing analytic parquet: {ANALYTIC_PATH}")
    if not NLP_PATH.exists():
        raise FileNotFoundError(f"Missing NLP parquet: {NLP_PATH}")

    analytic_columns = [
        "unique_key",
        "created_date",
        "created_year",
        "created_month",
        "created_weekday",
        "created_hour",
        "created_season",
        "agency",
        "complaint_type",
        "descriptor",
        "incident_zip",
        "community_board",
        "council_district",
        "borough",
        TARGET_COLUMN,
    ]
    nlp_columns = [
        "unique_key",
        "issue_family",
        "issue_subtype",
        "subtype_modeled_flag",
    ]

    analytic_df = pd.read_parquet(ANALYTIC_PATH, columns=analytic_columns)
    nlp_df = pd.read_parquet(NLP_PATH, columns=nlp_columns)
    df = analytic_df.merge(nlp_df, on="unique_key", how="left", validate="one_to_one")

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["created_year"] = pd.to_numeric(df["created_year"], errors="coerce").astype("Int64")
    df["created_month"] = pd.to_numeric(df["created_month"], errors="coerce").astype("Int64")
    df["created_hour"] = pd.to_numeric(df["created_hour"], errors="coerce").astype("Int64")
    df["complaint_type"] = df["complaint_type"].astype("string").fillna("Unknown")
    df["descriptor"] = df["descriptor"].astype("string").fillna("Unknown")
    df["borough"] = df["borough"].astype("string").str.upper().fillna("Unknown")
    df["agency"] = df["agency"].astype("string").str.upper().fillna("Unknown")
    df["created_season"] = df["created_season"].astype("string").fillna("Unknown")
    df["created_weekday"] = df["created_weekday"].astype("string").fillna("Unknown")
    df["community_board"] = df["community_board"].astype("string").fillna("Unknown")
    df["council_district"] = df["council_district"].astype("string").fillna("Unknown")
    df["incident_zip_clean"] = clean_zip_series(df["incident_zip"]).fillna("Unknown")
    df["issue_family"] = df["issue_family"].fillna(df["complaint_type"])
    df["issue_subtype"] = df["issue_subtype"].fillna("not_modeled")
    df["subtype_modeled_flag"] = df["subtype_modeled_flag"].fillna(False)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("string")

    return df


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    usable = df.loc[df[TARGET_COLUMN].notna() & df["created_year"].isin([TRAIN_YEAR, TEST_YEAR])].copy()
    usable["subtype_modeled_flag"] = np.where(usable["subtype_modeled_flag"], "modeled", "not_modeled")
    for column in POST_ROUTING_FEATURE_COLUMNS:
        usable[column] = usable[column].astype("string").fillna("Unknown")
    return usable


def stratified_sample(df: pd.DataFrame, max_rows: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if max_rows <= 0 or len(df.index) <= max_rows:
        return df.copy()

    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    sampled_frames: list[pd.DataFrame] = []
    allocated = 0

    for target_value, count in target_counts.items():
        target_df = df.loc[df[TARGET_COLUMN].eq(target_value)]
        target_max = max(1, int(round(max_rows * count / len(df.index))))
        target_max = min(target_max, len(target_df.index))
        allocated += target_max
        sampled_frames.append(target_df.sample(target_max, random_state=random_state))

    sampled = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled.index) > max_rows:
        sampled = sampled.sample(max_rows, random_state=random_state)
    elif len(sampled.index) < max_rows:
        remaining = df.loc[~df["unique_key"].isin(sampled["unique_key"])].copy()
        top_up = min(max_rows - len(sampled.index), len(remaining.index))
        if top_up > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(top_up, random_state=random_state)],
                ignore_index=True,
            )
    return sampled.reset_index(drop=True)


def train_test_split_frame(
    df: pd.DataFrame,
    train_sample_size: int = TRAIN_SAMPLE_SIZE,
    test_sample_size: int = TEST_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = stratified_sample(df.loc[df["created_year"].eq(TRAIN_YEAR)].copy(), max_rows=train_sample_size)
    test_df = stratified_sample(df.loc[df["created_year"].eq(TEST_YEAR)].copy(), max_rows=test_sample_size)
    if train_df.empty or test_df.empty:
        raise ValueError("Phase 8 train/test frame is empty after filtering.")
    return train_df, test_df


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=ENCODER_MIN_FREQUENCY),
                feature_columns,
            )
        ]
    )


def collapse_sparse_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_levels_by_column: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_levels_by_column = max_levels_by_column or TREE_MAX_LEVELS
    train_df = train_df.copy()
    test_df = test_df.copy()

    for column, max_levels in max_levels_by_column.items():
        if column not in train_df.columns:
            continue
        keep_values = train_df[column].value_counts().head(max_levels).index
        train_df[column] = train_df[column].where(train_df[column].isin(keep_values), other="other")
        test_df[column] = test_df[column].where(test_df[column].isin(keep_values), other="other")
    return train_df, test_df


def encode_tree_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    return encoder.fit_transform(train_x), encoder.transform(test_x)


def balanced_sample_weight(target: pd.Series) -> np.ndarray:
    class_weights = target.value_counts(normalize=True).rdiv(1.0)
    return target.map(class_weights).to_numpy(dtype=float)


def ordered_categorical(series: pd.Series, order: list[str]) -> pd.Categorical:
    present = [value for value in order if value in series.astype("string").dropna().unique()]
    extras = sorted(set(series.astype("string").dropna().unique()) - set(present))
    return pd.Categorical(series, categories=[*present, *extras], ordered=True)


def evaluate_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    classes: np.ndarray,
    model_name: str,
    feature_set: str,
    train_rows: int,
    test_rows: int,
) -> pd.DataFrame:
    rows = [
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "metric_scope": "overall",
            "target_class": pd.NA,
            "metric": "accuracy",
            "metric_value": float(accuracy_score(actual, predicted)),
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
        },
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "metric_scope": "overall",
            "target_class": pd.NA,
            "metric": "macro_f1",
            "metric_value": float(f1_score(actual, predicted, average="macro")),
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
        },
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "metric_scope": "overall",
            "target_class": pd.NA,
            "metric": "weighted_f1",
            "metric_value": float(f1_score(actual, predicted, average="weighted")),
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
        },
    ]

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=classes,
        zero_division=0,
    )
    for target_class, precision_value, recall_value, f1_value, support_value in zip(
        classes,
        precision,
        recall,
        f1,
        support,
    ):
        rows.extend(
            [
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "metric_scope": "class",
                    "target_class": str(target_class),
                    "metric": "precision",
                    "metric_value": float(precision_value),
                    "train_rows": int(train_rows),
                    "test_rows": int(test_rows),
                },
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "metric_scope": "class",
                    "target_class": str(target_class),
                    "metric": "recall",
                    "metric_value": float(recall_value),
                    "train_rows": int(train_rows),
                    "test_rows": int(test_rows),
                },
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "metric_scope": "class",
                    "target_class": str(target_class),
                    "metric": "f1",
                    "metric_value": float(f1_value),
                    "train_rows": int(train_rows),
                    "test_rows": int(test_rows),
                },
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "metric_scope": "class",
                    "target_class": str(target_class),
                    "metric": "support",
                    "metric_value": float(support_value),
                    "train_rows": int(train_rows),
                    "test_rows": int(test_rows),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_feature_importance(
    model: Pipeline,
    classes: np.ndarray,
    model_name: str,
    feature_set: str,
) -> pd.DataFrame:
    encoder = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = encoder.get_feature_names_out()
    coefficients = classifier.coef_
    mean_abs = np.abs(coefficients).mean(axis=0)
    strongest_class_index = np.abs(coefficients).argmax(axis=0)

    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "feature": feature_names,
            "feature_group": [name.split("__", 1)[-1].split("_", 1)[0] for name in feature_names],
            "mean_abs_coefficient": mean_abs.astype(float),
            "max_abs_coefficient": np.abs(coefficients).max(axis=0).astype(float),
            "strongest_class": classes[strongest_class_index].astype(str),
        }
    )
    importance["feature_set"] = ordered_categorical(importance["feature_set"], FEATURE_SET_ORDER)
    importance["model_name"] = ordered_categorical(importance["model_name"], MODEL_NAME_ORDER)
    importance["strongest_class"] = ordered_categorical(importance["strongest_class"], RESOLUTION_BUCKET_ORDER)
    return importance.sort_values(
        ["feature_set", "model_name", "mean_abs_coefficient", "feature"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def build_error_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    slice_columns = ["complaint_type", "agency", "borough", "issue_family"]
    frames: list[pd.DataFrame] = []

    for column in slice_columns:
        for feature_set in predictions["feature_set"].dropna().unique().tolist():
            subset = predictions.loc[predictions["feature_set"].eq(feature_set)].copy()
            value_counts = subset[column].value_counts().head(TOP_ERROR_SLICE_VALUES)
            eligible_values = value_counts.index.tolist()
            sliced = subset.loc[subset[column].isin(eligible_values)].copy()
            summary = (
                sliced.groupby(column, dropna=False, observed=True)
                .agg(
                    complaints=("unique_key", "size"),
                    accuracy=("correct_prediction_flag", "mean"),
                    median_confidence=("predicted_probability", "median"),
                    top_actual_bucket=(TARGET_COLUMN, lambda series: series.mode().iloc[0] if not series.mode().empty else "Unknown"),
                    top_predicted_bucket=("predicted_resolution_bucket", lambda series: series.mode().iloc[0] if not series.mode().empty else "Unknown"),
                )
                .reset_index()
                .rename(columns={column: "segment_value"})
            )
            summary = summary.loc[summary["complaints"].ge(MIN_ERROR_SLICE_ROWS)].copy()
            summary["model_name"] = "multinomial_logistic"
            summary["feature_set"] = feature_set
            summary["segment_column"] = column
            frames.append(summary)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(
        ["feature_set", "segment_column", "accuracy", "complaints"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def build_confusion_matrix_frame(
    predictions: pd.DataFrame,
    classes: np.ndarray,
    model_name: str,
    feature_set: str,
) -> pd.DataFrame:
    matrix = confusion_matrix(
        predictions[TARGET_COLUMN],
        predictions["predicted_resolution_bucket"],
        labels=classes,
    )
    rows = []
    for actual_index, actual_bucket in enumerate(classes):
        actual_total = int(matrix[actual_index].sum())
        for predicted_index, predicted_bucket in enumerate(classes):
            count = int(matrix[actual_index, predicted_index])
            rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "actual_resolution_bucket": str(actual_bucket),
                    "predicted_resolution_bucket": str(predicted_bucket),
                    "complaints": count,
                    "actual_bucket_total": actual_total,
                    "actual_bucket_share": float(count / actual_total) if actual_total else np.nan,
                }
            )
    confusion = pd.DataFrame(rows)
    confusion["feature_set"] = ordered_categorical(confusion["feature_set"], FEATURE_SET_ORDER)
    confusion["model_name"] = ordered_categorical(confusion["model_name"], MODEL_NAME_ORDER)
    confusion["actual_resolution_bucket"] = ordered_categorical(
        confusion["actual_resolution_bucket"], RESOLUTION_BUCKET_ORDER
    )
    confusion["predicted_resolution_bucket"] = ordered_categorical(
        confusion["predicted_resolution_bucket"], RESOLUTION_BUCKET_ORDER
    )
    return confusion.sort_values(
        ["feature_set", "model_name", "actual_resolution_bucket", "predicted_resolution_bucket"]
    ).reset_index(drop=True)


def fit_logistic_benchmark(
    feature_columns: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: str,
) -> tuple[Pipeline, np.ndarray, np.ndarray, pd.DataFrame]:
    logistic_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_columns)),
            ("model", LogisticRegression(**LOGISTIC_KWARGS)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        logistic_model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    logistic_predictions = logistic_model.predict(test_df[feature_columns])
    logistic_probabilities = logistic_model.predict_proba(test_df[feature_columns])
    logistic_classes = logistic_model.named_steps["model"].classes_
    logistic_metrics = evaluate_predictions(
        actual=test_df[TARGET_COLUMN],
        predicted=logistic_predictions,
        classes=logistic_classes,
        model_name="multinomial_logistic",
        feature_set=feature_set,
        train_rows=len(train_df.index),
        test_rows=len(test_df.index),
    )
    return logistic_model, logistic_predictions, logistic_probabilities, logistic_metrics


def fit_tree_benchmark(
    feature_columns: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_x = train_df[feature_columns].copy()
    test_x = test_df[feature_columns].copy()
    train_x, test_x = collapse_sparse_categories(train_x, test_x)
    train_x_encoded, test_x_encoded = encode_tree_features(train_x, test_x)

    tree_model = HistGradientBoostingClassifier(
        **TREE_KWARGS,
        categorical_features=[True] * len(feature_columns),
    )
    tree_model.fit(
        train_x_encoded,
        train_df[TARGET_COLUMN],
        sample_weight=balanced_sample_weight(train_df[TARGET_COLUMN]),
    )
    tree_predictions = tree_model.predict(test_x_encoded)
    tree_classes = np.asarray(tree_model.classes_)
    tree_metrics = evaluate_predictions(
        actual=test_df[TARGET_COLUMN],
        predicted=tree_predictions,
        classes=tree_classes,
        model_name="hist_gradient_boosting",
        feature_set=feature_set,
        train_rows=len(train_df.index),
        test_rows=len(test_df.index),
    )
    return tree_predictions, tree_classes, tree_metrics


def build_predictive_outputs(
    train_sample_size: int = TRAIN_SAMPLE_SIZE,
    test_sample_size: int = TEST_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = model_frame(load_predictive_analysis_frame())
    train_df, test_df = train_test_split_frame(
        df,
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
    )

    train_y = train_df[TARGET_COLUMN]
    test_y = test_df[TARGET_COLUMN]

    baseline_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(POST_ROUTING_FEATURE_COLUMNS)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    baseline_model.fit(train_df[POST_ROUTING_FEATURE_COLUMNS], train_y)
    baseline_predictions = baseline_model.predict(test_df[POST_ROUTING_FEATURE_COLUMNS])
    metrics_frames = [
        evaluate_predictions(
            actual=test_y,
            predicted=baseline_predictions,
            classes=baseline_model.named_steps["model"].classes_,
            model_name="most_frequent_baseline",
            feature_set="shared",
            train_rows=len(train_df.index),
            test_rows=len(test_df.index),
        )
    ]

    prediction_frames: list[pd.DataFrame] = []
    feature_importance_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []

    base_prediction_columns = [
        "unique_key",
        "created_date",
        "created_year",
        "complaint_type",
        "descriptor",
        "agency",
        "borough",
        "incident_zip_clean",
        "community_board",
        "issue_family",
        "issue_subtype",
        TARGET_COLUMN,
    ]

    for feature_set, feature_columns in FEATURE_SET_COLUMNS.items():
        logistic_model, logistic_predictions, logistic_probabilities, logistic_metrics = fit_logistic_benchmark(
            feature_columns,
            train_df,
            test_df,
            feature_set,
        )
        logistic_classes = logistic_model.named_steps["model"].classes_
        metrics_frames.append(logistic_metrics)
        prediction_probability = logistic_probabilities.max(axis=1)
        logistic_prediction_frame = test_df.loc[:, base_prediction_columns].copy()
        logistic_prediction_frame["predicted_resolution_bucket"] = logistic_predictions
        logistic_prediction_frame["predicted_probability"] = prediction_probability.astype(float)
        logistic_prediction_frame["correct_prediction_flag"] = logistic_prediction_frame[TARGET_COLUMN].eq(
            logistic_prediction_frame["predicted_resolution_bucket"]
        )
        logistic_prediction_frame["model_name"] = "multinomial_logistic"
        logistic_prediction_frame["feature_set"] = feature_set
        prediction_frames.append(logistic_prediction_frame)
        feature_importance_frames.append(
            build_feature_importance(
                logistic_model,
                logistic_classes,
                model_name="multinomial_logistic",
                feature_set=feature_set,
            )
        )
        confusion_frames.append(
            build_confusion_matrix_frame(
                logistic_prediction_frame,
                logistic_classes,
                model_name="multinomial_logistic",
                feature_set=feature_set,
            )
        )

        tree_predictions, tree_classes, tree_metrics = fit_tree_benchmark(
            feature_columns,
            train_df,
            test_df,
            feature_set=feature_set,
        )
        metrics_frames.append(tree_metrics)
        confusion_frames.append(
            build_confusion_matrix_frame(
                logistic_prediction_frame.assign(predicted_resolution_bucket=tree_predictions),
                tree_classes,
                model_name="hist_gradient_boosting",
                feature_set=feature_set,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_metrics = pd.concat(metrics_frames, ignore_index=True)
    feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
    error_slices = build_error_slices(predictions)
    confusion = pd.concat(confusion_frames, ignore_index=True)

    predictions["feature_set"] = ordered_categorical(predictions["feature_set"], FEATURE_SET_ORDER)
    predictions["model_name"] = ordered_categorical(predictions["model_name"], MODEL_NAME_ORDER)
    predictions[TARGET_COLUMN] = ordered_categorical(predictions[TARGET_COLUMN], RESOLUTION_BUCKET_ORDER)
    predictions["predicted_resolution_bucket"] = ordered_categorical(
        predictions["predicted_resolution_bucket"], RESOLUTION_BUCKET_ORDER
    )
    predictions = predictions.sort_values(
        ["feature_set", "model_name", "created_date", "unique_key"]
    ).reset_index(drop=True)

    model_metrics["feature_set"] = ordered_categorical(model_metrics["feature_set"], FEATURE_SET_ORDER)
    model_metrics["model_name"] = ordered_categorical(model_metrics["model_name"], MODEL_NAME_ORDER)
    model_metrics["target_class"] = ordered_categorical(model_metrics["target_class"], RESOLUTION_BUCKET_ORDER)
    model_metrics = model_metrics.sort_values(
        ["feature_set", "model_name", "metric_scope", "target_class", "metric"]
    ).reset_index(drop=True)

    error_slices["feature_set"] = ordered_categorical(error_slices["feature_set"], FEATURE_SET_ORDER)
    error_slices["model_name"] = ordered_categorical(error_slices["model_name"], MODEL_NAME_ORDER)
    error_slices = error_slices.sort_values(
        ["feature_set", "model_name", "segment_column", "accuracy", "complaints"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    confusion = confusion.sort_values(
        ["feature_set", "model_name", "actual_resolution_bucket", "predicted_resolution_bucket"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_metrics.to_parquet(MODEL_METRICS_PATH, index=False)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    feature_importance.to_parquet(FEATURE_IMPORTANCE_PATH, index=False)
    error_slices.to_parquet(ERROR_SLICES_PATH, index=False)
    confusion.to_parquet(CONFUSION_MATRIX_PATH, index=False)

    return df, model_metrics, predictions, feature_importance, error_slices, confusion


if __name__ == "__main__":
    build_predictive_outputs()
