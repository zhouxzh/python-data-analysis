from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "output"
DATA_FILE = DATA_DIR / "titanic.csv"

DATA_URLS = [
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
]


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_dataset() -> Path:
    if DATA_FILE.exists():
        return DATA_FILE

    for url in DATA_URLS:
        try:
            urlretrieve(url, DATA_FILE)
            return DATA_FILE
        except URLError:
            continue

    raise RuntimeError(
        "Unable to download Titanic dataset. Check network access or place titanic.csv in report/data/."
    )


def load_dataset() -> pd.DataFrame:
    dataset_path = ensure_dataset()
    df = pd.read_csv(dataset_path)

    if "survived" not in df.columns and "Survived" in df.columns:
        df = df.rename(
            columns={
                "Survived": "survived",
                "Pclass": "pclass",
                "Sex": "sex",
                "Age": "age",
                "SibSp": "sibsp",
                "Parch": "parch",
                "Fare": "fare",
                "Embarked": "embarked",
            }
        )

    keep_columns = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    missing_columns = [column for column in keep_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return df[keep_columns].copy()


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    age_group_median = cleaned.groupby(["sex", "pclass"])["age"].transform("median")
    cleaned["age"] = cleaned["age"].fillna(age_group_median)
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())
    cleaned["embarked"] = cleaned["embarked"].fillna(cleaned["embarked"].mode().iloc[0])
    cleaned["fare"] = cleaned["fare"].fillna(cleaned["fare"].median())

    cleaned["family_size"] = cleaned["sibsp"] + cleaned["parch"] + 1
    cleaned["is_alone"] = (cleaned["family_size"] == 1).astype(int)
    cleaned["age_band"] = pd.cut(
        cleaned["age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["child", "teen", "young_adult", "adult", "senior"],
        include_lowest=True,
    )

    return cleaned


def save_dataset_profile(df: pd.DataFrame) -> dict:
    survival_by_sex = (df.groupby("sex")["survived"].mean().sort_values(ascending=False) * 100).round(2)
    survival_by_class = (df.groupby("pclass")["survived"].mean().sort_index() * 100).round(2)
    survival_by_age = (df.groupby("age_band", observed=False)["survived"].mean() * 100).round(2)

    profile = {
        "sample_count": int(len(df)),
        "survival_rate": round(float(df["survived"].mean() * 100), 2),
        "female_survival_rate": round(float(survival_by_sex.get("female", 0.0)), 2),
        "male_survival_rate": round(float(survival_by_sex.get("male", 0.0)), 2),
        "first_class_survival_rate": round(float(survival_by_class.get(1, 0.0)), 2),
        "third_class_survival_rate": round(float(survival_by_class.get(3, 0.0)), 2),
        "young_adult_survival_rate": round(float(survival_by_age.get("young_adult", 0.0)), 2),
        "adult_survival_rate": round(float(survival_by_age.get("adult", 0.0)), 2),
    }

    with (OUTPUT_DIR / "dataset_profile.json").open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return profile


def create_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="deep")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=df, x="sex", y="survived", estimator="mean", errorbar=None)
    ax.set_title("Survival Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "survival_by_sex.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=df, x="pclass", y="survived", estimator="mean", errorbar=None)
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "survival_by_class.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue="survived", bins=25, kde=True, element="step", stat="density", common_norm=False)
    plt.title("Age Distribution by Survival")
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "age_distribution_by_survival.png", dpi=180)
    plt.close()

    corr_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size", "is_alone"]].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()


def build_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "family_size", "is_alone"]
    X = df[feature_columns].copy()
    y = df["survived"].astype(int)
    numeric_features = ["age", "sibsp", "parch", "fare", "family_size", "is_alone", "pclass"]
    categorical_features = ["sex", "embarked"]
    return X, y, numeric_features, categorical_features


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    X, y, numeric_features, categorical_features = build_model_data(df)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_specs = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42),
    }

    records = []
    best_name = ""
    best_auc = -1.0
    best_confusion = None

    for name, estimator in model_specs.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)

        records.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions)), 4),
                "recall": round(float(recall_score(y_test, predictions)), 4),
                "f1": round(float(f1_score(y_test, predictions)), 4),
                "roc_auc": round(float(auc), 4),
            }
        )

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_confusion = confusion_matrix(y_test, predictions)

    metrics_df = pd.DataFrame(records).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    confusion_df = pd.DataFrame(
        best_confusion,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    confusion_df.to_csv(OUTPUT_DIR / "best_model_confusion_matrix.csv")

    return metrics_df, confusion_df, best_name


def create_confusion_matrix_figure(confusion_df: pd.DataFrame, best_model_name: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix of {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_model_confusion_matrix.png", dpi=180)
    plt.close()


def create_markdown_summary(profile: dict, metrics_df: pd.DataFrame, best_model_name: str) -> None:
    best_row = metrics_df.loc[metrics_df["model"] == best_model_name].iloc[0]
    lines = [
        "# Titanic Analysis Summary",
        "",
        f"- Sample count: {profile['sample_count']}",
        f"- Overall survival rate: {profile['survival_rate']}%",
        f"- Female survival rate: {profile['female_survival_rate']}%",
        f"- Male survival rate: {profile['male_survival_rate']}%",
        f"- First class survival rate: {profile['first_class_survival_rate']}%",
        f"- Third class survival rate: {profile['third_class_survival_rate']}%",
        f"- Best model: {best_model_name}",
        f"- Accuracy: {best_row['accuracy']}",
        f"- Precision: {best_row['precision']}",
        f"- Recall: {best_row['recall']}",
        f"- F1: {best_row['f1']}",
        f"- ROC AUC: {best_row['roc_auc']}",
    ]

    with (OUTPUT_DIR / "analysis_summary.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    ensure_directories()
    raw_df = load_dataset()
    clean_df = clean_and_engineer(raw_df)

    profile = save_dataset_profile(clean_df)
    create_figures(clean_df)
    metrics_df, confusion_df, best_model_name = evaluate_models(clean_df)
    create_confusion_matrix_figure(confusion_df, best_model_name)
    create_markdown_summary(profile, metrics_df, best_model_name)

    print("Titanic analysis completed.")
    print(f"Dataset saved at: {DATA_FILE}")
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()