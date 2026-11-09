"""Leakage-aware model tuning, hold-out evaluation and model registration."""
from __future__ import annotations

from pathlib import Path
import json
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
import sklearn

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
FEATURES = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
]
TARGET = "Engine Condition"
RANDOM_STATE = 42
N_SPLITS = 3
N_ITER = 3


def choose_best_index(cv_results: dict) -> int:
    """Select by mean class-1 recall, using mean class-1 F1 as a tie-break."""
    recalls = np.asarray(cv_results["mean_test_recall_class1"], dtype=float)
    f1_scores = np.asarray(cv_results["mean_test_f1_class1"], dtype=float)
    best_recall = np.nanmax(recalls)
    candidates = np.flatnonzero(np.isclose(recalls, best_recall, rtol=0.0, atol=1e-12))
    return int(candidates[np.nanargmax(f1_scores[candidates])])


def load_workflow_artifact(bundle_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Workflow artifact not found: {bundle_path}")
    with zipfile.ZipFile(bundle_path) as archive:
        train = pd.read_csv(archive.open("train.csv"))
        test = pd.read_csv(archive.open("test.csv"))
        split_metadata = json.load(archive.open("split_metadata.json"))
    for label, frame in [("train", train), ("test", test)]:
        expected = FEATURES + [TARGET]
        if frame.columns.tolist() != expected:
            raise ValueError(f"{label} artifact schema does not match expected feature order.")
        if frame.isna().any().any():
            raise ValueError(f"{label} artifact contains missing values.")
    if len(train) != split_metadata["train_rows"] or len(test) != split_metadata["test_rows"]:
        raise ValueError("Workflow artifact row counts do not match split metadata.")
    return train, test, split_metadata


def run_training(bundle_path: str | Path | None = None):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle_path) if bundle_path else ARTIFACTS / "train_test_workflow_artifact.zip"
    train, test, split_metadata = load_workflow_artifact(bundle)

    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "recall_class1": "recall",
        "f1_class1": "f1",
        "roc_auc": "roc_auc",
    }

    # Training-only majority-class baseline: the hold-out test remains reserved.
    baseline_cv = cross_validate(
        DummyClassifier(strategy="most_frequent"),
        X_train, y_train, cv=cv, scoring=scoring,
        n_jobs=1, return_train_score=False, error_score="raise",
    )
    baseline = {
        "model": "DummyClassifier (most_frequent)",
        "cv_recall_class1_mean": float(np.mean(baseline_cv["test_recall_class1"])),
        "cv_recall_class1_std": float(np.std(baseline_cv["test_recall_class1"], ddof=1)),
        "cv_f1_class1_mean": float(np.mean(baseline_cv["test_f1_class1"])),
        "cv_roc_auc_mean": float(np.mean(baseline_cv["test_roc_auc"])),
    }

    candidates = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"),
            {
                "n_estimators": [25, 35, 45],
                "max_depth": [10, None],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", 0.8],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"),
            {
                "n_estimators": [25, 35, 45],
                "max_depth": [10, None],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", 0.8],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [25, 35, 45],
                "learning_rate": [0.03, 0.08],
                "max_depth": [1, 2],
                "min_samples_leaf": [1, 3],
                "subsample": [0.8, 1.0],
            },
        ),
    }

    experiment_rows = []
    search_objects = {}
    for model_name, (estimator, parameter_space) in candidates.items():
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameter_space,
            n_iter=N_ITER,
            scoring=scoring,
            refit=choose_best_index,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
            verbose=0,
        )
        search.fit(X_train, y_train)
        search_objects[model_name] = search
        best_index = search.best_index_
        cv_results = search.cv_results_

        experiment_rows.append({
            "model": model_name,
            "best_cv_recall_class1": float(cv_results["mean_test_recall_class1"][best_index]),
            "cv_recall_std": float(cv_results["std_test_recall_class1"][best_index]),
            "best_cv_f1_class1": float(cv_results["mean_test_f1_class1"][best_index]),
            "best_cv_roc_auc": float(cv_results["mean_test_roc_auc"][best_index]),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "search_iterations": N_ITER,
            "cv_folds": N_SPLITS,
            "selection_rule": "max mean recall_class1; tie-break max mean f1_class1",
        })
        details = pd.DataFrame(cv_results).copy()
        details.insert(0, "model", model_name)
        details.to_csv(
            ARTIFACTS / f"tuning_detail_{model_name.lower().replace(' ', '_')}.csv",
            index=False,
        )

    experiments = (
        pd.DataFrame(experiment_rows)
        .sort_values(
            ["best_cv_recall_class1", "best_cv_f1_class1"],
            ascending=[False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    winner_name = str(experiments.loc[0, "model"])
    winner_search = search_objects[winner_name]
    best_model = winner_search.best_estimator_

    # Exactly one final evaluation on the untouched hold-out test set.
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, list(best_model.classes_).index(1)]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    class_report = classification_report(
        y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
    )
    metrics = {
        "selected_model": winner_name,
        "test_rows": int(len(test)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "test_precision_class0": float(precision_score(y_test, predictions, pos_label=0, zero_division=0)),
        "test_recall_class0": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
        "test_f1_class0": float(f1_score(y_test, predictions, pos_label=0, zero_division=0)),
        "test_precision_class1": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "test_recall_class1": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "test_f1_class1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_average_precision_class1": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "classification_report": class_report,
    }

    experiments.to_csv(ARTIFACTS / "experiment_log.csv", index=False)
    experiments.to_json(ARTIFACTS / "experiment_log.json", orient="records", indent=2)
    pd.DataFrame([baseline]).to_csv(ARTIFACTS / "baseline_cv_metrics.csv", index=False)
    (ARTIFACTS / "holdout_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "best_params.json").write_text(json.dumps(winner_search.best_params_, indent=2))
    joblib.dump(best_model, ARTIFACTS / "best_model.joblib")

    metadata = {
        "model_name": winner_name,
        "features": FEATURES,
        "target": TARGET,
        "classes": [int(value) for value in best_model.classes_],
        "target_class_semantics": "Dataset documentation does not define which class means failure.",
        "selection_metric": "Three-fold stratified CV recall for numeric class 1; F1 tie-break",
        "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS,
        "search_iterations_per_candidate": N_ITER,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_metadata": split_metadata,
        "best_parameters": winner_search.best_params_,
        "test_metrics": {key: value for key, value in metrics.items() if key != "classification_report"},
        "library_versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Diagnostic figures
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, values_format="d"
    )
    ax.set_title("Final hold-out confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Final hold-out ROC curve")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_roc_curve.png", dpi=180)
    plt.close(fig)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1])
    ax.set_title(f"{winner_name} feature importance")
    ax.set_xlabel("Impurity-based importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close(fig)

    print("TRAINING-ONLY CROSS-VALIDATION RESULTS")
    print(experiments.to_string(index=False))
    print("\nTRAINING-ONLY BASELINE")
    print(json.dumps(baseline, indent=2))
    print("\nSELECTED MODEL:", winner_name)
    print("\nFINAL HOLD-OUT METRICS")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, labels=[0, 1], digits=4, zero_division=0))
    return experiments, metrics, best_model, baseline


if __name__ == "__main__":
    run_training()
