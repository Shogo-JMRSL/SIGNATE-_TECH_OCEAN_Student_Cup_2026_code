from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 20260828
TARGET = "購入フラグ"
ID_COLUMN = "企業ID"
TEXT_COLUMNS = ["企業概要", "組織図", "今後のDX展望"]
CATEGORICAL_COLUMNS = ["業界", "上場種別", "特徴"]
SURVEY_COLUMNS = [
    "アンケート１",
    "アンケート２",
    "アンケート３",
    "アンケート４",
    "アンケート５",
    "アンケート６",
    "アンケート７",
    "アンケート８",
    "アンケート９",
    "アンケート１０",
    "アンケート１１",
]
FINANCIAL_COLUMNS = [
    "資本金",
    "総資産",
    "流動資産",
    "固定資産",
    "負債",
    "短期借入金",
    "長期借入金",
    "純資産",
    "自己資本",
    "売上",
    "営業利益",
    "経常利益",
    "当期純利益",
    "営業CF",
    "減価償却費",
    "運転資本変動",
    "投資CF",
    "有形固定資産変動",
    "無形固定資産変動(ソフトウェア関連)",
]
DX_TERMS = [
    "DX教育",
    "教育投資",
    "人材育成",
    "eラーニング",
    "研修",
    "リスキリング",
    "デジタルリテラシー",
    "データ分析",
    "ワークショップ",
    "OJT",
    "積極",
    "最優先",
    "拡充",
    "慎重",
    "限定的",
    "段階的",
    "費用対効果",
    "ROI",
]
ORG_TERMS = ["DX推進", "IT", "デジタル", "情報システム", "人事", "教育", "研修", "経営企画"]


@dataclass
class CvResult:
    name: str
    oof: np.ndarray
    test_prob: np.ndarray
    fold_id: np.ndarray
    fold_metrics: list[dict[str, float]]
    global_threshold: float
    global_f1: float
    crossfit_threshold_f1: float
    crossfit_thresholds: list[float]
    extra: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    description = pd.read_csv(data_dir / "description.csv")
    sample = pd.read_csv(data_dir / "sample_submit.csv", header=None)
    return train, test, description, sample


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator / denominator.abs().clip(lower=1.0)


def build_text(df: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in ["業界", "上場種別", "特徴", *TEXT_COLUMNS]:
        pieces.append("【" + column + "】" + df[column].fillna("").astype(str))
    return pd.concat(pieces, axis=1).agg("\n".join, axis=1)


def text_similarity_groups(df: pd.DataFrame, threshold: float = 0.95) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster near-duplicate texts without using the target."""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=25_000,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(build_text(df))
    similarity = cosine_similarity(matrix, dense_output=True)
    np.fill_diagonal(similarity, 0.0)
    parent = np.arange(len(df))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    row_index, column_index = np.where(np.triu(similarity >= threshold, k=1))
    for left, right in zip(row_index, column_index, strict=True):
        union(int(left), int(right))
    roots = np.array([find(index) for index in range(len(df))])
    _, groups = np.unique(roots, return_inverse=True)
    sizes = pd.Series(groups).value_counts()
    stats = {
        "similarity_threshold": threshold,
        "group_count": int(sizes.size),
        "multirow_group_count": int((sizes > 1).sum()),
        "largest_group_size": int(sizes.max()),
        "rows_in_multirow_groups": int(sizes[sizes > 1].sum()),
    }
    return groups, stats


def engineer_tabular(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engineered = df.copy()
    for column in CATEGORICAL_COLUMNS:
        engineered[column] = engineered[column].fillna("__MISSING__").astype(str)

    for column in TEXT_COLUMNS + ["企業名"]:
        text = engineered[column].fillna("").astype(str)
        engineered[f"{column}_文字数"] = text.str.len().astype(float)
        engineered[f"{column}_改行数"] = text.str.count("\n").astype(float)
        engineered[f"{column}_数字数"] = text.str.count(r"\d").astype(float)

    dx_text = engineered["今後のDX展望"].fillna("").astype(str)
    org_text = engineered["組織図"].fillna("").astype(str)
    for term in DX_TERMS:
        engineered[f"展望語_{term}"] = dx_text.str.count(term).astype(float)
    for term in ORG_TERMS:
        engineered[f"組織語_{term}"] = org_text.str.count(term).astype(float)

    numeric_columns = engineered.select_dtypes(include=[np.number]).columns.tolist()
    engineered["欠損数"] = engineered.isna().sum(axis=1).astype(float)
    engineered["財務欠損数"] = engineered[FINANCIAL_COLUMNS].isna().sum(axis=1).astype(float)
    engineered["アンケート欠損数"] = engineered[SURVEY_COLUMNS].isna().sum(axis=1).astype(float)

    for column in ["従業員数", *FINANCIAL_COLUMNS]:
        values = pd.to_numeric(engineered[column], errors="coerce")
        engineered[f"logabs_{column}"] = np.sign(values) * np.log1p(values.abs())

    ratios = {
        "営業利益率": ("営業利益", "売上"),
        "経常利益率": ("経常利益", "売上"),
        "純利益率": ("当期純利益", "売上"),
        "総資産利益率": ("当期純利益", "総資産"),
        "自己資本比率": ("自己資本", "総資産"),
        "負債比率": ("負債", "総資産"),
        "流動資産比率": ("流動資産", "総資産"),
        "固定資産比率": ("固定資産", "総資産"),
        "借入金比率": ("短期借入金", "総資産"),
        "長期借入金比率": ("長期借入金", "総資産"),
        "営業CFマージン": ("営業CF", "売上"),
        "投資CFマージン": ("投資CF", "売上"),
        "従業員一人当たり売上": ("売上", "従業員数"),
        "従業員一人当たり営業利益": ("営業利益", "従業員数"),
        "従業員一人当たり総資産": ("総資産", "従業員数"),
        "従業員一人当たり資本金": ("資本金", "従業員数"),
    }
    for name, (numerator, denominator) in ratios.items():
        engineered[name] = _safe_ratio(engineered[numerator], engineered[denominator])

    survey = engineered[SURVEY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    engineered["DX成熟度平均"] = survey[["アンケート１", "アンケート３", "アンケート５", "アンケート８", "アンケート９", "アンケート１０", "アンケート１１"]].mean(axis=1)
    engineered["DX抵抗感反転"] = 6 - survey["アンケート４"]
    engineered["社内DX不満"] = 6 - survey["アンケート２"]
    engineered["既存ツール不満"] = 6 - survey["アンケート７"]
    engineered["ツール導入済み"] = (survey["アンケート６"] == 1).astype(float)
    engineered["低評価回答数"] = (survey <= 2).sum(axis=1).astype(float)
    engineered["高評価回答数"] = (survey >= 4).sum(axis=1).astype(float)

    drop_columns = [ID_COLUMN, "企業名", *TEXT_COLUMNS]
    engineered = engineered.drop(columns=drop_columns, errors="ignore")
    for column in engineered.columns:
        if column not in CATEGORICAL_COLUMNS:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return engineered, CATEGORICAL_COLUMNS.copy()


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 901), probabilities])
    best_threshold = 0.5
    best_score = -1.0
    best_distance = math.inf
    for threshold in candidates:
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        distance = abs(threshold - 0.5)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and distance < best_distance):
            best_score = score
            best_threshold = float(threshold)
            best_distance = distance
    return best_threshold, float(best_score)


def crossfit_threshold_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[float, list[float], np.ndarray]:
    predictions = np.zeros(len(y_true), dtype=int)
    thresholds: list[float] = []
    for fold in sorted(np.unique(fold_id)):
        train_mask = fold_id != fold
        valid_mask = fold_id == fold
        threshold, _ = best_f1_threshold(y_true[train_mask], probabilities[train_mask])
        thresholds.append(threshold)
        predictions[valid_mask] = (probabilities[valid_mask] >= threshold).astype(int)
    return float(f1_score(y_true, predictions, zero_division=0)), thresholds, predictions


def _fold_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1_at_0_5": float(f1_score(y_true, predictions, zero_division=0)),
    }


def run_catboost_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    name: str,
    params: dict[str, Any],
) -> CvResult:
    x_train, categorical = engineer_tabular(train)
    x_test, _ = engineer_tabular(test)
    oof = np.zeros(len(train), dtype=float)
    test_prob = np.zeros(len(test), dtype=float)
    fold_id = np.zeros(len(train), dtype=int)
    fold_metrics: list[dict[str, float]] = []
    feature_importances: list[np.ndarray] = []
    best_iterations: list[int] = []

    for fold, (train_index, valid_index) in enumerate(splits):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE + fold,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
            **params,
        )
        model.fit(
            x_train.iloc[train_index],
            y[train_index],
            cat_features=categorical,
            eval_set=(x_train.iloc[valid_index], y[valid_index]),
            use_best_model=True,
            early_stopping_rounds=120,
        )
        oof[valid_index] = model.predict_proba(x_train.iloc[valid_index])[:, 1]
        test_prob += model.predict_proba(x_test)[:, 1] / len(splits)
        fold_id[valid_index] = fold
        fold_metrics.append(_fold_metrics(y[valid_index], oof[valid_index]))
        feature_importances.append(model.get_feature_importance())
        best_iterations.append(int(model.get_best_iteration()))

    threshold, global_f1 = best_f1_threshold(y, oof)
    crossfit_f1, crossfit_thresholds, _ = crossfit_threshold_score(y, oof, fold_id)
    importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": np.mean(feature_importances, axis=0),
        }
    ).sort_values("importance", ascending=False)
    return CvResult(
        name=name,
        oof=oof,
        test_prob=test_prob,
        fold_id=fold_id,
        fold_metrics=fold_metrics,
        global_threshold=threshold,
        global_f1=global_f1,
        crossfit_threshold_f1=crossfit_f1,
        crossfit_thresholds=crossfit_thresholds,
        extra={
            "params": params,
            "best_iterations": best_iterations,
            "feature_importance": importance,
        },
    )


def make_text_pipeline(c_value: float, class_weight: str | None = None) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_df=0.995,
                    max_features=120_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=2_000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_text_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    name: str,
    c_value: float,
    class_weight: str | None,
) -> CvResult:
    train_text = build_text(train)
    test_text = build_text(test)
    oof = np.zeros(len(train), dtype=float)
    test_prob = np.zeros(len(test), dtype=float)
    fold_id = np.zeros(len(train), dtype=int)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_index, valid_index) in enumerate(splits):
        model = make_text_pipeline(c_value, class_weight)
        model.fit(train_text.iloc[train_index], y[train_index])
        oof[valid_index] = model.predict_proba(train_text.iloc[valid_index])[:, 1]
        test_prob += model.predict_proba(test_text)[:, 1] / len(splits)
        fold_id[valid_index] = fold
        fold_metrics.append(_fold_metrics(y[valid_index], oof[valid_index]))

    threshold, global_f1 = best_f1_threshold(y, oof)
    crossfit_f1, crossfit_thresholds, _ = crossfit_threshold_score(y, oof, fold_id)
    return CvResult(
        name=name,
        oof=oof,
        test_prob=test_prob,
        fold_id=fold_id,
        fold_metrics=fold_metrics,
        global_threshold=threshold,
        global_f1=global_f1,
        crossfit_threshold_f1=crossfit_f1,
        crossfit_thresholds=crossfit_thresholds,
        extra={"c_value": c_value, "class_weight": class_weight},
    )


def summarize_cv(result: CvResult) -> dict[str, Any]:
    y_metrics = pd.DataFrame(result.fold_metrics)
    return {
        "model": result.name,
        "global_threshold": result.global_threshold,
        "oof_f1_tuned": result.global_f1,
        "crossfit_threshold_f1": result.crossfit_threshold_f1,
        "mean_fold_auc": y_metrics["roc_auc"].mean(),
        "std_fold_auc": y_metrics["roc_auc"].std(ddof=0),
        "mean_fold_average_precision": y_metrics["average_precision"].mean(),
        "mean_fold_f1_at_0_5": y_metrics["f1_at_0_5"].mean(),
        "threshold_std": float(np.std(result.crossfit_thresholds)),
    }


def blend_results(
    tabular: CvResult,
    text: CvResult,
    y: np.ndarray,
    weight_tabular: float,
) -> CvResult:
    oof = weight_tabular * tabular.oof + (1 - weight_tabular) * text.oof
    test_prob = weight_tabular * tabular.test_prob + (1 - weight_tabular) * text.test_prob
    threshold, global_f1 = best_f1_threshold(y, oof)
    crossfit_f1, thresholds, predictions = crossfit_threshold_score(y, oof, tabular.fold_id)
    fold_metrics = []
    for fold in sorted(np.unique(tabular.fold_id)):
        mask = tabular.fold_id == fold
        metric = _fold_metrics(y[mask], oof[mask])
        metric["f1_crossfit_threshold"] = float(f1_score(y[mask], predictions[mask], zero_division=0))
        fold_metrics.append(metric)
    return CvResult(
        name=f"blend_tabular_{weight_tabular:.2f}",
        oof=oof,
        test_prob=test_prob,
        fold_id=tabular.fold_id.copy(),
        fold_metrics=fold_metrics,
        global_threshold=threshold,
        global_f1=global_f1,
        crossfit_threshold_f1=crossfit_f1,
        crossfit_thresholds=thresholds,
        extra={"weight_tabular": weight_tabular, "weight_text": 1 - weight_tabular},
    )


def compute_data_quality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    data_dir: Path,
) -> dict[str, Any]:
    y = train[TARGET]
    expected_test_columns = [column for column in train.columns if column != TARGET]
    survey_validity = {}
    for column in SURVEY_COLUMNS:
        allowed = {1, 2} if column == "アンケート６" else {1, 2, 3, 4, 5}
        invalid = int((~train[column].dropna().isin(allowed)).sum() + (~test[column].dropna().isin(allowed)).sum())
        survey_validity[column] = invalid

    missing = []
    for column in expected_test_columns:
        train_rate = float(train[column].isna().mean())
        test_rate = float(test[column].isna().mean())
        if train_rate or test_rate:
            missing.append({"column": column, "train_rate": train_rate, "test_rate": test_rate})
    missing.sort(key=lambda row: max(row["train_rate"], row["test_rate"]), reverse=True)

    numeric_drift = []
    for column in train[expected_test_columns].select_dtypes(include=[np.number]).columns:
        if column == ID_COLUMN:
            continue
        a = train[column].dropna().to_numpy()
        b = test[column].dropna().to_numpy()
        if len(a) > 3 and len(b) > 3 and len(np.unique(np.r_[a, b])) > 1:
            statistic, p_value = ks_2samp(a, b)
            numeric_drift.append({"column": column, "ks_statistic": float(statistic), "p_value": float(p_value)})
    numeric_drift.sort(key=lambda row: row["ks_statistic"], reverse=True)

    categorical_drift = []
    for column in CATEGORICAL_COLUMNS:
        train_share = train[column].fillna("__MISSING__").value_counts(normalize=True)
        test_share = test[column].fillna("__MISSING__").value_counts(normalize=True)
        levels = train_share.index.union(test_share.index)
        total_variation = 0.5 * float(sum(abs(train_share.get(level, 0) - test_share.get(level, 0)) for level in levels))
        unseen = sorted(set(test_share.index) - set(train_share.index))
        categorical_drift.append({"column": column, "total_variation": total_variation, "unseen_test_levels": unseen})

    text = build_text(train)
    text_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=25_000, dtype=np.float32)
    train_matrix = text_vectorizer.fit_transform(text)
    similarity_train = cosine_similarity(train_matrix, dense_output=True)
    np.fill_diagonal(similarity_train, 0.0)
    upper = similarity_train[np.triu_indices_from(similarity_train, k=1)]
    train_near_duplicate_pairs = int((upper >= 0.95).sum())
    max_train_similarity = float(upper.max()) if len(upper) else 0.0

    test_matrix = text_vectorizer.transform(build_text(test))
    cross_similarity = cosine_similarity(train_matrix, test_matrix, dense_output=True)
    train_test_near_duplicates = int((cross_similarity.max(axis=0) >= 0.95).sum())
    max_train_test_similarity = float(cross_similarity.max())

    accounting = {
        "assets_equal_current_plus_fixed_share": float(np.isclose(train["総資産"], train["流動資産"] + train["固定資産"], rtol=0, atol=1).mean()),
        "assets_equal_liabilities_plus_net_assets_share": float(np.isclose(train["総資産"], train["負債"] + train["純資産"], rtol=0, atol=1).mean()),
    }
    _, similarity_group_stats = text_similarity_groups(train, threshold=0.95)
    return {
        "input_hashes": {path.name: sha256(path) for path in sorted(data_dir.glob("*.csv"))},
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": len(expected_test_columns),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "target_positive_rate": float(y.mean()),
        "train_id_unique": bool(train[ID_COLUMN].is_unique),
        "test_id_unique": bool(test[ID_COLUMN].is_unique),
        "train_exact_duplicates": int(train.duplicated().sum()),
        "test_exact_duplicates": int(test.duplicated().sum()),
        "train_test_columns_match": expected_test_columns == list(test.columns),
        "sample_rows": int(len(sample)),
        "sample_ids_match_test": bool(sample.iloc[:, 0].reset_index(drop=True).equals(test[ID_COLUMN].reset_index(drop=True))),
        "sample_has_two_columns": sample.shape[1] == 2,
        "missing_rates": missing,
        "survey_invalid_value_counts": survey_validity,
        "numeric_drift_top": numeric_drift[:10],
        "categorical_drift": categorical_drift,
        "train_near_duplicate_pairs_ge_0_95": train_near_duplicate_pairs,
        "max_train_similarity": max_train_similarity,
        "test_rows_with_train_similarity_ge_0_95": train_test_near_duplicates,
        "max_train_test_similarity": max_train_test_similarity,
        "similarity_group_stats": similarity_group_stats,
        "accounting_consistency": accounting,
    }


def segment_rates(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ["業界", "上場種別", "特徴", *SURVEY_COLUMNS]:
        grouped = train.groupby(column, dropna=False)[TARGET].agg(["size", "sum", "mean"]).reset_index()
        for _, row in grouped.iterrows():
            n = int(row["size"])
            rate = float(row["mean"])
            z = 1.96
            denominator = 1 + z * z / n
            center = (rate + z * z / (2 * n)) / denominator
            margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
            rows.append(
                {
                    "segment_column": column,
                    "segment_value": "欠損" if pd.isna(row[column]) else str(row[column]),
                    "companies": n,
                    "purchases": int(row["sum"]),
                    "purchase_rate": rate,
                    "wilson_low_95": max(0.0, center - margin),
                    "wilson_high_95": min(1.0, center + margin),
                }
            )
    return pd.DataFrame(rows).sort_values(["segment_column", "purchase_rate"], ascending=[True, False])


def fit_final_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    cat_params: dict[str, Any],
    best_iterations: list[int],
    text_c: float,
    text_class_weight: str | None,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, categorical = engineer_tabular(train)
    x_test, _ = engineer_tabular(test)
    final_iterations = max(100, int(np.median([value for value in best_iterations if value >= 0]) * 1.10))
    final_cat_params = dict(cat_params)
    final_cat_params["iterations"] = final_iterations
    cat_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
        **final_cat_params,
    )
    cat_model.fit(x_train, y, cat_features=categorical)
    cat_train_prob = cat_model.predict_proba(x_train)[:, 1]
    cat_test_prob = cat_model.predict_proba(x_test)[:, 1]
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    cat_model.save_model(model_dir / "catboost_model.cbm")

    text_model = make_text_pipeline(text_c, text_class_weight)
    text_model.fit(build_text(train), y)
    text_train_prob = text_model.predict_proba(build_text(train))[:, 1]
    text_test_prob = text_model.predict_proba(build_text(test))[:, 1]
    joblib.dump(text_model, model_dir / "text_pipeline.joblib")

    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": cat_model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    vectorizer: TfidfVectorizer = text_model.named_steps["tfidf"]
    classifier: LogisticRegression = text_model.named_steps["logistic"]
    terms = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    positive = pd.DataFrame(
        {"term": terms[np.argsort(coefficients)[-100:][::-1]], "coefficient": np.sort(coefficients)[-100:][::-1]}
    )
    negative = pd.DataFrame(
        {"term": terms[np.argsort(coefficients)[:100]], "coefficient": np.sort(coefficients)[:100]}
    )
    train_prob = np.c_[cat_train_prob, text_train_prob]
    test_prob = np.c_[cat_test_prob, text_test_prob]
    return train_prob, test_prob, importance, positive, negative


def tail_stress_test(
    train: pd.DataFrame,
    y: np.ndarray,
    cat_params: dict[str, Any],
    text_c: float,
    text_class_weight: str | None,
) -> dict[str, Any]:
    order = np.argsort(train[ID_COLUMN].to_numpy())
    cutoff = int(len(train) * 0.80)
    development_index = order[:cutoff]
    holdout_index = order[cutoff:]
    development = train.iloc[development_index].reset_index(drop=True)
    holdout = train.iloc[holdout_index].reset_index(drop=True)
    development_y = y[development_index]
    holdout_y = y[holdout_index]
    development_groups, _ = text_similarity_groups(development, threshold=0.95)
    inner_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 77)
    inner = list(inner_splitter.split(development, development_y, groups=development_groups))

    tabular = run_catboost_cv(development, holdout, development_y, inner, "tail_tabular", cat_params)
    text = run_text_cv(development, holdout, development_y, inner, "tail_text", text_c, text_class_weight)
    blends = [blend_results(tabular, text, development_y, weight) for weight in np.linspace(0, 1, 11)]
    best = max(blends, key=lambda result: (result.crossfit_threshold_f1, result.global_f1))
    holdout_prob = best.test_prob
    predictions = holdout_prob >= best.global_threshold
    return {
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "holdout_id_min": int(holdout[ID_COLUMN].min()),
        "holdout_id_max": int(holdout[ID_COLUMN].max()),
        "development_positive_rate": float(development_y.mean()),
        "holdout_positive_rate": float(holdout_y.mean()),
        "selected_weight_tabular": best.extra["weight_tabular"],
        "selected_threshold": best.global_threshold,
        "holdout_f1": float(f1_score(holdout_y, predictions, zero_division=0)),
        "holdout_precision": float(precision_score(holdout_y, predictions, zero_division=0)),
        "holdout_recall": float(recall_score(holdout_y, predictions, zero_division=0)),
        "holdout_accuracy": float(accuracy_score(holdout_y, predictions)),
        "holdout_roc_auc": float(roc_auc_score(holdout_y, holdout_prob)),
    }


def run_pipeline(project_dir: Path) -> dict[str, Any]:
    data_dir = project_dir / "data"
    output_dir = project_dir / "output"
    baseline_dir = output_dir / "experiments" / "baseline"
    sales_dir = output_dir / "analysis" / "sales"
    text_analysis_dir = output_dir / "analysis" / "text"
    company_segments_dir = output_dir / "analysis" / "company_segments"
    validation_dir = output_dir / "validation"
    submission_dir = project_dir / "submissions" / "baseline_v1"
    for directory in [
        output_dir,
        baseline_dir,
        sales_dir,
        text_analysis_dir,
        company_segments_dir,
        validation_dir,
        submission_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    train, test, description, sample = load_inputs(data_dir)
    y = train[TARGET].to_numpy(dtype=int)
    features = train.drop(columns=[TARGET])

    data_quality = compute_data_quality(train, test, sample, data_dir)
    write_json(validation_dir / "data_quality_summary.json", data_quality)

    groups, group_stats = text_similarity_groups(features, threshold=0.95)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(features, y, groups=groups))

    catboost_configs = [
        (
            "catboost_depth5",
            {
                "iterations": 1_000,
                "learning_rate": 0.035,
                "depth": 5,
                "l2_leaf_reg": 7.0,
                "random_strength": 0.7,
                "bagging_temperature": 0.5,
            },
        ),
        (
            "catboost_depth6",
            {
                "iterations": 900,
                "learning_rate": 0.04,
                "depth": 6,
                "l2_leaf_reg": 9.0,
                "random_strength": 0.5,
                "bagging_temperature": 0.8,
            },
        ),
    ]
    tabular_results = [
        run_catboost_cv(features, test, y, splits, name, params)
        for name, params in catboost_configs
    ]
    text_configs = [
        ("text_c0_7", 0.7, None),
        ("text_c1_5", 1.5, None),
        ("text_c1_5_balanced", 1.5, "balanced"),
    ]
    text_results = [
        run_text_cv(features, test, y, splits, name, c_value, class_weight)
        for name, c_value, class_weight in text_configs
    ]

    best_tabular = max(tabular_results, key=lambda result: (result.crossfit_threshold_f1, result.global_f1))
    best_text = max(text_results, key=lambda result: (result.crossfit_threshold_f1, result.global_f1))
    blend_results_all = [
        blend_results(best_tabular, best_text, y, float(weight))
        for weight in np.linspace(0, 1, 21)
    ]
    best_blend = max(blend_results_all, key=lambda result: (result.crossfit_threshold_f1, result.global_f1))
    alternative_blend = max(
        (
            result
            for result in blend_results_all
            if abs(float(result.extra["weight_tabular"]) - float(best_blend.extra["weight_tabular"])) >= 0.19
        ),
        key=lambda result: (result.crossfit_threshold_f1, result.global_f1),
    )

    # 閾値スイープ: OOF上で固定閾値を評価して比較できるようにする
    threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    threshold_rows: list[dict[str, float]] = []
    for t in threshold_grid:
        preds = (best_blend.oof >= t).astype(int)
        threshold_rows.append(
            {
                "threshold": float(t),
                "oof_f1": float(f1_score(y, preds, zero_division=0)),
                "oof_precision": float(precision_score(y, preds, zero_division=0)),
                "oof_recall": float(recall_score(y, preds, zero_division=0)),
                "predicted_positive_count": int(preds.sum()),
                "predicted_positive_rate": float(preds.mean()),
            }
        )
    try:
        pd.DataFrame(threshold_rows).to_csv(baseline_dir / "model_thresholds_oof.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

    all_results = [*tabular_results, *text_results, *blend_results_all]
    comparison = pd.DataFrame([summarize_cv(result) for result in all_results])
    comparison.insert(0, "validation_scheme", "stratified_group_5fold")
    comparison = comparison.sort_values(
        ["crossfit_threshold_f1", "oof_f1_tuned"], ascending=False
    )

    random_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(features, y))
    random_tabular = run_catboost_cv(
        features,
        test,
        y,
        random_splits,
        f"random_{best_tabular.name}",
        best_tabular.extra["params"],
    )
    random_text = run_text_cv(
        features,
        test,
        y,
        random_splits,
        f"random_{best_text.name}",
        float(best_text.extra["c_value"]),
        best_text.extra["class_weight"],
    )
    random_blend = blend_results(random_tabular, random_text, y, float(best_blend.extra["weight_tabular"]))
    random_rows = pd.DataFrame([summarize_cv(result) for result in [random_tabular, random_text, random_blend]])
    random_rows.insert(0, "validation_scheme", "random_5fold_diagnostic")
    comparison = pd.concat([comparison, random_rows], ignore_index=True)
    comparison.to_csv(baseline_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    selected_tabular_params = best_tabular.extra["params"]
    selected_iterations = best_tabular.extra["best_iterations"]
    selected_text_c = float(best_text.extra["c_value"])
    selected_text_class_weight = best_text.extra["class_weight"]
    _, final_test_components, importance, positive_terms, negative_terms = fit_final_models(
        features,
        test,
        y,
        selected_tabular_params,
        selected_iterations,
        selected_text_c,
        selected_text_class_weight,
        output_dir,
    )

    weight_tabular = float(best_blend.extra["weight_tabular"])
    final_test_probability = weight_tabular * final_test_components[:, 0] + (1 - weight_tabular) * final_test_components[:, 1]
    threshold = float(best_blend.global_threshold)
    final_prediction = (final_test_probability >= threshold).astype(int)
    alternative_weight_tabular = float(alternative_blend.extra["weight_tabular"])
    alternative_probability = (
        alternative_weight_tabular * final_test_components[:, 0]
        + (1 - alternative_weight_tabular) * final_test_components[:, 1]
    )
    alternative_threshold = float(alternative_blend.global_threshold)
    alternative_prediction = (alternative_probability >= alternative_threshold).astype(int)

    # 各固定閾値でのテスト提出ファイルを出力
    threshold_test_rows: list[dict[str, float]] = []
    for t in threshold_grid:
        test_pred = (final_test_probability >= t).astype(int)
        try:
            pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(int), TARGET: test_pred}).to_csv(
                output_dir / f"submission_threshold_{int(t*100):02d}.csv",
                index=False,
                header=False,
                encoding="utf-8",
            )
        except Exception:
            pass
        threshold_test_rows.append(
            {
                "threshold": float(t),
                "predicted_positive_count": int(test_pred.sum()),
                "predicted_positive_rate": float(test_pred.mean()),
            }
        )

    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(int), TARGET: final_prediction})
    submission.to_csv(submission_dir / "submission_baseline_v1.csv", index=False, header=False, encoding="utf-8")
    alternative_submission = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN].astype(int), TARGET: alternative_prediction}
    )
    alternative_submission.to_csv(
        submission_dir / "submission_baseline_v1_alternative.csv", index=False, header=False, encoding="utf-8"
    )
    probabilities = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].astype(int),
            "catboost_probability": final_test_components[:, 0],
            "text_probability": final_test_components[:, 1],
            "ensemble_probability": final_test_probability,
            "alternative_probability": alternative_probability,
            "prediction": final_prediction,
            "alternative_prediction": alternative_prediction,
        }
    )
    probabilities.to_csv(baseline_dir / "test_probabilities.csv", index=False, encoding="utf-8-sig")
    prospects = pd.concat(
        [test[[ID_COLUMN, "企業名", "業界", "上場種別", "特徴"]].reset_index(drop=True), probabilities.drop(columns=[ID_COLUMN])],
        axis=1,
    ).sort_values("ensemble_probability", ascending=False)
    prospects.head(100).to_csv(sales_dir / "top_prospects.csv", index=False, encoding="utf-8-sig")

    importance.to_csv(baseline_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    positive_terms.to_csv(text_analysis_dir / "text_terms_positive.csv", index=False, encoding="utf-8-sig")
    negative_terms.to_csv(text_analysis_dir / "text_terms_negative.csv", index=False, encoding="utf-8-sig")
    rates = segment_rates(train)
    rates.to_csv(company_segments_dir / "segment_purchase_rates.csv", index=False, encoding="utf-8-sig")

    crossfit_f1, crossfit_thresholds, crossfit_predictions = crossfit_threshold_score(y, best_blend.oof, best_blend.fold_id)
    oof = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN].astype(int),
            TARGET: y,
            "fold": best_blend.fold_id,
            "catboost_oof_probability": best_tabular.oof,
            "text_oof_probability": best_text.oof,
            "ensemble_oof_probability": best_blend.oof,
            "crossfit_threshold_prediction": crossfit_predictions,
        }
    )
    oof.to_csv(baseline_dir / "cv_oof_predictions.csv", index=False, encoding="utf-8-sig")

    stress = tail_stress_test(
        features,
        y,
        selected_tabular_params,
        selected_text_c,
        selected_text_class_weight,
    )
    selected_predictions = (best_blend.oof >= threshold).astype(int)
    metrics = {
        "random_state": RANDOM_STATE,
        "validation_scheme": "stratified_group_5fold",
        "text_similarity_groups": group_stats,
        "selected_tabular_model": best_tabular.name,
        "selected_text_model": best_text.name,
        "selected_model": best_blend.name,
        "weight_tabular": weight_tabular,
        "weight_text": 1 - weight_tabular,
        "oof_threshold": threshold,
        "oof_f1_tuned": float(f1_score(y, selected_predictions, zero_division=0)),
        "oof_precision_tuned": float(precision_score(y, selected_predictions, zero_division=0)),
        "oof_recall_tuned": float(recall_score(y, selected_predictions, zero_division=0)),
        "oof_accuracy_tuned": float(accuracy_score(y, selected_predictions)),
        "oof_roc_auc": float(roc_auc_score(y, best_blend.oof)),
        "oof_average_precision": float(average_precision_score(y, best_blend.oof)),
        "crossfit_threshold_f1": crossfit_f1,
        "crossfit_thresholds": crossfit_thresholds,
        "fold_auc": [row["roc_auc"] for row in best_blend.fold_metrics],
        "random_cv_diagnostic": summarize_cv(random_blend),
        "predicted_positive_count": int(final_prediction.sum()),
        "predicted_positive_rate": float(final_prediction.mean()),
        "threshold_sweep_oof": threshold_rows,
        "threshold_sweep_test": threshold_test_rows,
        "alternative_submission": {
            "model": alternative_blend.name,
            "weight_tabular": alternative_weight_tabular,
            "weight_text": 1 - alternative_weight_tabular,
            "threshold": alternative_threshold,
            "crossfit_threshold_f1": alternative_blend.crossfit_threshold_f1,
            "oof_f1_tuned": alternative_blend.global_f1,
            "predicted_positive_count": int(alternative_prediction.sum()),
            "predicted_positive_rate": float(alternative_prediction.mean()),
            "prediction_disagreement_count": int((alternative_prediction != final_prediction).sum()),
        },
        "tail_stress_test": stress,
        "input_shapes": {"train": list(train.shape), "test": list(test.shape), "description": list(description.shape)},
    }
    write_json(baseline_dir / "model_metrics.json", metrics)
    write_json(
        output_dir / "model" / "model_config.json",
        {
            "catboost_params": selected_tabular_params,
            "catboost_final_iterations": max(100, int(np.median([v for v in selected_iterations if v >= 0]) * 1.10)),
            "text_c": selected_text_c,
            "text_class_weight": selected_text_class_weight,
            "weight_tabular": weight_tabular,
            "threshold": threshold,
            "alternative_weight_tabular": alternative_weight_tabular,
            "alternative_threshold": alternative_threshold,
            "text_columns": ["業界", "上場種別", "特徴", *TEXT_COLUMNS],
            "tabular_dropped_columns": [ID_COLUMN, "企業名", *TEXT_COLUMNS],
        },
    )
    return {
        "metrics": metrics,
        "data_quality": data_quality,
        "comparison": comparison,
        "feature_importance": importance,
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "segment_rates": rates,
        "submission": submission,
        "prospects": prospects,
    }
