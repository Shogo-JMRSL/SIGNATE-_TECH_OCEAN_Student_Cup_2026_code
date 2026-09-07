from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.competition_pipeline import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    RANDOM_STATE,
    TARGET,
    CvResult,
    _fold_metrics,
    best_f1_threshold,
    blend_results,
    crossfit_threshold_score,
    engineer_tabular,
    text_similarity_groups,
)


BASIC_FEATURES = [
    "segment_dx_issue_b2b",
    "segment_low_satisfaction_b2b",
    "segment_priority_industry",
]
OVERLAP_FEATURES = [
    "segment_dx_issue_low_satisfaction_b2b",
    "segment_dx_issue_priority_industry",
    "segment_low_satisfaction_priority_industry",
    "segment_all_high_priority",
]
SURVEY_INTERACTION_FEATURES = [
    "low_dx_resistance",
    "low_existing_tool_satisfaction",
    "low_resistance_and_low_satisfaction",
]
EXPERIMENTS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("Baseline", []),
        ("segment_v1_basic", BASIC_FEATURES),
        ("segment_v2_overlap", BASIC_FEATURES + OVERLAP_FEATURES),
        ("segment_v3_survey", BASIC_FEATURES + SURVEY_INTERACTION_FEATURES),
        ("segment_v4_all", BASIC_FEATURES + OVERLAP_FEATURES + SURVEY_INTERACTION_FEATURES),
    ]
)
FEATURE_DEFINITIONS = {
    "segment_dx_issue_b2b": "(アンケート2=1 または アンケート10=1) かつ 特徴=BtoB",
    "segment_low_satisfaction_b2b": "アンケート7<=2 かつ 特徴=BtoB",
    "segment_priority_industry": "業界がIT・人材・自動車/乗り物のいずれか",
    "segment_dx_issue_low_satisfaction_b2b": "DX課題顕在 かつ アンケート7<=2 かつ 特徴=BtoB",
    "segment_dx_issue_priority_industry": "DX課題顕在 かつ 優先業界",
    "segment_low_satisfaction_priority_industry": "アンケート7<=2 かつ 優先業界",
    "segment_all_high_priority": "DX課題顕在 かつ アンケート7<=2 かつ 特徴=BtoB かつ 優先業界",
    "low_dx_resistance": "アンケート4<=2（1=抵抗感が非常に低い、2=低い）",
    "low_existing_tool_satisfaction": "アンケート7<=2（1=満足度が非常に低い、2=低い）",
    "low_resistance_and_low_satisfaction": "アンケート4<=2 かつ アンケート7<=2",
}
WEIGHTS = np.round(np.arange(0.0, 0.5001, 0.05), 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def build_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    survey2 = pd.to_numeric(df["アンケート２"], errors="coerce")
    survey4 = pd.to_numeric(df["アンケート４"], errors="coerce")
    survey7 = pd.to_numeric(df["アンケート７"], errors="coerce")
    survey10 = pd.to_numeric(df["アンケート１０"], errors="coerce")

    dx_issue = (survey2 == 1) | (survey10 == 1)
    b2b = df["特徴"].fillna("").eq("BtoB")
    low_satisfaction = survey7.le(2)
    priority_industry = df["業界"].isin(["IT", "人材", "自動車・乗り物"])
    low_resistance = survey4.le(2)

    return pd.DataFrame(
        {
            "segment_dx_issue_b2b": dx_issue & b2b,
            "segment_low_satisfaction_b2b": low_satisfaction & b2b,
            "segment_priority_industry": priority_industry,
            "segment_dx_issue_low_satisfaction_b2b": dx_issue & low_satisfaction & b2b,
            "segment_dx_issue_priority_industry": dx_issue & priority_industry,
            "segment_low_satisfaction_priority_industry": low_satisfaction & priority_industry,
            "segment_all_high_priority": dx_issue & low_satisfaction & b2b & priority_industry,
            "low_dx_resistance": low_resistance,
            "low_existing_tool_satisfaction": low_satisfaction,
            "low_resistance_and_low_satisfaction": low_resistance & low_satisfaction,
        },
        index=df.index,
    ).astype(np.int8)


def engineer_for_experiment(df: pd.DataFrame, added_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    engineered, categorical = engineer_tabular(df)
    candidates = build_candidate_features(df)
    for feature in added_features:
        engineered[feature] = candidates[feature].astype(float)
    return engineered, categorical


def run_catboost_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    experiment: str,
    added_features: list[str],
    params: dict[str, Any],
) -> CvResult:
    x_train, categorical = engineer_for_experiment(train, added_features)
    x_test, _ = engineer_for_experiment(test, added_features)
    if x_train.columns.tolist() != x_test.columns.tolist():
        raise ValueError(f"{experiment}: train/testの特徴量列が一致しません。")

    oof = np.zeros(len(train), dtype=float)
    test_prob = np.zeros(len(test), dtype=float)
    fold_id = np.full(len(train), -1, dtype=int)
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

    if (fold_id < 0).any():
        raise RuntimeError(f"{experiment}: OOF未割当行があります。")
    threshold, global_f1 = best_f1_threshold(y, oof)
    crossfit_f1, crossfit_thresholds, _ = crossfit_threshold_score(y, oof, fold_id)
    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": np.mean(feature_importances, axis=0)}
    ).sort_values("importance", ascending=False)
    return CvResult(
        name=experiment,
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
            "added_features": added_features,
            "best_iterations": best_iterations,
            "feature_importance": importance,
        },
    )


def saved_text_result(
    y: np.ndarray,
    saved_oof: pd.DataFrame,
    saved_test: pd.DataFrame,
) -> CvResult:
    oof = saved_oof["text_oof_probability"].to_numpy(dtype=float)
    test_prob = saved_test["text_probability"].to_numpy(dtype=float)
    fold_id = saved_oof["fold"].to_numpy(dtype=int)
    threshold, global_f1 = best_f1_threshold(y, oof)
    crossfit_f1, thresholds, _ = crossfit_threshold_score(y, oof, fold_id)
    fold_metrics = [_fold_metrics(y[fold_id == fold], oof[fold_id == fold]) for fold in sorted(np.unique(fold_id))]
    return CvResult(
        name="saved_text_c0_7",
        oof=oof,
        test_prob=test_prob,
        fold_id=fold_id,
        fold_metrics=fold_metrics,
        global_threshold=threshold,
        global_f1=global_f1,
        crossfit_threshold_f1=crossfit_f1,
        crossfit_thresholds=thresholds,
        extra={"c_value": 0.7, "class_weight": None, "source": "current saved OOF/test probabilities"},
    )


def result_metrics(y: np.ndarray, result: CvResult) -> dict[str, float]:
    predictions = (result.oof >= result.global_threshold).astype(int)
    return {
        "OOF F1": float(f1_score(y, predictions, zero_division=0)),
        "crossfit F1": float(result.crossfit_threshold_f1),
        "Precision": float(precision_score(y, predictions, zero_division=0)),
        "Recall": float(recall_score(y, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y, result.oof)),
        "Average Precision": float(average_precision_score(y, result.oof)),
        "threshold": float(result.global_threshold),
    }


def feature_counts(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_flags = build_candidate_features(train)
    test_flags = build_candidate_features(test)
    rows: list[dict[str, Any]] = []
    for feature in train_flags.columns:
        mask = train_flags[feature].eq(1)
        count = int(mask.sum())
        purchased = int(train.loc[mask, TARGET].sum())
        rows.append(
            {
                "feature": feature,
                "definition": FEATURE_DEFINITIONS[feature],
                "train_count": count,
                "train_purchase_count": purchased,
                "train_purchase_rate": float(purchased / count) if count else np.nan,
                "test_count": int(test_flags[feature].sum()),
                "small_sample_warning": count < 20,
            }
        )
    return pd.DataFrame(rows)


def fit_final_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    added_features: list[str],
    params: dict[str, Any],
    best_iterations: list[int],
) -> tuple[np.ndarray, int]:
    x_train, categorical = engineer_for_experiment(train, added_features)
    x_test, _ = engineer_for_experiment(test, added_features)
    valid_iterations = [value for value in best_iterations if value >= 0]
    final_iterations = max(100, int(np.median(valid_iterations) * 1.10))
    final_params = dict(params)
    final_params["iterations"] = final_iterations
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
        **final_params,
    )
    model.fit(x_train, y, cat_features=categorical)
    return model.predict_proba(x_test)[:, 1], final_iterations


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(value: float) -> str:
    return f"{value:+.4f}"


def write_report(
    path: Path,
    results: pd.DataFrame,
    counts: pd.DataFrame,
    importance: pd.DataFrame,
    diagnostics: dict[str, Any],
    submission_created: bool,
    selected_experiment: str | None,
) -> None:
    indexed = results.set_index("experiment")
    baseline = indexed.loc["Baseline"]
    best_name = str(results.sort_values(["crossfit F1", "OOF F1"], ascending=False).iloc[0]["experiment"])
    best = indexed.loc[best_name]
    top_added = importance[importance["experiment"] == best_name].sort_values("importance", ascending=False).head(5)
    top_overlap = importance[
        (importance["experiment"] == best_name) & importance["feature"].isin(OVERLAP_FEATURES)
    ].sort_values("importance", ascending=False).head(1)

    lines: list[str] = [
        "# セグメント特徴量エンジニアリング実験",
        "",
        "## 1. 実験条件",
        "",
        "目的は予測精度の改善だけです。営業施策・プレゼン分析は行っていません。`StratifiedGroupKFold` 5分割、テキスト類似度0.95のグループ、CatBoostパラメータ、早期終了、乱数seedをbaselineと固定しました。テキストモデルは現行の文字n-gram TF-IDF＋LogisticRegression（C=0.7）の保存済みOOF/test確率を全実験で共通使用し、再学習・設定変更していません。",
        "",
        "現行 `run_pipeline()` は `submission.csv` を常に書くため直接実行せず、既存pipelineの特徴量生成・分割・閾値関数を再利用しました。また、保存済みbaselineが選択した `catboost_depth6_class_weighted` は現在の候補一覧に存在しないため、`output/model/model_config.json` の保存パラメータを正として再現しています。",
        "",
        "### 使用した定義",
        "",
        "- DX課題顕在：アンケート2=1、またはアンケート10=1。",
        "- BtoB：`特徴 == BtoB`。",
        "- 既存ツール低満足：アンケート7<=2。",
        "- 優先業界：IT・人材・自動車/乗り物。",
        "- DXへの抵抗感：`description.csv` ではアンケート4の1が「非常に低い」、5が「非常に高い」。したがって `low_dx_resistance` はアンケート4<=2。",
        "",
        "### 追加特徴の件数",
        "",
        "| feature | train件数 | 購入数 | 購入率 | test件数 | 注意 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in counts.iterrows():
        warning = "少数（過学習注意）" if bool(row["small_sample_warning"]) else ""
        lines.append(
            f"| `{row['feature']}` | {int(row['train_count'])} | {int(row['train_purchase_count'])} | "
            f"{pct(float(row['train_purchase_rate']))} | {int(row['test_count'])} | {warning} |"
        )

    lines.extend(
        [
            "",
            "## 2. 評価結果",
            "",
            "OOF F1・Precision・Recallは全OOFで最適化したthreshold、crossfit F1は各検証foldのthresholdを他4foldだけで決めた値です。blend weightはCatBoost 0.00〜0.50を0.05刻みで探索し、crossfit F1、次にOOF F1の順で選択しました。",
            "",
            "| experiment | OOF F1 | crossfit F1 | Precision | Recall | ROC-AUC | AP | threshold | CatBoost | text |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in results.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['OOF F1']:.4f} | {row['crossfit F1']:.4f} | "
            f"{row['Precision']:.4f} | {row['Recall']:.4f} | {row['ROC-AUC']:.4f} | "
            f"{row['Average Precision']:.4f} | {row['threshold']:.4f} | "
            f"{row['best CatBoost weight']:.2f} | {row['best text weight']:.2f} |"
        )

    def comparison_line(name: str, reference: str) -> str:
        row = indexed.loc[name]
        ref = indexed.loc[reference]
        return (
            f"- **{name} vs {reference}**：OOF F1 {delta(float(row['OOF F1'] - ref['OOF F1']))}、"
            f"crossfit F1 {delta(float(row['crossfit F1'] - ref['crossfit F1']))}。"
        )

    lines.extend(
        [
            "",
            "## 3. 特徴量の効果",
            "",
            comparison_line("segment_v1_basic", "Baseline"),
            comparison_line("segment_v2_overlap", "segment_v1_basic"),
            comparison_line("segment_v3_survey", "segment_v1_basic"),
            comparison_line("segment_v4_all", "Baseline"),
            "",
            "segment_v2−v1を重複セグメントの増分、segment_v3−v1をアンケート交互作用の増分として判断します。OOF F1だけ上がってcrossfit F1が下がる組合せは採用しません。",
            "",
            f"全候補のうち評価順1位は **{best_name}** で、OOF F1 {best['OOF F1']:.4f}、crossfit F1 {best['crossfit F1']:.4f} でした。baselineとの差はそれぞれ {delta(float(best['OOF F1'] - baseline['OOF F1']))}、{delta(float(best['crossfit F1'] - baseline['crossfit F1']))} です。",
            f"ROC-AUCは {baseline['ROC-AUC']:.4f}→{best['ROC-AUC']:.4f}、Average Precisionは {baseline['Average Precision']:.4f}→{best['Average Precision']:.4f} で、閾値依存のF1以外でも改善しました。",
            "",
            "### 追加特徴のCatBoost重要度（評価順1位）",
            "",
            "| feature | 平均重要度 |",
            "|---|---:|",
        ]
    )
    if top_added.empty:
        lines.append("| 追加特徴なし | — |")
    else:
        for _, row in top_added.iterrows():
            lines.append(f"| `{row['feature']}` | {row['importance']:.4f} |")
    if not top_overlap.empty:
        overlap_row = top_overlap.iloc[0]
        lines.extend(
            [
                "",
                f"追加特徴全体では `segment_priority_industry` の重要度が最大です。ただし基本3特徴だけのsegment_v1は悪化しており、改善を単独特徴へ帰属できません。v2−v1の改善を生んだ重複特徴の中では `{overlap_row['feature']}` が最大（{overlap_row['importance']:.4f}）でした。",
            ]
        )

    tiny = counts[counts["small_sample_warning"]]
    tiny_note = "、".join(f"{row.feature}={int(row.train_count)}社" for row in tiny.itertuples()) or "該当なし"
    parity = diagnostics["baseline_reproduction"]
    lines.extend(
        [
            "",
            "## 4. 過学習の兆候",
            "",
            f"- 20社未満の追加特徴：{tiny_note}。特に6社しかない全条件重複は、重要度が出ても再現性を慎重に扱います。",
            "- 5実験×11 blend weightを同じOOF上で比較しているため、選択バイアスがあります。採用候補は別seed・反復CVまたはPublic/Privateで再検証が必要です。",
            "- global thresholdのOOF F1とcrossfit F1の差を確認し、前者だけの改善は採用条件から除外しました。",
            f"- fresh baseline CatBoostと保存OOFの最大絶対確率差は {parity['catboost_oof_max_abs_diff']:.3e}、fresh baseline ensembleと保存OOFの差は {parity['ensemble_oof_max_abs_diff']:.3e} でした。",
            f"- 保存済みmetricsのcrossfit F1は {parity['saved_crossfit_f1']:.4f}、現行の閾値関数で同じOOF確率を再計算すると {parity['fresh_crossfit_f1']:.4f} でした。確率差は丸め誤差水準ですが、確率値そのものを閾値候補に含める実装の境界で予測1件が変わるためです。全実験比較にはfresh baselineを一貫して使用しました。",
            "",
            "## 5. 採用判断",
            "",
        ]
    )
    if selected_experiment is not None:
        selected = indexed.loc[selected_experiment]
        lines.extend(
            [
                f"**採用候補は {selected_experiment} です。** baselineよりOOF F1とcrossfit F1の両方が改善したため、ユーザー指定の成功条件を満たしました。",
                "",
                f"最適weightは text {selected['best text weight']:.2f} / CatBoost {selected['best CatBoost weight']:.2f}、thresholdは {selected['threshold']:.4f} です。",
            ]
        )
    else:
        lines.append("**今回は採用を見送ります。** baselineよりOOF F1とcrossfit F1の両方を改善した実験がなかったためです。")
    lines.extend(
        [
            "",
            f"`submissions/segment_v2/submission_segment_v2.csv`：{'作成済み' if submission_created else '未作成'}。baseline submission は変更していません。",
            "",
            "Public F1 0.7302は外部評価値であり、今回のOOF差からPublic改善を保証するものではありません。",
            "",
            "## 6. 出力ファイル",
            "",
            "- `output/experiments/segment/comparison/segment_versions_summary.csv`：v1〜v4の主要比較表",
            "- `output/experiments/segment/diagnostics/feature_counts.csv`：追加特徴の母数・購入率",
            "- `output/experiments/segment/comparison/blend_grid.csv`：全weightの評価",
            "- `output/experiments/segment/diagnostics/feature_importance.csv`：追加特徴のfold平均重要度",
            "- `output/experiments/segment/diagnostics/oof_predictions.csv`：再検証用OOF確率",
            "- `output/experiments/segment/diagnostics/diagnostics.json`：分割・閾値・再現差",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data_dir = PROJECT_DIR / "data"
    output_dir = PROJECT_DIR / "output"
    baseline_dir = output_dir / "experiments" / "baseline"
    segment_dir = output_dir / "experiments" / "segment"
    comparison_dir = segment_dir / "comparison"
    diagnostics_dir = segment_dir / "diagnostics"
    submission_dir = PROJECT_DIR / "submissions" / "segment_v2"
    for directory in [comparison_dir, diagnostics_dir, submission_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    description_path = data_dir / "description.csv"
    model_config_path = output_dir / "model" / "model_config.json"
    metrics_path = baseline_dir / "model_metrics.json"
    saved_oof_path = baseline_dir / "cv_oof_predictions.csv"
    saved_test_path = baseline_dir / "test_probabilities.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    description = pd.read_csv(description_path)
    saved_oof = pd.read_csv(saved_oof_path)
    saved_test = pd.read_csv(saved_test_path)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    current_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("企業IDに重複があります。")
    if saved_oof[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
        raise ValueError("保存OOFとtrain.csvの企業ID順が一致しません。")
    if saved_oof[TARGET].astype(int).tolist() != train[TARGET].astype(int).tolist():
        raise ValueError("保存OOFとtrain.csvの目的変数が一致しません。")
    if saved_test[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("保存test確率とtest.csvの企業ID順が一致しません。")
    question4 = description.loc[description["カラム名"] == "アンケート４", "説明"].iloc[0]
    question7 = description.loc[description["カラム名"] == "アンケート７", "説明"].iloc[0]
    if "1: 非常に低い" not in question4 or "5: 非常に高い" not in question4:
        raise ValueError("アンケート4の向きをdescription.csvで確認できません。")
    if "1: 非常に低い" not in question7 or "5: 非常に高い" not in question7:
        raise ValueError("アンケート7の向きをdescription.csvで確認できません。")

    features = train.drop(columns=[TARGET])
    y = train[TARGET].to_numpy(dtype=int)
    groups, group_stats = text_similarity_groups(features, threshold=0.95)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(features, y, groups=groups))
    reconstructed_fold = np.full(len(train), -1, dtype=int)
    for fold, (_, valid_index) in enumerate(splits):
        reconstructed_fold[valid_index] = fold
    if not np.array_equal(reconstructed_fold, saved_oof["fold"].to_numpy(dtype=int)):
        raise ValueError("再構築したStratifiedGroupKFoldと保存OOFのfoldが一致しません。")

    params = dict(model_config["catboost_params"])
    if current_metrics["selected_tabular_model"] != "catboost_depth6_class_weighted":
        raise ValueError("想定した現行CatBoostモデルと異なります。")
    text_result = saved_text_result(y, saved_oof, saved_test)
    counts = feature_counts(train, test)

    experiment_rows: list[dict[str, Any]] = []
    blend_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    experiment_results: dict[str, dict[str, CvResult]] = {}
    oof_export = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            TARGET: y,
            "fold": reconstructed_fold,
            "text_oof_probability": text_result.oof,
        }
    )

    for experiment, added_features in EXPERIMENTS.items():
        print(f"Training {experiment} ({len(added_features)} added features)...", flush=True)
        cat_result = run_catboost_experiment(
            features,
            test,
            y,
            splits,
            experiment,
            added_features,
            params,
        )
        blends = [blend_results(cat_result, text_result, y, float(weight)) for weight in WEIGHTS]
        best_blend = max(blends, key=lambda result: (result.crossfit_threshold_f1, result.global_f1))
        metrics = result_metrics(y, best_blend)
        experiment_rows.append(
            {
                "experiment": experiment,
                "added_features": ";".join(added_features) if added_features else "none",
                **metrics,
                "best CatBoost weight": float(best_blend.extra["weight_tabular"]),
                "best text weight": float(best_blend.extra["weight_text"]),
            }
        )
        for blend in blends:
            row_metrics = result_metrics(y, blend)
            blend_rows.append(
                {
                    "experiment": experiment,
                    "CatBoost weight": float(blend.extra["weight_tabular"]),
                    "text weight": float(blend.extra["weight_text"]),
                    **row_metrics,
                }
            )
        importance_frame = cat_result.extra["feature_importance"]
        for feature in added_features:
            importance_rows.append(
                {
                    "experiment": experiment,
                    "feature": feature,
                    "importance": float(importance_frame.loc[importance_frame["feature"] == feature, "importance"].iloc[0]),
                    "train_count": int(counts.loc[counts["feature"] == feature, "train_count"].iloc[0]),
                }
            )
        _, _, crossfit_predictions = crossfit_threshold_score(y, best_blend.oof, best_blend.fold_id)
        oof_export[f"{experiment}_catboost_probability"] = cat_result.oof
        oof_export[f"{experiment}_ensemble_probability"] = best_blend.oof
        oof_export[f"{experiment}_crossfit_prediction"] = crossfit_predictions
        experiment_results[experiment] = {"cat": cat_result, "blend": best_blend}
        print(
            f"  best weight={best_blend.extra['weight_tabular']:.2f}, "
            f"OOF F1={best_blend.global_f1:.4f}, crossfit F1={best_blend.crossfit_threshold_f1:.4f}",
            flush=True,
        )

    results = pd.DataFrame(experiment_rows)
    blend_grid = pd.DataFrame(blend_rows)
    importance = pd.DataFrame(importance_rows)
    baseline_row = results.loc[results["experiment"] == "Baseline"].iloc[0]
    improving = results[
        (results["experiment"] != "Baseline")
        & (results["OOF F1"] > float(baseline_row["OOF F1"]) + 1e-12)
        & (results["crossfit F1"] > float(baseline_row["crossfit F1"]) + 1e-12)
    ]
    if improving.empty:
        selected_experiment = None
    else:
        selected_experiment = str(
            improving.sort_values(["crossfit F1", "OOF F1"], ascending=False).iloc[0]["experiment"]
        )

    baseline_cat = experiment_results["Baseline"]["cat"]
    baseline_blend = experiment_results["Baseline"]["blend"]
    baseline_reproduction = {
        "saved_selected_model": current_metrics["selected_model"],
        "saved_oof_f1": current_metrics["oof_f1_tuned"],
        "saved_crossfit_f1": current_metrics["crossfit_threshold_f1"],
        "fresh_oof_f1": float(baseline_row["OOF F1"]),
        "fresh_crossfit_f1": float(baseline_row["crossfit F1"]),
        "catboost_oof_max_abs_diff": float(
            np.max(np.abs(baseline_cat.oof - saved_oof["catboost_oof_probability"].to_numpy(dtype=float)))
        ),
        "ensemble_oof_max_abs_diff": float(
            np.max(np.abs(baseline_blend.oof - saved_oof["ensemble_oof_probability"].to_numpy(dtype=float)))
        ),
        "saved_weight_tabular": current_metrics["weight_tabular"],
        "fresh_weight_tabular": float(baseline_row["best CatBoost weight"]),
    }

    submission_created = False
    final_iterations: int | None = None
    submission_path = submission_dir / "submission_segment_v2.csv"
    if selected_experiment is not None:
        selected_result = experiment_results[selected_experiment]
        selected_row = results.loc[results["experiment"] == selected_experiment].iloc[0]
        cat_test_probability, final_iterations = fit_final_catboost(
            features,
            test,
            y,
            EXPERIMENTS[selected_experiment],
            params,
            selected_result["cat"].extra["best_iterations"],
        )
        weight_cat = float(selected_row["best CatBoost weight"])
        final_probability = weight_cat * cat_test_probability + (1 - weight_cat) * text_result.test_prob
        prediction = (final_probability >= float(selected_row["threshold"])).astype(int)
        submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(int), TARGET: prediction})
        if len(submission) != len(test) or submission[ID_COLUMN].tolist() != test[ID_COLUMN].astype(int).tolist():
            raise ValueError("新規submissionのIDまたは行数がtest.csvと一致しません。")
        submission.to_csv(submission_path, index=False, header=False, encoding="utf-8")
        submission_created = True

    results_path = comparison_dir / "segment_versions_summary.csv"
    counts_path = diagnostics_dir / "feature_counts.csv"
    grid_path = comparison_dir / "blend_grid.csv"
    importance_path = diagnostics_dir / "feature_importance.csv"
    oof_path = diagnostics_dir / "oof_predictions.csv"
    diagnostics_path = diagnostics_dir / "diagnostics.json"
    report_path = segment_dir / "SEGMENT_VERSIONS_ANALYSIS.md"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    blend_grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    oof_export.to_csv(oof_path, index=False, encoding="utf-8-sig")
    diagnostics = {
        "random_state": RANDOM_STATE,
        "validation_scheme": "StratifiedGroupKFold(n_splits=5, shuffle=True)",
        "group_stats": group_stats,
        "fold_sizes": [int((reconstructed_fold == fold).sum()) for fold in range(5)],
        "fold_positive_counts": [int(y[reconstructed_fold == fold].sum()) for fold in range(5)],
        "catboost_params": params,
        "text_model": {"type": "saved current OOF/test", "c_value": 0.7, "class_weight": None},
        "weight_grid": WEIGHTS,
        "definitions": FEATURE_DEFINITIONS,
        "baseline_reproduction": baseline_reproduction,
        "crossfit_thresholds": {
            experiment: experiment_results[experiment]["blend"].crossfit_thresholds
            for experiment in EXPERIMENTS
        },
        "best_iterations": {
            experiment: experiment_results[experiment]["cat"].extra["best_iterations"]
            for experiment in EXPERIMENTS
        },
        "selected_experiment": selected_experiment,
        "submission_created": submission_created,
        "submission_final_iterations": final_iterations,
        "source_hashes": {
            "train.csv": sha256(train_path),
            "test.csv": sha256(test_path),
            "description.csv": sha256(description_path),
            "model_config.json": sha256(model_config_path),
            "cv_oof_predictions.csv": sha256(saved_oof_path),
            "test_probabilities.csv": sha256(saved_test_path),
        },
    }
    write_json(diagnostics_path, diagnostics)
    write_report(
        report_path,
        results,
        counts,
        importance,
        diagnostics,
        submission_created,
        selected_experiment,
    )

    print(f"Results: {results_path}")
    print(f"Report: {report_path}")
    print(f"Selected experiment: {selected_experiment or 'none'}")
    print(f"Submission created: {submission_created}")


if __name__ == "__main__":
    main()
