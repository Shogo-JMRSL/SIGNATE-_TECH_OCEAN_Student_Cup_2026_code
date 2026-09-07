from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_DIR))

from src.competition_pipeline import (  # noqa: E402
    ID_COLUMN,
    RANDOM_STATE,
    TARGET,
    best_f1_threshold,
    crossfit_threshold_score,
    text_similarity_groups,
)


MODEL_SPECS: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("企業概要 only", ("企業概要",)),
        ("組織図 only", ("組織図",)),
        ("今後のDX展望 only", ("今後のDX展望",)),
        ("特徴 only", ("特徴",)),
        ("業界 only", ("業界",)),
        ("上場種別 only", ("上場種別",)),
        ("企業概要 + 組織図", ("企業概要", "組織図")),
        ("企業概要 + 今後のDX展望", ("企業概要", "今後のDX展望")),
        ("組織図 + 今後のDX展望", ("組織図", "今後のDX展望")),
        (
            "企業概要 + 組織図 + 今後のDX展望",
            ("企業概要", "組織図", "今後のDX展望"),
        ),
        (
            "current full-text baseline",
            ("業界", "上場種別", "特徴", "企業概要", "組織図", "今後のDX展望"),
        ),
    ]
)

PROBABILITY_COLUMNS = {
    "企業概要 only": "企業概要_probability",
    "組織図 only": "組織図_probability",
    "今後のDX展望 only": "今後のDX展望_probability",
    "特徴 only": "特徴_probability",
    "業界 only": "業界_probability",
    "上場種別 only": "上場種別_probability",
    "企業概要 + 組織図": "企業概要_組織図_probability",
    "企業概要 + 今後のDX展望": "企業概要_今後のDX展望_probability",
    "組織図 + 今後のDX展望": "組織図_今後のDX展望_probability",
    "企業概要 + 組織図 + 今後のDX展望": "主要3テキスト_probability",
    "current full-text baseline": "current_full_text_baseline_probability",
}

SINGLE_MODELS = list(MODEL_SPECS)[:6]
COMBINATION_MODELS = list(MODEL_SPECS)[6:10]
BASELINE_MODEL = "current full-text baseline"


@dataclass
class ExperimentResult:
    model: str
    columns: tuple[str, ...]
    oof: np.ndarray
    fold_id: np.ndarray
    crossfit_predictions: np.ndarray
    threshold: float
    oof_f1: float
    crossfit_f1: float
    crossfit_thresholds: list[float]


def build_selected_text(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    pieces = ["【" + column + "】" + df[column].fillna("").astype(str) for column in columns]
    return pd.concat(pieces, axis=1).agg("\n".join, axis=1)


def make_model() -> Pipeline:
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
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_experiment(
    model_name: str,
    columns: tuple[str, ...],
    features: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> ExperimentResult:
    text = build_selected_text(features, columns)
    oof = np.zeros(len(features), dtype=float)
    fold_id = np.full(len(features), -1, dtype=int)

    for fold, (train_index, valid_index) in enumerate(splits):
        model = make_model()
        model.fit(text.iloc[train_index], y[train_index])
        oof[valid_index] = model.predict_proba(text.iloc[valid_index])[:, 1]
        fold_id[valid_index] = fold

    if np.any(fold_id < 0):
        raise RuntimeError(f"OOF prediction is incomplete for {model_name}")

    threshold, oof_f1 = best_f1_threshold(y, oof)
    crossfit_f1, crossfit_thresholds, crossfit_predictions = crossfit_threshold_score(
        y, oof, fold_id
    )
    return ExperimentResult(
        model=model_name,
        columns=columns,
        oof=oof,
        fold_id=fold_id,
        crossfit_predictions=crossfit_predictions,
        threshold=threshold,
        oof_f1=oof_f1,
        crossfit_f1=crossfit_f1,
        crossfit_thresholds=crossfit_thresholds,
    )


def comparison_row(result: ExperimentResult, y: np.ndarray) -> dict[str, object]:
    tuned_predictions = (result.oof >= result.threshold).astype(int)
    row: dict[str, object] = {
        "model": result.model,
        "text_columns": " | ".join(result.columns),
        "oof_f1": result.oof_f1,
        "crossfit_f1": result.crossfit_f1,
        "precision": float(precision_score(y, tuned_predictions, zero_division=0)),
        "recall": float(recall_score(y, tuned_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, result.oof)),
        "average_precision": float(average_precision_score(y, result.oof)),
        "threshold": result.threshold,
    }
    for fold in sorted(np.unique(result.fold_id)):
        mask = result.fold_id == fold
        row[f"fold_{fold}_f1"] = float(
            f1_score(y[mask], result.crossfit_predictions[mask], zero_division=0)
        )
        row[f"fold_{fold}_roc_auc"] = float(roc_auc_score(y[mask], result.oof[mask]))
        row[f"fold_{fold}_threshold"] = float(result.crossfit_thresholds[int(fold)])
    return row


def fmt(value: float) -> str:
    return f"{value:.4f}"


def model_summary_line(row: pd.Series) -> str:
    return (
        f"- **{row.name}**: OOF F1 {fmt(row['oof_f1'])}, "
        f"cross-fit F1 {fmt(row['crossfit_f1'])}, ROC-AUC {fmt(row['roc_auc'])}"
    )


def choose_ensemble_candidates(comparison: pd.DataFrame) -> list[str]:
    best_single = comparison[comparison["model"].isin(SINGLE_MODELS)].iloc[0]["model"]
    best_combo = comparison[comparison["model"].isin(COMBINATION_MODELS)].iloc[0]["model"]
    candidates = [str(best_single), str(best_combo)]
    for model in comparison.sort_values(
        ["crossfit_f1", "roc_auc", "oof_f1"], ascending=False
    )["model"]:
        if model not in candidates:
            candidates.append(str(model))
        if len(candidates) == 3:
            break
    return candidates


def write_analysis(
    path: Path,
    comparison: pd.DataFrame,
    group_stats: dict[str, object],
) -> None:
    singles = comparison[comparison["model"].isin(SINGLE_MODELS)]
    combinations = comparison[comparison["model"].isin(COMBINATION_MODELS)]
    best_single = singles.iloc[0]
    weakest_single = singles.iloc[-1]
    best_combo = combinations.iloc[0]
    baseline = comparison.loc[comparison["model"] == BASELINE_MODEL].iloc[0]
    outlook = comparison.loc[comparison["model"] == "今後のDX展望 only"].iloc[0]
    candidates = choose_ensemble_candidates(comparison)

    outlook_delta = float(outlook["oof_f1"] - baseline["oof_f1"])
    combo_delta = float(best_combo["oof_f1"] - baseline["oof_f1"])
    if baseline["oof_f1"] > best_single["oof_f1"]:
        full_text_single_statement = (
            f"全文結合は最強単独列よりOOF F1が{fmt(float(baseline['oof_f1'] - best_single['oof_f1']))}高く、"
            "単独列との比較では性能が上がりました。"
        )
    else:
        full_text_single_statement = (
            f"全文結合は最強単独列よりOOF F1が{fmt(float(best_single['oof_f1'] - baseline['oof_f1']))}低く、"
            "有効な情報が他列で薄まった可能性があります。"
        )
    if combo_delta > 0:
        full_text_combo_statement = (
            f"最強の主要列組み合わせは全文結合を{fmt(combo_delta)}上回りました。"
        )
    elif combo_delta < 0:
        full_text_combo_statement = (
            f"全文結合は最強の主要列組み合わせを{fmt(-combo_delta)}上回りました。"
        )
    else:
        full_text_combo_statement = "全文結合と最強の主要列組み合わせのOOF F1は同値でした。"

    table_columns = [
        "model",
        "oof_f1",
        "crossfit_f1",
        "precision",
        "recall",
        "roc_auc",
        "average_precision",
        "threshold",
    ]
    table_lines = [
        "| model | OOF F1 | cross-fit F1 | Precision | Recall | ROC-AUC | Average Precision | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison[table_columns].iterrows():
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    fmt(row["oof_f1"]),
                    fmt(row["crossfit_f1"]),
                    fmt(row["precision"]),
                    fmt(row["recall"]),
                    fmt(row["roc_auc"]),
                    fmt(row["average_precision"]),
                    fmt(row["threshold"]),
                ]
            )
            + " |"
        )

    candidate_rows = comparison.set_index("model").loc[candidates]
    candidate_lines = [model_summary_line(row) for _, row in candidate_rows.iterrows()]

    report = f"""# Phase 1 テキスト列分離実験

## 結論

- 最も強かった単独テキスト列は **{best_single['model']}** でした（OOF F1 {fmt(best_single['oof_f1'])}、cross-fit F1 {fmt(best_single['crossfit_f1'])}、ROC-AUC {fmt(best_single['roc_auc'])}）。
- 最も弱かった単独テキスト列は **{weakest_single['model']}** でした（OOF F1 {fmt(weakest_single['oof_f1'])}）。
- 最も強かった主要列の組み合わせは **{best_combo['model']}** でした（OOF F1 {fmt(best_combo['oof_f1'])}、cross-fit F1 {fmt(best_combo['crossfit_f1'])}、ROC-AUC {fmt(best_combo['roc_auc'])}）。
- 現在の全文モデルはOOF F1 {fmt(baseline['oof_f1'])}、cross-fit F1 {fmt(baseline['crossfit_f1'])}、ROC-AUC {fmt(baseline['roc_auc'])}でした。

依頼文にあるOOF F1約0.71は既存のCatBoost＋全文テキストのブレンド値です。本Phase 1では入力テキスト列だけの差を比較するため、固定したTF-IDF＋LogisticRegressionによる全文テキスト単体をbaselineとしています。

## 今後のDX展望の強さ

`今後のDX展望 only` はOOF F1 {fmt(outlook['oof_f1'])}、cross-fit F1 {fmt(outlook['crossfit_f1'])}、ROC-AUC {fmt(outlook['roc_auc'])}、Average Precision {fmt(outlook['average_precision'])}でした。全文モデルとの差は {outlook_delta:+.4f} です。

## 全文結合の効果

{full_text_single_statement} {full_text_combo_statement}

## 比較表（OOF F1降順）

{chr(10).join(table_lines)}

## 第二段階のアンサンブル候補

{chr(10).join(candidate_lines)}

候補は、単独列の最良モデル、主要列組み合わせの最良モデル、およびcross-fit F1・ROC-AUCが高い残りのモデルから選びました。今回のPhase 1ではアンサンブル学習およびsubmission作成は行っていません。

## 検証条件と注意点

- CV: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})`
- 類似テキストgroup: 既存の全文結合に対して類似度0.95以上を同一groupに固定（group数 {group_stats['group_count']}、最大group {group_stats['largest_group_size']}行）
- 全モデルでTF-IDFとLogisticRegressionの設定を固定し、入力列だけを変更しました。
- OOF F1、Precision、Recallは全OOFから選んだ最適thresholdで計算しています。cross-fit F1は各fold以外のOOFだけでthresholdを選んでいます。
- fold別F1はcross-fit threshold、fold別ROC-AUCは確率から計算し、比較CSVに保存しています。
- 同じCVで11モデルを比較する探索的分析のため、僅差は確定的な優劣ではありません。第二段階では保存済みOOF予測で補完性と安定性を再検証してください。
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 text-column experiments")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR,
        help="SIGNATE project directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    train_path = project_dir / "data" / "train.csv"
    output_dir = project_dir / "output" / "analysis" / "text"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    required_columns = {ID_COLUMN, TARGET, *(column for columns in MODEL_SPECS.values() for column in columns)}
    missing_columns = sorted(required_columns - set(train.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    y = train[TARGET].to_numpy(dtype=int)
    features = train.drop(columns=[TARGET])
    groups, group_stats = text_similarity_groups(features, threshold=0.95)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(features, y, groups=groups))

    results: list[ExperimentResult] = []
    for index, (model_name, columns) in enumerate(MODEL_SPECS.items(), start=1):
        print(f"[{index:02d}/{len(MODEL_SPECS)}] {model_name}", flush=True)
        results.append(run_experiment(model_name, columns, features, y, splits))

    rows = [comparison_row(result, y) for result in results]
    comparison = pd.DataFrame(rows).sort_values(
        ["oof_f1", "crossfit_f1", "roc_auc"],
        ascending=False,
        kind="stable",
    )
    comparison_path = output_dir / "phase1_text_column_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    fold_id = results[0].fold_id
    if any(not np.array_equal(result.fold_id, fold_id) for result in results[1:]):
        raise RuntimeError("Fold IDs differ across models")
    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET: y,
            "fold": fold_id,
        }
    )
    for result in results:
        oof_frame[PROBABILITY_COLUMNS[result.model]] = result.oof
    oof_path = output_dir / "phase1_text_oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False, encoding="utf-8-sig")

    analysis_path = output_dir / "PHASE1_TEXT_ANALYSIS.md"
    write_analysis(analysis_path, comparison.reset_index(drop=True), group_stats)

    best_single = comparison[comparison["model"].isin(SINGLE_MODELS)].iloc[0]
    best_combo = comparison[comparison["model"].isin(COMBINATION_MODELS)].iloc[0]
    baseline = comparison.loc[comparison["model"] == BASELINE_MODEL].iloc[0]
    print(f"Baseline OOF F1: {baseline['oof_f1']:.4f}")
    print(f"Best single: {best_single['model']} ({best_single['oof_f1']:.4f})")
    print(f"Best combination: {best_combo['model']} ({best_combo['oof_f1']:.4f})")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {oof_path}")
    print(f"Wrote: {analysis_path}")


if __name__ == "__main__":
    main()
