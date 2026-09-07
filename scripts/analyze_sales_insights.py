from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, ticker


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "購入フラグ"
ID_COLUMN = "企業ID"
PROBABILITY_COLUMN = "ensemble_oof_probability"
CATEGORICAL_COLUMNS = ["業界", "上場種別", "特徴"]
NUMERIC_COLUMNS = [
    "従業員数",
    "売上",
    "営業利益",
    "総資産",
    "資本金",
    "事業所数",
    "店舗数",
    "工場数",
]
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
TEXT_COLUMNS = ["企業概要", "組織図", "今後のDX展望"]
KEYWORDS = [
    "DX推進",
    "情報システム",
    "IT",
    "デジタル",
    "人事",
    "教育",
    "研修",
    "人材育成",
    "リスキリング",
    "AI",
    "データ分析",
    "クラウド",
    "IoT",
    "業務改善",
    "自動化",
    "人材不足",
    "積極",
    "慎重",
    "段階的",
    "ROI",
    "費用対効果",
    "eラーニング",
]
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\BIZ-UDGothicR.ttc"),
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
]


def write_csv_if_changed(frame: pd.DataFrame, path: Path) -> bool:
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    content = buffer.getvalue()
    if path.exists() and path.read_bytes() == content:
        return False
    path.write_bytes(content)
    return True


def configure_japanese_font() -> tuple[str, Path]:
    font_path = next((path for path in FONT_CANDIDATES if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("日本語フォントが見つかりません。")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#667085",
            "axes.labelcolor": "#202733",
            "text.color": "#202733",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
        }
    )
    return font_name, font_path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_groups(
    train: pd.DataFrame,
    variable: str,
    labels: pd.Series,
    ordered_labels: list[str],
    analysis_type: str,
) -> pd.DataFrame:
    overall_rate = float(train[TARGET].mean())
    rows: list[dict[str, object]] = []
    for order, label in enumerate(ordered_labels):
        mask = labels == label
        total = int(mask.sum())
        if total == 0:
            continue
        purchased = int(train.loc[mask, TARGET].sum())
        rate = purchased / total
        low, high = wilson_interval(purchased, total)
        rows.append(
            {
                "分析種別": analysis_type,
                "変数": variable,
                "区分": label,
                "区分順": order,
                "企業数": total,
                "購入企業数": purchased,
                "非購入企業数": total - purchased,
                "購入率": rate,
                "全体比差": rate - overall_rate,
                "Wilson下限95": low,
                "Wilson上限95": high,
                "安定比較対象": total >= 20,
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(train: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in CATEGORICAL_COLUMNS:
        labels = train[column].fillna("欠損").astype(str)
        counts = labels.value_counts()
        ordered = counts.index.tolist()
        frames.append(summarize_groups(train, column, labels, ordered, "カテゴリ"))
    return pd.concat(frames, ignore_index=True)


def format_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}百万"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}千"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def quantile_labels(series: pd.Series) -> tuple[pd.Series, list[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    nonmissing = numeric.dropna()
    _, boundaries = pd.qcut(nonmissing, q=4, retbins=True, duplicates="drop")
    cut = pd.cut(numeric, bins=boundaries, include_lowest=True, duplicates="drop")
    output = pd.Series(index=series.index, dtype=object)
    ordered: list[str] = []
    for index, interval in enumerate(cut.cat.categories, start=1):
        label = f"Q{index}: {format_number(interval.left)}–{format_number(interval.right)}"
        output.loc[cut == interval] = label
        ordered.append(label)
    if numeric.isna().any():
        output.loc[numeric.isna()] = "欠損"
        ordered.append("欠損")
    return output, ordered


def numeric_summary(train: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in NUMERIC_COLUMNS:
        labels, ordered = quantile_labels(train[column])
        frames.append(summarize_groups(train, column, labels, ordered, "数値四分位"))
    return pd.concat(frames, ignore_index=True)


def format_answer(value: object) -> str:
    if pd.isna(value):
        return "欠損"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def survey_summary(train: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in SURVEY_COLUMNS:
        values = pd.to_numeric(train[column], errors="coerce")
        answers = sorted(values.dropna().unique())
        ordered = [format_answer(value) for value in answers]
        labels = values.map(format_answer)
        if values.isna().any():
            ordered.append("欠損")
        frames.append(summarize_groups(train, column, labels, ordered, "アンケート"))
    return pd.concat(frames, ignore_index=True)


def keyword_summary(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in TEXT_COLUMNS:
        text = train[column].fillna("").astype(str)
        for keyword in KEYWORDS:
            present = text.str.contains(keyword, case=False, regex=False)
            present_count = int(present.sum())
            absent_count = int((~present).sum())
            if present_count == 0:
                continue
            purchased = int(train.loc[present, TARGET].sum())
            absent_purchased = int(train.loc[~present, TARGET].sum())
            rate = purchased / present_count
            absent_rate = absent_purchased / absent_count if absent_count else float("nan")
            low, high = wilson_interval(purchased, present_count)
            rows.append(
                {
                    "テキスト列": column,
                    "キーワード": keyword,
                    "該当企業数": present_count,
                    "該当購入企業数": purchased,
                    "該当購入率": rate,
                    "非該当企業数": absent_count,
                    "非該当購入企業数": absent_purchased,
                    "非該当購入率": absent_rate,
                    "購入率差": rate - absent_rate,
                    "Wilson下限95": low,
                    "Wilson上限95": high,
                    "安定比較対象": present_count >= 20 and absent_count >= 20,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["安定比較対象", "購入率差", "該当企業数"], ascending=[False, False, False]
    )


def text_length_summary(train: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in TEXT_COLUMNS:
        length = train[column].fillna("").astype(str).str.len()
        labels, ordered = quantile_labels(length)
        frames.append(summarize_groups(train, f"{column}_文字数", labels, ordered, "テキスト長"))
    return pd.concat(frames, ignore_index=True)


def segment_stats(
    train: pd.DataFrame,
    name: str,
    rule: str,
    mask: pd.Series,
    priority: str,
    clarity: str,
    threshold: float,
) -> dict[str, object]:
    selected = train.loc[mask]
    unpurchased = selected[selected[TARGET] == 0]
    unpurchased_mean = float(unpurchased[PROBABILITY_COLUMN].mean()) if len(unpurchased) else 0.0
    return {
        "優先区分": priority,
        "企業群": name,
        "定義": rule,
        "特徴の明確さ": clarity,
        "企業数": int(len(selected)),
        "購入企業数": int(selected[TARGET].sum()),
        "非購入企業数": int((selected[TARGET] == 0).sum()),
        "購入率": float(selected[TARGET].mean()),
        "平均OOF確率": float(selected[PROBABILITY_COLUMN].mean()),
        "未購入企業平均OOF確率": unpurchased_mean,
        "閾値以上の未購入企業数": int((unpurchased[PROBABILITY_COLUMN] >= threshold).sum()),
        "機会指数": float(len(unpurchased) * unpurchased_mean),
    }


def priority_segments(train: pd.DataFrame, threshold: float) -> pd.DataFrame:
    survey = {column: pd.to_numeric(train[column], errors="coerce") for column in SURVEY_COLUMNS}
    employee = pd.to_numeric(train["従業員数"], errors="coerce")
    operating_profit = pd.to_numeric(train["営業利益"], errors="coerce")
    outlook = train["今後のDX展望"].fillna("").astype(str)
    organization = train["組織図"].fillna("").astype(str)
    digital_industry = train["業界"].isin(["IT", "人材", "自動車・乗り物"])
    pain = (survey["アンケート２"] == 1) | (survey["アンケート１０"] == 1)
    tool_dissatisfaction = survey["アンケート７"] <= 2

    specs: list[tuple[str, str, pd.Series, str, str]] = [
        (
            "DX課題顕在×BtoB",
            "アンケート2=1 または アンケート10=1、かつ特徴=BtoB",
            pain & (train["特徴"] == "BtoB"),
            "A：最優先",
            "高",
        ),
        (
            "デジタル集約業界",
            "業界がIT・人材・自動車/乗り物",
            digital_industry,
            "A：最優先",
            "高",
        ),
        (
            "既存ツール低満足×BtoB",
            "アンケート7が1〜2、かつ特徴=BtoB",
            tool_dissatisfaction & (train["特徴"] == "BtoB"),
            "B：次に優先",
            "高",
        ),
        (
            "DX関心高・導入初期",
            "アンケート1>=4、アンケート3<=2",
            (survey["アンケート１"] >= 4) & (survey["アンケート３"] <= 2),
            "C：潜在顧客",
            "高",
        ),
        (
            "大規模・高収益",
            "従業員数が中央値以上、営業利益が上位25%",
            (employee >= employee.median()) & (operating_profit >= operating_profit.quantile(0.75)),
            "B：次に優先",
            "中",
        ),
        (
            "DX推進組織あり",
            "組織図にDX推進を含む",
            organization.str.contains("DX推進", regex=False),
            "B：次に優先",
            "高",
        ),
        (
            "リスキリング・人材育成関心",
            "今後のDX展望にリスキリングまたは人材育成を含む",
            outlook.str.contains("リスキリング|人材育成", regex=True),
            "C：潜在顧客",
            "高",
        ),
        (
            "慎重・ROI重視",
            "今後のDX展望に慎重・段階的・費用対効果・ROIのいずれかを含む",
            outlook.str.contains("慎重|段階的|費用対効果|ROI", regex=True),
            "C：育成対象",
            "中",
        ),
    ]
    rows = [
        segment_stats(train, name, rule, mask, priority, clarity, threshold)
        for name, rule, mask, priority, clarity in specs
    ]
    return pd.DataFrame(rows)


def mask_evidence(
    train: pd.DataFrame,
    feature: str,
    segment: str,
    mask: pd.Series,
    comparison: str,
    comparison_mask: pd.Series,
) -> dict[str, object]:
    selected = train.loc[mask, TARGET]
    compared = train.loc[comparison_mask, TARGET]
    return {
        "特徴": feature,
        "注目区分": segment,
        "企業数": int(len(selected)),
        "購入企業数": int(selected.sum()),
        "購入率": float(selected.mean()),
        "比較区分": comparison,
        "比較企業数": int(len(compared)),
        "比較購入企業数": int(compared.sum()),
        "比較購入率": float(compared.mean()),
        "購入率差": float(selected.mean() - compared.mean()),
    }


def key_evidence(train: pd.DataFrame) -> pd.DataFrame:
    survey2 = pd.to_numeric(train["アンケート２"], errors="coerce")
    survey7 = pd.to_numeric(train["アンケート７"], errors="coerce")
    survey10 = pd.to_numeric(train["アンケート１０"], errors="coerce")
    profit = pd.to_numeric(train["営業利益"], errors="coerce")
    organization = train["組織図"].fillna("").astype(str)
    outlook = train["今後のDX展望"].fillna("").astype(str)
    rows = [
        mask_evidence(
            train,
            "デジタル集約業界",
            "IT・人材・自動車/乗り物",
            train["業界"].isin(["IT", "人材", "自動車・乗り物"]),
            "その他業界",
            ~train["業界"].isin(["IT", "人材", "自動車・乗り物"]),
        ),
        mask_evidence(
            train,
            "営業利益",
            "上位25%",
            profit >= profit.quantile(0.75),
            "下位25%",
            profit <= profit.quantile(0.25),
        ),
        mask_evidence(train, "特徴", "BtoB", train["特徴"] == "BtoB", "BtoC", train["特徴"] == "BtoC"),
        mask_evidence(train, "上場種別", "PR", train["上場種別"] == "PR", "ST", train["上場種別"] == "ST"),
        mask_evidence(train, "アンケート2", "回答1", survey2 == 1, "回答5", survey2 == 5),
        mask_evidence(train, "アンケート7", "回答1", survey7 == 1, "回答5", survey7 == 5),
        mask_evidence(train, "アンケート10", "回答1", survey10 == 1, "回答5", survey10 == 5),
        mask_evidence(
            train,
            "組織図キーワード",
            "DX推進あり",
            organization.str.contains("DX推進", regex=False),
            "DX推進なし",
            ~organization.str.contains("DX推進", regex=False),
        ),
        mask_evidence(
            train,
            "DX展望キーワード",
            "リスキリングあり",
            outlook.str.contains("リスキリング", regex=False),
            "リスキリングなし",
            ~outlook.str.contains("リスキリング", regex=False),
        ),
        mask_evidence(
            train,
            "DX展望キーワード",
            "慎重あり",
            outlook.str.contains("慎重", regex=False),
            "慎重なし",
            ~outlook.str.contains("慎重", regex=False),
        ),
    ]
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#98A2B3")
    ax.spines["bottom"].set_color("#98A2B3")


def save_industry_chart(categorical: pd.DataFrame, overall: float, path: Path) -> None:
    data = categorical[
        (categorical["変数"] == "業界") & (categorical["企業数"] >= 20)
    ].sort_values("購入率")
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=180)
    bars = ax.barh(data["区分"], data["購入率"], color="#3572B8", edgecolor="#244D7C")
    ax.axvline(overall, color="#6B7280", linestyle="--", linewidth=1.4, label=f"全体 {overall:.1%}")
    ax.set_title("主要業界別のDX教材購入率（20社以上）", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("購入率")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(0.55, float(data["購入率"].max()) + 0.10))
    style_axis(ax)
    ax.legend(loc="lower right", frameon=False)
    for bar, (_, row) in zip(bars, data.iterrows(), strict=True):
        ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{row['購入率']:.1%} (n={int(row['企業数'])})",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def short_bin_label(label: str) -> str:
    if label == "欠損":
        return label
    return label.split(":", 1)[0]


def save_numeric_chart(numeric: pd.DataFrame, overall: float, path: Path) -> None:
    variables = ["従業員数", "営業利益", "総資産", "事業所数"]
    selected = numeric[numeric["変数"].isin(variables)]
    y_max = min(0.60, max(0.40, float(selected["購入率"].max()) + 0.08))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    for ax, variable in zip(axes.flat, variables, strict=True):
        data = numeric[numeric["変数"] == variable].sort_values("区分順")
        labels = [short_bin_label(label) for label in data["区分"]]
        x = np.arange(len(data))
        bars = ax.bar(x, data["購入率"], color="#3572B8", edgecolor="#244D7C", width=0.65)
        ax.axhline(overall, color="#6B7280", linestyle="--", linewidth=1.2)
        ax.set_title(f"{variable}：四分位別購入率", fontsize=13, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#D7DCE2", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, (_, row) in zip(bars, data.iterrows(), strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{row['購入率']:.1%}\nn={int(row['企業数'])}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    fig.suptitle("企業規模・財務指標と購入率", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_survey_chart(survey: pd.DataFrame, overall: float, path: Path) -> None:
    variables = ["アンケート２", "アンケート７", "アンケート１０"]
    selected = survey[survey["変数"].isin(variables)]
    y_max = max(0.55, float(selected["購入率"].max()) + 0.08)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=180)
    titles = {
        "アンケート２": "DX化への満足度",
        "アンケート７": "既存ツールへの満足度",
        "アンケート１０": "外部DX連携の充実度",
    }
    for ax, variable in zip(axes, variables, strict=True):
        data = survey[survey["変数"] == variable].sort_values("区分順")
        x = np.arange(len(data))
        bars = ax.bar(x, data["購入率"], color="#D8902F", edgecolor="#8A5C20", width=0.65)
        ax.axhline(overall, color="#6B7280", linestyle="--", linewidth=1.2)
        ax.set_title(f"{variable}\n{titles[variable]}", fontsize=12, fontweight="bold")
        ax.set_xticks(x, data["区分"])
        ax.set_xlabel("回答値")
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#D7DCE2", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, (_, row) in zip(bars, data.iterrows(), strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{row['購入率']:.1%}\nn={int(row['企業数'])}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    fig.suptitle("営業課題に直結するアンケート回答別購入率", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_keyword_chart(keyword: pd.DataFrame, path: Path) -> None:
    reliable = keyword[keyword["安定比較対象"]].copy()
    positive = reliable.nlargest(5, "購入率差")
    negative = reliable.nsmallest(5, "購入率差")
    data = pd.concat([negative, positive]).drop_duplicates(["テキスト列", "キーワード"])
    data = data.sort_values("購入率差")
    labels = data["テキスト列"] + "：" + data["キーワード"]
    colors = np.where(data["購入率差"] >= 0, "#3572B8", "#D8902F")
    fig, ax = plt.subplots(figsize=(11.5, 7.4), dpi=180)
    bars = ax.barh(labels, data["購入率差"], color=colors, edgecolor="#475467", linewidth=0.6)
    ax.axvline(0, color="#344054", linewidth=1.2)
    ax.set_title("キーワード有無による購入率差（各群20社以上）", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("キーワードあり − なし（購入率差）")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, (_, row) in zip(bars, data.iterrows(), strict=True):
        value = float(row["購入率差"])
        ax.text(
            value + (0.008 if value >= 0 else -0.008),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1%} (n={int(row['該当企業数'])})",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )
    limit = max(abs(float(data["購入率差"].min())), abs(float(data["購入率差"].max()))) + 0.08
    ax.set_xlim(-limit, limit)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_priority_chart(segments: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=180)
    top_names = {"DX課題顕在×BtoB", "デジタル集約業界", "既存ツール低満足×BtoB"}
    for _, row in segments.iterrows():
        highlight = row["企業群"] in top_names
        color = "#D8902F" if highlight else "#8FAFD1"
        size = 90 + float(row["機会指数"]) * 16
        ax.scatter(
            row["非購入企業数"],
            row["未購入企業平均OOF確率"],
            s=size,
            color=color,
            edgecolor="#344054",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.annotate(
            row["企業群"],
            (row["非購入企業数"], row["未購入企業平均OOF確率"]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_title("営業候補セグメント：購入可能性 × 市場規模", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("未購入企業数（市場規模）")
    ax.set_ylabel("未購入企業の平均OOF購入確率")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=1))
    ax.grid(color="#D7DCE2", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pct(value: float) -> str:
    return f"{value:.1%}"


def pp(value: float) -> str:
    return f"{value * 100:+.1f}pt"


def row_for(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column] == value
    matches = frame[mask]
    if len(matches) != 1:
        raise KeyError(f"Expected one row for {conditions}, got {len(matches)}")
    return matches.iloc[0]


def write_report(
    path: Path,
    train: pd.DataFrame,
    categorical: pd.DataFrame,
    numeric: pd.DataFrame,
    survey: pd.DataFrame,
    keyword: pd.DataFrame,
    segments: pd.DataFrame,
    evidence: pd.DataFrame,
    model_metrics: dict[str, object],
    font_name: str,
) -> None:
    overall = float(train[TARGET].mean())
    industry_reliable = categorical[
        (categorical["変数"] == "業界") & (categorical["企業数"] >= 20)
    ].sort_values("購入率", ascending=False)
    top_industries = industry_reliable.head(7)
    selected_numeric = []
    for variable in ["従業員数", "売上", "営業利益", "総資産", "資本金", "事業所数", "店舗数", "工場数"]:
        rows = numeric[(numeric["変数"] == variable) & (numeric["区分"] != "欠損")].sort_values("区分順")
        selected_numeric.append(
            {
                "変数": variable,
                "下位区分": rows.iloc[0]["区分"],
                "下位企業数": int(rows.iloc[0]["企業数"]),
                "下位購入率": float(rows.iloc[0]["購入率"]),
                "上位区分": rows.iloc[-1]["区分"],
                "上位企業数": int(rows.iloc[-1]["企業数"]),
                "上位購入率": float(rows.iloc[-1]["購入率"]),
            }
        )
    numeric_comparison = pd.DataFrame(selected_numeric)

    key_survey_rows = []
    for variable in ["アンケート２", "アンケート７", "アンケート１０"]:
        for answer in ["1", "5"]:
            row = row_for(survey, 変数=variable, 区分=answer)
            key_survey_rows.append(row)
    key_survey = pd.DataFrame(key_survey_rows)
    survey_topics = {
        "アンケート１": ("DX戦略の明確さ", "4〜5でやや高いが単調ではない"),
        "アンケート２": ("社内DX化への満足度", "回答1が突出して高い"),
        "アンケート３": ("最新デジタル技術の導入状況", "回答5が高いが途中は変動"),
        "アンケート４": ("DX変革への抵抗感", "抵抗感4〜5で低い"),
        "アンケート５": ("サイバーセキュリティ整備", "高回答側でやや高い"),
        "アンケート６": ("改善・自動化ツール導入", "回答差は小さい"),
        "アンケート７": ("既存ツールへの満足度", "回答上昇に伴い明確に低下"),
        "アンケート８": ("DX成果の実感", "回答5だけ低い非単調型"),
        "アンケート９": ("技術イベント参加率", "回答差は小さい"),
        "アンケート１０": ("外部DX連携の充実度", "概ね回答上昇に伴い低下"),
        "アンケート１１": ("DX情報収集状況", "回答差は小さい"),
    }

    reliable_keywords = keyword[keyword["安定比較対象"]]
    positive_keywords = reliable_keywords.nlargest(6, "購入率差")
    negative_keywords = reliable_keywords.nsmallest(6, "購入率差")

    priority_names = ["DX課題顕在×BtoB", "デジタル集約業界", "既存ツール低満足×BtoB"]
    priority = segments.set_index("企業群").loc[priority_names].reset_index()

    lines: list[str] = [
        "# 1. 分析概要",
        "",
        f"本分析は、SIGNATEコンペで提供された `data/train.csv` の **{len(train):,}社**を企業単位で集計したものです。購入企業は **{int(train[TARGET].sum())}社**、全体購入率は **{pct(overall)}** です。企業は架空企業であり、ここで得た示唆はコンペデータ内の関連として扱います。",
        "",
        "営業優先度には、観測購入率・企業数・特徴の説明可能性に加え、既存のStratifiedGroup 5-fold OOF予測を使用しました。現行モデルはCatBoost 30%＋テキスト70%で、保存指標は "
        f"OOF F1 **{float(model_metrics['oof_f1_tuned']):.3f}**、ROC-AUC **{float(model_metrics['oof_roc_auc']):.3f}**、閾値 **{float(model_metrics['oof_threshold']):.3f}** です。OOF確率は企業の優先順位付けにのみ使い、施策による成約確率とは解釈しません。",
        "",
        "出力した詳細表：",
        "",
        "- [`categorical_purchase_summary.csv`](evidence/categorical_purchase_summary.csv)",
        "- [`numeric_purchase_summary.csv`](evidence/numeric_purchase_summary.csv)",
        "- [`survey_purchase_summary.csv`](evidence/survey_purchase_summary.csv)",
        "- [`text_keyword_purchase_summary.csv`](evidence/text_keyword_purchase_summary.csv)",
        "- [`text_length_purchase_summary.csv`](evidence/text_length_purchase_summary.csv)",
        "- [`priority_segments.csv`](evidence/priority_segments.csv)",
        "- [`key_evidence.csv`](evidence/key_evidence.csv)",
        "",
        "# 2. DX教材を購入しやすい企業",
        "",
        "## データから分かった事実：カテゴリ属性",
        "",
        "20社以上ある業界では、次の購入率が高くなりました。件数の小さい通信（5社中4社購入）などは上位評価から除外しています。",
        "",
        "| 業界 | 企業数 | 購入企業数 | 購入率 | 全体との差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in top_industries.iterrows():
        lines.append(
            f"| {row['区分']} | {int(row['企業数'])} | {int(row['購入企業数'])} | {pct(row['購入率'])} | {pp(row['全体比差'])} |"
        )
    lines.extend(
        [
            "",
            "![主要業界別購入率](evidence/industry_purchase_rate.png)",
            "",
        ]
    )

    for variable, high, low in [
        ("上場種別", "PR", "ST"),
        ("特徴", "BtoB", "BtoC"),
    ]:
        high_row = row_for(categorical, 変数=variable, 区分=high)
        low_row = row_for(categorical, 変数=variable, 区分=low)
        lines.append(
            f"- **{variable}**：{high}は{int(high_row['企業数'])}社中{int(high_row['購入企業数'])}社、購入率{pct(high_row['購入率'])}。"
            f"{low}は{int(low_row['企業数'])}社中{int(low_row['購入企業数'])}社、{pct(low_row['購入率'])}。"
        )

    lines.extend(
        [
            "",
            "## データから分かった事実：企業規模・財務・拠点",
            "",
            "数値項目は欠損を残したまま、回答済み企業を四分位に分けました。金額の単位は元データどおり百万円です。",
            "",
            "| 変数 | 下位区分 | 企業数 | 購入率 | 上位区分 | 企業数 | 購入率 |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for _, row in numeric_comparison.iterrows():
        lines.append(
            f"| {row['変数']} | {row['下位区分']} | {row['下位企業数']} | {pct(row['下位購入率'])} | "
            f"{row['上位区分']} | {row['上位企業数']} | {pct(row['上位購入率'])} |"
        )
    lines.extend(
        [
            "",
            "営業利益は下位25%の購入率が10.9%に対し上位25%は33.7%、従業員数は下位25%の14.0%に対し上位25%は29.6%でした。店舗数・工場数は欠損が多いため、回答済み企業だけの差を全体へ一般化しません。",
            "",
            "![企業規模・財務指標](evidence/numeric_purchase_rate.png)",
            "",
            "## データから分かった事実：アンケート",
            "",
            "| アンケート | 回答 | 企業数 | 購入企業数 | 購入率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in key_survey.iterrows():
        lines.append(
            f"| {row['変数']} | {row['区分']} | {int(row['企業数'])} | {int(row['購入企業数'])} | {pct(row['購入率'])} |"
        )
    lines.extend(
        [
            "",
            "- アンケート2（DX化への満足度）は回答1が34.4%、回答5が16.9%。",
            "- アンケート7（既存ツールへの満足度）は回答1が46.4%、回答5が12.7%。回答7の欠損356社はアンケート6の条件に伴う構造的欠損です。",
            "- アンケート10（外部DX連携の充実度）は回答1が37.0%、回答5が15.6%。",
            "- アンケート6と11は回答間の購入率差がそれぞれ3.7pt、3.6ptで小さく、単独ターゲティング根拠としては弱い結果でした。",
            "",
            "### アンケート1〜11の全問概況",
            "",
            "端点比較だけで単調性を断定せず、全回答値は `survey_purchase_summary.csv` で確認できるようにしています。",
            "",
            "| 問 | 質問テーマ | 低い側：企業数/購入数/率 | 高い側：企業数/購入数/率 | 観測パターン |",
            "|---|---|---:|---:|---|",
        ]
    )
    for variable in SURVEY_COLUMNS:
        rows = survey[(survey["変数"] == variable) & (survey["区分"] != "欠損")].sort_values("区分順")
        low = rows.iloc[0]
        high = rows.iloc[-1]
        topic, pattern = survey_topics[variable]
        lines.append(
            f"| {variable.replace('アンケート', '')} | {topic} | 回答{low['区分']}：{int(low['企業数'])}/{int(low['購入企業数'])}/{pct(low['購入率'])} | "
            f"回答{high['区分']}：{int(high['企業数'])}/{int(high['購入企業数'])}/{pct(high['購入率'])} | {pattern} |"
        )
    lines.extend(
        [
            "",
            "![主要アンケート](evidence/survey_purchase_rate_key.png)",
            "",
            "## データから分かった事実：企業概要・組織図・今後のDX展望",
            "",
            "各テキスト列でキーワード有無を比較し、両群20社以上のものを安定比較対象としました。",
            "",
            "| テキスト列 | キーワード | 該当企業数 | 該当購入率 | 非該当購入率 | 差 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    combined_keywords = pd.concat([positive_keywords.head(5), negative_keywords.head(5)]).drop_duplicates(
        ["テキスト列", "キーワード"]
    )
    for _, row in combined_keywords.sort_values("購入率差", ascending=False).iterrows():
        lines.append(
            f"| {row['テキスト列']} | {row['キーワード']} | {int(row['該当企業数'])} | "
            f"{pct(row['該当購入率'])} | {pct(row['非該当購入率'])} | {pp(row['購入率差'])} |"
        )
    lines.extend(
        [
            "",
            "特に、企業概要に `IT` を含む79社は43.0%、組織図に `DX推進` を含む348社は30.7%、今後のDX展望に `リスキリング` を含む141社は31.9%でした。一方、DX展望に `慎重` を含む469社は12.6%でした。キーワードは定型文や業界の代理変数になり得るため、言葉そのものの因果効果とは解釈しません。",
            "",
            "![キーワード購入率差](evidence/keyword_purchase_lift.png)",
            "",
            "## 購入しやすい企業の特徴（3〜5点）",
            "",
            "1. **デジタル集約業界**：IT・人材・自動車/乗り物は131社中61社購入、購入率46.6%。",
            "2. **規模と収益力が高い**：従業員数、営業利益、総資産、事業所数の上位層で購入率が高い。特に営業利益上位25%は33.7%。",
            "3. **DXの課題が顕在化している**：DX満足度、既存ツール満足度、外部DX連携が最低評価の企業で購入率が高い。",
            "4. **BtoBかつPR区分**：BtoBは27.4%に対しBtoCは6.5%、PRは28.4%に対しSTは11.5%。",
            "5. **推進組織・人材育成テーマが明示されている**：組織図のDX推進、展望のリスキリング・人材育成・業務改善が正の関連を示した。",
            "",
            "# 3. 購入率が高い理由についての仮説",
            "",
            "以下では、観測事実と営業仮説を明確に分けます。",
            "",
            "| データから確認できた事実 | そこから考えられる仮説 | 他の可能性・注意点 |",
            "|---|---|---|",
            "| DX満足度1は34.4%、外部連携1は37.0% | 現状への不満が強く、外部の教育・伴走支援を求める需要が顕在化している | 購入後も満足度が低い可能性があり、前後関係は不明 |",
            "| 既存ツール満足度1は46.4%、回答5は12.7% | ツール導入だけでは成果が出ず、活用スキルや業務設計の研修が必要 | 不満足企業に特定業界が多い可能性がある |",
            "| デジタル集約業界は46.6% | 技術変化が速く、継続的なリスキリングが事業課題になりやすい | 業界ラベルがテキストや企業規模を代理している可能性 |",
            "| 営業利益上位25%は33.7%、下位25%は10.9% | 教育予算を確保しやすく、全社研修へ拡張しやすい | 利益規模と従業員数の相関による見かけの差かもしれない |",
            "| 組織図のDX推進ありは30.7%、なしは18.3% | 専任組織が提案窓口・予算責任者となり、導入を進めやすい | 組織図の記述量や定型テンプレートの影響があり得る |",
            "| DX展望の慎重ありは12.6%、なしは44.0% | 投資判断にROI・社内合意・小さな成功事例が必要 | `慎重` が特定の合成文章テンプレートを識別している可能性 |",
            "",
            "# 4. 優先的に営業すべき企業群",
            "",
            "優先度は、購入率、母数、未購入企業のOOF確率、ルールの説明しやすさを合わせて判断しました。セグメントは重複するため、企業数を合算しません。",
            "",
            "| 優先 | 企業群 | 企業数 | 購入率 | 未購入企業数 | 未購入平均OOF | 閾値以上未購入 | 機会指数 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in priority.iterrows():
        lines.append(
            f"| {row['優先区分']} | {row['企業群']} | {int(row['企業数'])} | {pct(row['購入率'])} | "
            f"{int(row['非購入企業数'])} | {pct(row['未購入企業平均OOF確率'])} | "
            f"{int(row['閾値以上の未購入企業数'])} | {row['機会指数']:.1f} |"
        )
    lines.extend(
        [
            "",
            "**最初に営業するなら `DX課題顕在×BtoB` です。** 145社という母数、37.9%の購入率、90社の未購入母数、16社のモデル閾値以上候補があり、課題も「DX満足度または外部連携の不足」と説明できます。次にデジタル集約業界、既存ツール低満足×BtoBへ展開します。",
            "",
            "`機会指数 = 未購入企業数 × 未購入企業の平均OOF確率` は相対比較用であり、期待成約件数の保証値ではありません。",
            "",
            "![営業機会マップ](evidence/priority_opportunity_map.png)",
            "",
            "# 5. セグメント別の提案内容",
            "",
            "| 企業群 | 課題仮説 | 提案するDX教育 | 提案時の訴求ポイント |",
            "|---|---|---|---|",
            "| DX課題顕在×BtoB | DX満足度が低い、または外部パートナー連携が不足 | 現状診断＋部門別DX基礎＋伴走ワークショップ | 研修単体ではなく、課題棚卸しから90日実行計画まで提示 |",
            "| デジタル集約業界 | 技術更新が速く、職種別スキル更新が継続的に必要 | AI・データ分析・クラウド・実践型リスキリング | 業界ユースケース、職種別ラーニングパス、成果物評価を訴求 |",
            "| 既存ツール低満足×BtoB | 導入済みツールを業務成果へ結び付けるスキルが不足 | ツール活用高度化、業務改善、データ活用、現場演習 | 現行ツールを題材にし、利用率・工数削減など測定指標を設定 |",
            "| DX関心高・導入初期 | 方針はあるが技術導入が遅れ、着手方法が不明 | DX入門、デジタルリテラシー、小規模PoC | 低リスク・短期間・少人数から開始し、次段階の判断基準を提示 |",
            "| 大規模・高収益 | 対象者が多く、階層・職種ごとの要件が異なる | 全社DXアカデミー、管理職・専門職別カリキュラム | ガバナンス、受講データ、社内認定、拠点展開の運用設計を訴求 |",
            "| DX推進組織あり | 推進組織はあるが、現場展開や人材パイプラインが課題 | DX推進リーダー育成、社内講師育成、実案件型研修 | 推進部門を窓口に、現場で再現できる育成制度として提案 |",
            "| 慎重・ROI重視 | 投資効果と社内合意が導入障壁 | 少人数トライアル、無料診断、短期研修 | 事前KPI、費用対効果、継続・停止条件を明示 |",
            "",
            "# 6. 仮説検証方法",
            "",
            "## 営業施策のA/Bテスト",
            "",
            "- 同一セグメント・OOF確率帯の企業を層別化してランダム割付する。",
            "- A群は通常資料、B群は本分析に基づくセグメント別提案を使用する。",
            "- 主要指標は商談化率または成約率、補助指標は資料請求率、返信率、商談単価、獲得費用、受注までの日数とする。",
            "- テスト前に最低検出効果と必要標本数を決め、途中経過だけで勝敗を決めない。",
            "",
            "## セグメント別メッセージテスト",
            "",
            "- 既存ツール低満足群では「ツール活用高度化」と「一般DX研修」を比較する。",
            "- DX関心高・導入初期群では「小規模PoC」と「全社研修構想」を比較する。",
            "- 慎重・ROI重視群では「ROI事例＋短期トライアル」と通常価格訴求を比較する。",
            "",
            "## 追加アンケート",
            "",
            "DX教育予算、人材不足職種、既存研修、意思決定者、導入予定時期、利用中ツール、期待成果を追加取得し、今回の仮説を直接測定します。",
            "",
            "## 時系列検証",
            "",
            "未購入企業を3〜6か月追跡し、商談・提案・購入の時系列を保存します。OOF確率帯と実購入率の校正、セグメント別リフト、予測ドリフトを確認します。",
            "",
            "# 7. 今後のアクション",
            "",
            "1. `DX課題顕在×BtoB`、`デジタル集約業界`、`既存ツール低満足×BtoB` の未購入企業から、モデル閾値以上の候補を初回パイロット対象にする。",
            "2. 3セグメントで提案資料を分け、通常資料とのランダム比較を4〜8週間実施する。",
            "3. CRMにセグメント、OOF確率帯、接触日、提案内容、返信、商談、成約、売上を記録する。",
            "4. 企業数が少ないカテゴリやテキストキーワードだけで対象を除外せず、営業担当者の確認を入れる。",
            "5. 実営業データが蓄積したら、購入フラグではなく商談化率・成約率・売上を目的変数として再評価する。",
            "",
            "## 重要な制約",
            "",
            "- 本分析は観察データの相関分析であり、因果関係を証明しません。",
            "- 多数の属性・語を比較しているため、偶然大きく見える差が含まれる可能性があります。",
            "- モデルOOF確率は現行コンペモデルの順位付け情報であり、営業施策による増分効果ではありません。",
            "- コンペ上の架空企業データから得た結果を、実市場へ直接一般化しないでください。",
            f"- グラフは `{font_name}` を使用し、購入率の分母と企業数を併記しています。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIGNATEデータからDX教材の営業示唆を生成します。")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    train_path = project_dir / "data" / "train.csv"
    baseline_dir = project_dir / "output" / "experiments" / "baseline"
    oof_path = baseline_dir / "cv_oof_predictions.csv"
    model_metrics_path = baseline_dir / "model_metrics.json"
    sales_dir = project_dir / "output" / "analysis" / "sales"
    analysis_dir = sales_dir / "evidence"
    report_path = sales_dir / "DX_SALES_INSIGHTS.md"

    train = pd.read_csv(train_path)
    oof = pd.read_csv(oof_path)
    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    required = {
        ID_COLUMN,
        TARGET,
        *CATEGORICAL_COLUMNS,
        *NUMERIC_COLUMNS,
        *SURVEY_COLUMNS,
        *TEXT_COLUMNS,
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"train.csvに必要な列がありません: {missing}")
    if len(oof) != len(train):
        raise ValueError("OOF予測とtrain.csvの行数が一致しません。")
    if oof[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
        raise ValueError("OOF予測とtrain.csvの企業ID順が一致しません。")
    if oof[TARGET].astype(int).tolist() != train[TARGET].astype(int).tolist():
        raise ValueError("OOF予測とtrain.csvの購入フラグが一致しません。")
    if not oof[PROBABILITY_COLUMN].between(0, 1).all():
        raise ValueError("OOF確率が0〜1の範囲外です。")

    train = train.copy()
    train[PROBABILITY_COLUMN] = oof[PROBABILITY_COLUMN].to_numpy(dtype=float)
    threshold = float(model_metrics["oof_threshold"])
    font_name, _ = configure_japanese_font()

    categorical = categorical_summary(train)
    numeric = numeric_summary(train)
    survey = survey_summary(train)
    keyword = keyword_summary(train)
    text_length = text_length_summary(train)
    segments = priority_segments(train, threshold)
    evidence = key_evidence(train)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv_if_changed(categorical, analysis_dir / "categorical_purchase_summary.csv")
    write_csv_if_changed(numeric, analysis_dir / "numeric_purchase_summary.csv")
    write_csv_if_changed(survey, analysis_dir / "survey_purchase_summary.csv")
    write_csv_if_changed(keyword, analysis_dir / "text_keyword_purchase_summary.csv")
    write_csv_if_changed(text_length, analysis_dir / "text_length_purchase_summary.csv")
    write_csv_if_changed(segments, analysis_dir / "priority_segments.csv")
    write_csv_if_changed(evidence, analysis_dir / "key_evidence.csv")

    overall = float(train[TARGET].mean())
    save_industry_chart(categorical, overall, analysis_dir / "industry_purchase_rate.png")
    save_numeric_chart(numeric, overall, analysis_dir / "numeric_purchase_rate.png")
    save_survey_chart(survey, overall, analysis_dir / "survey_purchase_rate_key.png")
    save_keyword_chart(keyword, analysis_dir / "keyword_purchase_lift.png")
    save_priority_chart(segments, analysis_dir / "priority_opportunity_map.png")
    write_report(
        report_path,
        train,
        categorical,
        numeric,
        survey,
        keyword,
        segments,
        evidence,
        model_metrics,
        font_name,
    )

    print(f"Report: {report_path}")
    print(f"Artifacts: {analysis_dir}")
    print(f"CSV files: {len(list(analysis_dir.glob('*.csv')))}")
    print(f"PNG files: {len(list(analysis_dir.glob('*.png')))}")
    print(f"Overall purchase rate: {overall:.4f}")


if __name__ == "__main__":
    main()
