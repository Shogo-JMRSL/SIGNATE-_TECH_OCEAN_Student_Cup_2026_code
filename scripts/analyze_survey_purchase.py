from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, ticker
from scipy.stats import chi2_contingency, spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "購入フラグ"
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
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\BIZ-UDGothicR.ttc"),
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path(r"C:\Windows\Fonts\msgothic.ttc"),
]


def configure_japanese_font() -> tuple[str, Path]:
    font_path = next((path for path in FONT_CANDIDATES if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("利用可能な日本語フォントが見つかりません。")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#5B6472",
            "axes.labelcolor": "#202733",
            "text.color": "#202733",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    return font_name, font_path


def format_answer(value: object) -> str:
    if pd.isna(value):
        return "欠損"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def summarize_survey(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for survey in SURVEY_COLUMNS:
        values = train[survey]
        numeric_answers = sorted(pd.to_numeric(values.dropna(), errors="raise").unique())
        for answer in numeric_answers:
            mask = values == answer
            total = int(mask.sum())
            purchased = int(train.loc[mask, TARGET].sum())
            rows.append(
                {
                    "アンケート": survey,
                    "回答値": format_answer(answer),
                    "全企業数": total,
                    "購入企業数": purchased,
                    "非購入企業数": total - purchased,
                    "購入率": purchased / total,
                }
            )
        if values.isna().any():
            mask = values.isna()
            total = int(mask.sum())
            purchased = int(train.loc[mask, TARGET].sum())
            rows.append(
                {
                    "アンケート": survey,
                    "回答値": "欠損",
                    "全企業数": total,
                    "購入企業数": purchased,
                    "非購入企業数": total - purchased,
                    "購入率": purchased / total,
                }
            )
    summary = pd.DataFrame(rows)
    for survey in SURVEY_COLUMNS:
        survey_rows = summary[summary["アンケート"] == survey]
        if int(survey_rows["全企業数"].sum()) != len(train):
            raise RuntimeError(f"{survey} の集計企業数が学習データ件数と一致しません。")
        if not np.array_equal(
            survey_rows["全企業数"].to_numpy(),
            (
                survey_rows["購入企業数"].to_numpy()
                + survey_rows["非購入企業数"].to_numpy()
            ),
        ):
            raise RuntimeError(f"{survey} の購入・非購入内訳が一致しません。")
    return summary


def add_count_labels(ax: plt.Axes, bars: object, values: np.ndarray) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{int(value)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#202733",
        )


def add_rate_labels(ax: plt.Axes, bars: object, values: np.ndarray) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#202733",
        )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA3AE")
    ax.spines["bottom"].set_color("#9AA3AE")


def create_charts(summary: pd.DataFrame, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    count_color = "#3572B8"
    rate_color = "#D8902F"

    for number, survey in enumerate(SURVEY_COLUMNS, start=1):
        rows = summary[summary["アンケート"] == survey]
        labels = rows["回答値"].astype(str).tolist()
        counts = rows["購入企業数"].to_numpy(dtype=int)
        rates = rows["購入率"].to_numpy(dtype=float)
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)
        bars = ax.bar(x, counts, width=0.62, color=count_color, edgecolor="#244D7C", linewidth=0.8)
        ax.set_title(f"{survey}：回答別の購入企業数", fontsize=16, fontweight="bold", pad=16)
        ax.set_xlabel("回答値", fontsize=12, labelpad=9)
        ax.set_ylabel("購入企業数（社）", fontsize=12, labelpad=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(1.0, float(counts.max()) * 1.22))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        style_axis(ax)
        add_count_labels(ax, bars, counts)
        fig.tight_layout()
        fig.savefig(chart_dir / f"survey{number:02d}_purchase_count.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)
        bars = ax.bar(x, rates, width=0.62, color=rate_color, edgecolor="#8A5C20", linewidth=0.8)
        ax.set_title(f"{survey}：回答別の購入率", fontsize=16, fontweight="bold", pad=16)
        ax.set_xlabel("回答値", fontsize=12, labelpad=9)
        ax.set_ylabel("購入率", fontsize=12, labelpad=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        style_axis(ax)
        add_rate_labels(ax, bars, rates)
        fig.tight_layout()
        fig.savefig(chart_dir / f"survey{number:02d}_purchase_rate.png", bbox_inches="tight")
        plt.close(fig)


def corrected_cramers_v(table: np.ndarray) -> tuple[float, float]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    n = float(table.sum())
    rows, columns = table.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    value = math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0
    return float(value), float(p_value)


def association_metrics(train: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for survey in SURVEY_COLUMNS:
        rows = summary[summary["アンケート"] == survey].copy()
        contingency_all = rows[["非購入企業数", "購入企業数"]].to_numpy(dtype=int)
        v_all, p_all = corrected_cramers_v(contingency_all)
        answered_rows = rows[rows["回答値"] != "欠損"]
        contingency_answered = answered_rows[["非購入企業数", "購入企業数"]].to_numpy(dtype=int)
        v_answered, p_answered = corrected_cramers_v(contingency_answered)

        answered_mask = train[survey].notna()
        rho, rho_p = spearmanr(
            pd.to_numeric(train.loc[answered_mask, survey]),
            train.loc[answered_mask, TARGET],
        )
        if np.isnan(rho):
            rho, rho_p = 0.0, 1.0

        rates = answered_rows["購入率"].to_numpy(dtype=float)
        rate_range = float(rates.max() - rates.min())
        differences = np.diff(rates)
        mostly_up = int((differences >= 0).sum()) >= max(1, len(differences) - 1)
        mostly_down = int((differences <= 0).sum()) >= max(1, len(differences) - 1)
        if rate_range <= 0.05:
            pattern = "回答による差が小さい"
        elif mostly_up and rates[-1] - rates[0] >= 0.08:
            pattern = "回答値とともに概ね上昇"
        elif mostly_down and rates[0] - rates[-1] >= 0.08:
            pattern = "回答値とともに概ね下降"
        elif rho >= 0.15 and rho_p < 0.05:
            pattern = "上昇傾向（非単調）"
        elif rho <= -0.15 and rho_p < 0.05:
            pattern = "下降傾向（非単調）"
        else:
            sorted_rates = np.sort(rates)
            top_gap = float(sorted_rates[-1] - sorted_rates[-2]) if len(sorted_rates) > 1 else 0.0
            pattern = "特定回答が高い" if top_gap >= 0.05 else "非単調・混合"

        highest = rows.loc[rows["購入率"].idxmax()]
        lowest = rows.loc[rows["購入率"].idxmin()]
        records.append(
            {
                "アンケート": survey,
                "最高回答": highest["回答値"],
                "最高購入率": float(highest["購入率"]),
                "最高回答企業数": int(highest["全企業数"]),
                "最低回答": lowest["回答値"],
                "最低購入率": float(lowest["購入率"]),
                "回答済み購入率差": rate_range,
                "Spearman_rho": float(rho),
                "Spearman_p": float(rho_p),
                "Cramers_V_欠損含む": v_all,
                "Cramers_V_回答済み": v_answered,
                "chi2_p_欠損含む": p_all,
                "chi2_p_回答済み": p_answered,
                "傾向": pattern,
            }
        )
    return pd.DataFrame(records)


def pct(value: float) -> str:
    return f"{value:.1%}"


def survey_bullet_list(metrics: pd.DataFrame, direction: str) -> str:
    selected = metrics[metrics["傾向"].str.contains(direction, regex=False)]
    if selected.empty:
        return "- 該当なし"
    return "\n".join(
        f"- **{row['アンケート']}**：{row['傾向']}（Spearman rho {row['Spearman_rho']:+.3f}）"
        for _, row in selected.iterrows()
    )


def write_analysis(
    path: Path,
    train: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    font_name: str,
    font_path: Path,
) -> None:
    lines = [
        "# アンケート回答と購入フラグの関係",
        "",
        "## 概要",
        "",
        f"- 対象：`data/train.csv` の {len(train):,} 社（購入 {int(train[TARGET].sum()):,} 社、非購入 {int((train[TARGET] == 0).sum()):,} 社）",
        "- 購入率は `購入企業数 / 全企業数` で計算し、欠損は削除せず `欠損` として集計しました。",
        f"- グラフの日本語フォント：`{font_name}`（`{font_path}`）",
        "",
        "## 各アンケートで購入率が最も高い回答",
        "",
        "| アンケート | 最高回答 | 購入率 | 企業数 | 最低回答 | 最低購入率 | 回答済み差 | 傾向 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['アンケート']} | {row['最高回答']} | {pct(row['最高購入率'])} | "
            f"{int(row['最高回答企業数'])} | {row['最低回答']} | {pct(row['最低購入率'])} | "
            f"{pct(row['回答済み購入率差'])} | {row['傾向']} |"
        )

    special = metrics[metrics["傾向"] == "特定回答が高い"]
    flat = metrics[metrics["傾向"] == "回答による差が小さい"]
    lines.extend(
        [
            "",
            "## 回答値が上がるほど購入率が高いアンケート",
            "",
            survey_bullet_list(metrics, "上昇"),
            "",
            "## 回答値が上がるほど購入率が低いアンケート",
            "",
            survey_bullet_list(metrics, "下降"),
            "",
            "## 特定回答だけ購入率が高いアンケート",
            "",
            "\n".join(
                f"- **{row['アンケート']}**：回答{row['最高回答']}が{pct(row['最高購入率'])}で最高"
                for _, row in special.iterrows()
            )
            if not special.empty
            else "- 該当なし",
            "",
            "## 回答による差がほとんどないアンケート",
            "",
            "\n".join(
                f"- **{row['アンケート']}**：回答済みカテゴリ間の購入率差は{pct(row['回答済み購入率差'])}"
                for _, row in flat.iterrows()
            )
            if not flat.empty
            else "- 該当なし",
            "",
            "## 購入フラグとの関係が特に強そうな上位3つ",
            "",
        ]
    )

    ranked = metrics.sort_values(
        ["Cramers_V_欠損含む", "Cramers_V_回答済み"], ascending=False
    ).head(3)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        missing_note = ""
        if row["Cramers_V_欠損含む"] - row["Cramers_V_回答済み"] >= 0.03:
            missing_note = " 欠損を含めた関連が強く、欠損パターンの影響にも注意が必要です。"
        lines.append(
            f"{rank}. **{row['アンケート']}** — 補正Cramér's Vは欠損込み{row['Cramers_V_欠損含む']:.3f}、"
            f"回答済みのみ{row['Cramers_V_回答済み']:.3f}。最高は回答{row['最高回答']}の"
            f"{pct(row['最高購入率'])}、最低は回答{row['最低回答']}の{pct(row['最低購入率'])}です。{missing_note}"
        )

    smallest_group = int(summary["全企業数"].min())
    lines.extend(
        [
            "",
            "## 解釈上の注意",
            "",
            "- これは記述的な関連であり、アンケート回答が購入を引き起こしたことを示すものではありません。",
            f"- 最小カテゴリは {smallest_group} 社です。企業数が少ないカテゴリの購入率は変動しやすいため、棒の高さと企業数を併せて確認してください。",
            "- アンケート7の欠損はアンケート6の回答条件に伴う構造的欠損の可能性があるため、通常の無回答と同一視しないでください。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="アンケート回答別の購入数・購入率を集計して可視化します。")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    train_path = project_dir / "data" / "train.csv"
    output_dir = project_dir / "output" / "analysis" / "survey"
    chart_dir = output_dir / "charts"
    summary_path = output_dir / "survey_purchase_summary.csv"
    analysis_path = output_dir / "SURVEY_PURCHASE_ANALYSIS.md"

    train = pd.read_csv(train_path)
    required = {TARGET, *SURVEY_COLUMNS}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"必要な列がありません: {missing}")
    if not set(train[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} は0/1である必要があります。")

    font_name, font_path = configure_japanese_font()
    summary = summarize_survey(train)
    metrics = association_metrics(train, summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    create_charts(summary, chart_dir)
    write_analysis(analysis_path, train, summary, metrics, font_name, font_path)

    print(f"Font: {font_name} ({font_path})")
    print(f"Summary: {summary_path} ({len(summary)} rows)")
    print(f"Charts: {chart_dir} ({len(list(chart_dir.glob('*.png')))} PNG files)")
    print(f"Analysis: {analysis_path}")
    print("Top 3 associations:")
    ranked = metrics.sort_values(
        ["Cramers_V_欠損含む", "Cramers_V_回答済み"], ascending=False
    ).head(3)
    for _, row in ranked.iterrows():
        print(
            f"  {row['アンケート']}: V_all={row['Cramers_V_欠損含む']:.3f}, "
            f"V_answered={row['Cramers_V_回答済み']:.3f}"
        )


if __name__ == "__main__":
    main()
