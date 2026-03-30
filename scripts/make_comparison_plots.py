"""
make_comparison_plots.py - Create comparison plots from repeated experiment CSVs.

Usage:
    python -m scripts.make_comparison_plots
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd

from src.data import is_maximization


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.csv"
AGGREGATED_RESULTS_FILE = RESULTS_DIR / "aggregated_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
BOXPLOTS_DIR = PLOTS_DIR / "boxplots"
MEDIAN_COMPARISONS_DIR = PLOTS_DIR / "median_comparisons"

RAW_REQUIRED_COLUMNS = {
    "algorithm",
    "dataset",
    "budget",
    "best_performance",
}
AGG_REQUIRED_COLUMNS = {
    "algorithm",
    "dataset",
    "budget",
    "median_best_performance",
    "q1_best_performance",
    "q3_best_performance",
}

ALGORITHM_ALIASES = {
    "baseline_random_search": "baseline_random_search",
    "bestconfig": "improved_bestconfig",
    "improved_bestconfig": "improved_bestconfig",
}
ALGORITHM_DISPLAY_NAMES = {
    "baseline_random_search": "Random Search",
    "improved_bestconfig": "BestConfig",
}
ALGORITHM_ORDER = ["baseline_random_search", "improved_bestconfig"]
MARKER_CYCLE = ["o", "s", "^", "D", "P", "X"]
LINESTYLE_CYCLE = ["-", "--", "-.", ":"]
MEDIAN_ERRORBAR_CAPSIZE = 3
MEDIAN_LABEL_BUDGET_LIMIT = 4
LABEL_HORIZONTAL_SHIFT_POINTS = [-12, 12, -18, 18, -24, 24]
BOX_LINEWIDTH = 1.6
WHISKER_LINEWIDTH = 1.6
CAP_LINEWIDTH = 1.6
MEDIAN_LINEWIDTH = 2.4


def ensure_required_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def ensure_required_columns(df, required_columns, file_path):
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
        )


def canonical_algorithm_name(name):
    normalized_name = str(name).strip()
    return ALGORITHM_ALIASES.get(normalized_name, normalized_name)


def prepare_results(df):
    prepared_df = df.copy()
    prepared_df["algorithm"] = prepared_df["algorithm"].map(canonical_algorithm_name)
    return prepared_df


def display_algorithm_name(name):
    return ALGORITHM_DISPLAY_NAMES.get(name, name)


def algorithm_names_for(*dataframes):
    algorithms = set()
    for df in dataframes:
        algorithms.update(df["algorithm"].dropna().unique())

    ordered_algorithms = [name for name in ALGORITHM_ORDER if name in algorithms]
    remaining_algorithms = sorted(name for name in algorithms if name not in ordered_algorithms)
    return ordered_algorithms + remaining_algorithms


def dataset_names_for(*dataframes):
    datasets = set()
    for df in dataframes:
        datasets.update(df["dataset"].dropna().unique())
    return sorted(datasets)


def objective_note(dataset_name):
    if is_maximization(dataset_name):
        return "higher is better"
    return "lower is better"


def ensure_output_directories():
    BOXPLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MEDIAN_COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    ensure_required_file(RAW_RESULTS_FILE)
    ensure_required_file(AGGREGATED_RESULTS_FILE)

    raw_results = pd.read_csv(RAW_RESULTS_FILE)
    aggregated_results = pd.read_csv(AGGREGATED_RESULTS_FILE)

    ensure_required_columns(raw_results, RAW_REQUIRED_COLUMNS, RAW_RESULTS_FILE)
    ensure_required_columns(
        aggregated_results, AGG_REQUIRED_COLUMNS, AGGREGATED_RESULTS_FILE
    )

    return prepare_results(raw_results), prepare_results(aggregated_results)


def algorithm_colors(algorithm_names):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3"]

    return {
        algorithm_name: default_colors[index % len(default_colors)]
        for index, algorithm_name in enumerate(algorithm_names)
    }


def algorithm_styles(algorithm_names):
    return {
        algorithm_name: {
            "marker": MARKER_CYCLE[index % len(MARKER_CYCLE)],
            "linestyle": LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)],
        }
        for index, algorithm_name in enumerate(algorithm_names)
    }


def finite_numeric_values(values):
    return [float(value) for value in values if pd.notna(value)]


def value_format_spec(values):
    finite_values = finite_numeric_values(values)
    if not finite_values:
        return 2, False

    max_abs = max(abs(value) for value in finite_values)
    min_non_zero = min((abs(value) for value in finite_values if value != 0), default=0.0)
    display_range = max(finite_values) - min(finite_values)

    if max_abs >= 1000:
        if display_range >= 100:
            return 0, True
        if display_range >= 10:
            return 1, True
        return 2, True

    if max_abs >= 1:
        if display_range >= 100:
            return 0, False
        if display_range >= 10:
            return 1, False
        if display_range >= 1:
            return 2, False
        return 3, False

    if min_non_zero == 0:
        return 6, False

    decimals = max(4, -math.floor(math.log10(min_non_zero)) + 1)
    if 0 < display_range < 1:
        decimals = max(decimals, -math.floor(math.log10(display_range)) + 2)
    return min(decimals, 8), False


def format_numeric_value(value, decimals, use_grouping):
    if pd.isna(value):
        return ""

    format_string = f"{{:,.{decimals}f}}" if use_grouping else f"{{:.{decimals}f}}"
    formatted_value = format_string.format(float(value))
    if "." in formatted_value:
        formatted_value = formatted_value.rstrip("0").rstrip(".")
    if formatted_value in {"-0", "-0.0"}:
        return "0"
    return formatted_value


def apply_value_axis_format(ax, values):
    decimals, use_grouping = value_format_spec(values)
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: format_numeric_value(x, decimals, use_grouping)
        )
    )
    return decimals, use_grouping


def minimum_budget_gap(budgets):
    sorted_budgets = sorted(float(budget) for budget in budgets)
    if len(sorted_budgets) < 2:
        return 1.0

    gaps = [
        sorted_budgets[index + 1] - sorted_budgets[index]
        for index in range(len(sorted_budgets) - 1)
    ]
    positive_gaps = [gap for gap in gaps if gap > 0]
    if not positive_gaps:
        return 1.0
    return min(positive_gaps)


def median_x_positions(dataset_rows, budgets, algorithm_names):
    positions = {algorithm_name: {} for algorithm_name in algorithm_names}
    overlap_step = minimum_budget_gap(budgets) * 0.04

    for budget in budgets:
        budget_rows = dataset_rows.loc[dataset_rows["budget"] == budget]
        median_groups = {}

        for algorithm_name in algorithm_names:
            algorithm_budget_rows = budget_rows.loc[
                budget_rows["algorithm"] == algorithm_name
            ]
            if algorithm_budget_rows.empty:
                continue

            median_value = float(
                algorithm_budget_rows["median_best_performance"].iloc[0]
            )
            median_groups.setdefault(median_value, []).append(algorithm_name)

        for grouped_algorithms in median_groups.values():
            ordered_group = sorted(
                grouped_algorithms,
                key=lambda name: algorithm_names.index(name),
            )
            count = len(ordered_group)
            offsets = [
                (index - (count - 1) / 2) * overlap_step
                for index in range(count)
            ]
            for algorithm_name, offset in zip(ordered_group, offsets):
                positions[algorithm_name][budget] = float(budget) + offset

    return positions


def identical_medians_across_algorithms(dataset_rows, budgets, algorithm_names):
    if len(algorithm_names) < 2:
        return False

    for budget in budgets:
        budget_rows = dataset_rows.loc[dataset_rows["budget"] == budget]
        budget_medians = []
        for algorithm_name in algorithm_names:
            algorithm_rows = budget_rows.loc[
                budget_rows["algorithm"] == algorithm_name,
                "median_best_performance",
            ]
            if algorithm_rows.empty:
                return False
            budget_medians.append(float(algorithm_rows.iloc[0]))

        if len(set(budget_medians)) != 1:
            return False

    return True


def label_closeness_threshold(values):
    finite_values = finite_numeric_values(values)
    if not finite_values:
        return 0.0

    value_range = max(finite_values) - min(finite_values)
    if value_range == 0:
        max_abs = max(abs(value) for value in finite_values)
        if max_abs == 0:
            return 1e-9
        return max_abs * 0.01

    return value_range * 0.04


def annotate_median_points(
    ax,
    plotted_points,
    algorithm_names,
    values,
    budgets,
):
    decimals, use_grouping = value_format_spec(values)
    close_threshold = label_closeness_threshold(values)

    for budget in budgets:
        budget_points = plotted_points.get(budget, [])
        if not budget_points:
            continue

        ordered_points = sorted(
            budget_points,
            key=lambda point: (point["y"], algorithm_names.index(point["algorithm"])),
        )
        last_labeled_y = None
        last_labeled_text = None

        for algorithm_index, point in enumerate(ordered_points):
            label = format_numeric_value(point["y"], decimals, use_grouping)
            is_close_to_previous = (
                last_labeled_y is not None
                and abs(point["y"] - last_labeled_y) <= close_threshold
            )

            if is_close_to_previous and label == last_labeled_text:
                continue

            x_shift = 0
            if is_close_to_previous:
                x_shift = LABEL_HORIZONTAL_SHIFT_POINTS[
                    algorithm_index % len(LABEL_HORIZONTAL_SHIFT_POINTS)
                ]

            vertical_offset = 6 + algorithm_names.index(point["algorithm"]) * 9
            ax.annotate(
                label,
                (point["x"], point["y"]),
                textcoords="offset points",
                xytext=(x_shift, vertical_offset),
                ha="center",
                fontsize=8,
            )

            last_labeled_y = point["y"]
            last_labeled_text = label


def save_figure(output_file):
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def boxplot_positions(budgets, algorithm_names):
    group_centers = list(range(1, len(budgets) + 1))
    if not algorithm_names:
        return group_centers, {}, 0.35

    if len(algorithm_names) == 1:
        return group_centers, {algorithm_names[0]: [float(x) for x in group_centers]}, 0.45

    step = min(0.35, 0.8 / len(algorithm_names))
    offsets = [
        (index - (len(algorithm_names) - 1) / 2) * step
        for index in range(len(algorithm_names))
    ]
    positions = {
        algorithm_name: [center + offsets[index] for center in group_centers]
        for index, algorithm_name in enumerate(algorithm_names)
    }
    width = max(0.18, step * 0.8)
    return group_centers, positions, width


def draw_algorithm_boxplots(ax, dataset_rows, budgets, algorithm_names, colors):
    centers, all_positions, box_width = boxplot_positions(budgets, algorithm_names)

    for algorithm_name in algorithm_names:
        series_list = []
        positions = []
        for budget_index, budget in enumerate(budgets):
            values = dataset_rows.loc[
                (dataset_rows["algorithm"] == algorithm_name)
                & (dataset_rows["budget"] == budget),
                "best_performance",
            ].dropna()

            if values.empty:
                continue

            series_list.append(values.tolist())
            positions.append(all_positions[algorithm_name][budget_index])

        if not series_list:
            continue

        artists = ax.boxplot(
            series_list,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            # Hide fliers for cleaner report visuals without changing the data.
            showfliers=False,
            boxprops={"linewidth": BOX_LINEWIDTH},
            whiskerprops={"linewidth": WHISKER_LINEWIDTH},
            capprops={"linewidth": CAP_LINEWIDTH},
            medianprops={"color": "black", "linewidth": MEDIAN_LINEWIDTH},
        )
        for box in artists["boxes"]:
            box.set(facecolor=colors[algorithm_name], alpha=0.6)
        for median in artists["medians"]:
            median.set(color="black", linewidth=MEDIAN_LINEWIDTH)

    ax.set_xticks(centers)
    ax.set_xticklabels([str(budget) for budget in budgets])
    return centers


def make_boxplot(dataset_name, raw_results, colors, algorithm_names):
    dataset_rows = raw_results.loc[raw_results["dataset"] == dataset_name].copy()
    if dataset_rows.empty:
        return None

    budgets = sorted(dataset_rows["budget"].dropna().unique().tolist())
    algorithms_in_dataset = [
        name for name in algorithm_names if name in set(dataset_rows["algorithm"].unique())
    ]
    if not budgets or not algorithms_in_dataset:
        return None

    figure_width = max(7.0, 1.8 * len(budgets) + 2.0)
    plt.figure(figsize=(figure_width, 4.8))
    ax = plt.gca()

    draw_algorithm_boxplots(ax, dataset_rows, budgets, algorithms_in_dataset, colors)
    ax.set_title(
        f"{dataset_name}: best performance by budget ({objective_note(dataset_name)})"
    )
    ax.set_xlabel("Budget")
    ax.set_ylabel("Best performance")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    apply_value_axis_format(ax, dataset_rows["best_performance"])

    legend_handles = [
        Patch(
            facecolor=colors[algorithm_name],
            edgecolor="black",
            alpha=0.6,
            label=display_algorithm_name(algorithm_name),
        )
        for algorithm_name in algorithms_in_dataset
    ]
    ax.legend(
        handles=legend_handles,
        title="Algorithm",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    output_file = BOXPLOTS_DIR / f"{dataset_name}_boxplot.png"
    save_figure(output_file)
    return output_file


def make_median_comparison_plot(dataset_name, aggregated_results, colors, algorithm_names):
    dataset_rows = aggregated_results.loc[
        aggregated_results["dataset"] == dataset_name
    ].copy()
    if dataset_rows.empty:
        return None

    budgets = sorted(dataset_rows["budget"].dropna().unique().tolist())
    algorithms_in_dataset = [
        name for name in algorithm_names if name in set(dataset_rows["algorithm"].unique())
    ]
    if not budgets or not algorithms_in_dataset:
        return None

    figure_width = max(8.5, 1.4 * len(budgets) + 3.0)
    plt.figure(figsize=(figure_width, 4.8))
    ax = plt.gca()
    styles = algorithm_styles(algorithms_in_dataset)
    x_positions = median_x_positions(dataset_rows, budgets, algorithms_in_dataset)
    all_y_values = []
    show_labels = len(budgets) <= MEDIAN_LABEL_BUDGET_LIMIT
    plotted_points = {budget: [] for budget in budgets}
    identical_medians = identical_medians_across_algorithms(
        dataset_rows,
        budgets,
        algorithms_in_dataset,
    )

    if identical_medians:
        print(
            f"Note: {dataset_name} has identical medians across algorithms at all budgets; see the boxplot for spread."
        )

    for algorithm_name in algorithms_in_dataset:
        algorithm_rows = dataset_rows.loc[
            dataset_rows["algorithm"] == algorithm_name
        ].sort_values("budget")

        x_values = [
            x_positions[algorithm_name][budget]
            for budget in algorithm_rows["budget"].tolist()
        ]
        y_values = algorithm_rows["median_best_performance"].tolist()
        lower_errors = (
            algorithm_rows["median_best_performance"]
            - algorithm_rows["q1_best_performance"]
        ).clip(lower=0)
        upper_errors = (
            algorithm_rows["q3_best_performance"]
            - algorithm_rows["median_best_performance"]
        ).clip(lower=0)

        ax.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors.tolist(), upper_errors.tolist()],
            marker=styles[algorithm_name]["marker"],
            linestyle=styles[algorithm_name]["linestyle"],
            label=display_algorithm_name(algorithm_name),
            color=colors[algorithm_name],
            linewidth=1.5,
            markersize=6,
            capsize=MEDIAN_ERRORBAR_CAPSIZE,
            elinewidth=1.0,
        )

        for x_value, budget, y_value in zip(
            x_values,
            algorithm_rows["budget"].tolist(),
            y_values,
        ):
            plotted_points[budget].append(
                {
                    "algorithm": algorithm_name,
                    "x": x_value,
                    "y": y_value,
                }
            )

        all_y_values.extend(y_values)
        all_y_values.extend(algorithm_rows["q1_best_performance"].tolist())
        all_y_values.extend(algorithm_rows["q3_best_performance"].tolist())

    title = f"{dataset_name}: median best performance by budget ({objective_note(dataset_name)})"
    if identical_medians:
        title += "\nMedians identical across algorithms; see boxplot for spread"
    ax.set_title(title)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Median best performance")
    ax.set_xticks(budgets)
    ax.grid(alpha=0.3)
    ax.legend(
        title="Algorithm",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )
    apply_value_axis_format(ax, all_y_values)

    if show_labels:
        annotate_median_points(
            ax,
            plotted_points,
            algorithms_in_dataset,
            all_y_values,
            budgets,
        )

    padding = minimum_budget_gap(budgets) * 0.12
    ax.set_xlim(min(budgets) - padding, max(budgets) + padding)

    output_file = MEDIAN_COMPARISONS_DIR / f"{dataset_name}_median_comparison.png"
    save_figure(output_file)
    return output_file


def create_plots(raw_results, aggregated_results):
    datasets = dataset_names_for(raw_results, aggregated_results)
    algorithm_names = algorithm_names_for(raw_results, aggregated_results)
    colors = algorithm_colors(algorithm_names)
    created_files = []

    for dataset_name in datasets:
        boxplot_file = make_boxplot(dataset_name, raw_results, colors, algorithm_names)
        if boxplot_file is not None:
            created_files.append(boxplot_file)

        median_plot_file = make_median_comparison_plot(
            dataset_name,
            aggregated_results,
            colors,
            algorithm_names,
        )
        if median_plot_file is not None:
            created_files.append(median_plot_file)

    return created_files


def print_summary(created_files):
    print("Created comparison plot files:")
    if not created_files:
        print("- No files were created.")
        return

    for file_path in created_files:
        print(f"- {file_path.relative_to(REPO_ROOT)}")

    print(f"Total files created: {len(created_files)}")


def main():
    ensure_output_directories()
    raw_results, aggregated_results = load_results()
    created_files = create_plots(raw_results, aggregated_results)
    print_summary(created_files)


if __name__ == "__main__":
    main()