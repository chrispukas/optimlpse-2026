from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ddo_suite.experiments.aggregate import AggregatedResults, AggregatedCurve

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import enum

class Bands(enum.Enum):
    QUANTILE = 0
    STD = 1

band_mapping: dict[str, Bands] = \
    {
        "quantile": Bands.QUANTILE,
        "std": Bands.STD,
    }

class Central(enum.Enum):
    MEDIAN = 0
    MEAN = 1

central_mapping: dict[str, Central] = \
    {
        "median": Central.MEDIAN,
        "mean": Central.MEAN
    }


@dataclass
class _PlotConfig:
    n_cols: int = 2
    max_per_fig: int = 8
    log_y: bool = True
    band: Bands | str = "quantile"
    rolling_line: Central | str = "median"

    padding_at_zero: bool = False

    x_label: str = "function evaluations"
    y_label: str = "objective value (best so far)"
    
    def _remap_factory[T_Enum: enum.Enum](self, attribute_id: str, mapping: dict[str, T_Enum]) -> None:
        if not hasattr(self, attribute_id):
            raise ValueError(f"Invalid attribute id: {attribute_id}!")

        attr_val: Any = getattr(self, attribute_id)
        if isinstance(attr_val, str):
            if attr_val not in mapping:
                raise ValueError(f"({type(list(mapping.values()))}) {attr_val} must be one of {mapping.keys()}")
            setattr(self, attribute_id, mapping[attr_val])

    def __post_init__(self) -> None:
        self._remap_factory("band", mapping=band_mapping)
        self._remap_factory("rolling_line", mapping=central_mapping)

# ----------------------------------
# --- EXPOSED PLOTTING FUNCTIONS ---
# ----------------------------------

# Tracked attributes are found in BenchmarkResults

def plot_all_convergence(
    aggregated: AggregatedResults,
    n_cols: int = 2, max_per_fig: int = 8, log_y: bool = True, band: str = "quantile", line: str = "median", 
    *args: Any, **kwargs: Any,
) -> list[Figure]:
    plot_config: _PlotConfig = _PlotConfig(n_cols=n_cols, max_per_fig=max_per_fig, log_y=log_y, band=band, rolling_line=line)
    aggregated.set_attribute_id("best_f")
    return _get_figures_for_curves(agg=aggregated, curves=aggregated.curves, config=plot_config)


def plot_all_cumulative_elapsed(
    aggregated: AggregatedResults,
    n_cols: int = 2, max_per_fig: int = 8, log_y: bool = True, band: str = "quantile", line: str = "median", 
    *args: Any, **kwargs: Any,
) -> list[Figure]:
    plot_config: _PlotConfig = _PlotConfig(n_cols=n_cols, max_per_fig=max_per_fig, log_y=log_y, band=band, rolling_line=line)
    plot_config.y_label = "cumulative elapsed (sec)"
    aggregated.set_attribute_id("cumulative_elapsed")
    return _get_figures_for_curves(agg=aggregated, curves=aggregated.curves, config=plot_config)

def plot_all_elapsed(
    aggregated: AggregatedResults,
    n_cols: int = 2, max_per_fig: int = 8, log_y: bool = True, band: str = "quantile", line: str = "median", 
    *args: Any, **kwargs: Any,
) -> list[Figure]:
    plot_config: _PlotConfig = _PlotConfig(n_cols=n_cols, max_per_fig=max_per_fig, log_y=log_y, band=band, rolling_line=line)
    plot_config.y_label = "elapsed (sec)"
    plot_config.padding_at_zero = True
    aggregated.set_attribute_id("elapsed")
    return _get_figures_for_curves(agg=aggregated, curves=aggregated.curves, config=plot_config)

def plot_all_cumulative_violations(
    aggregated: AggregatedResults,
    n_cols: int = 2, max_per_fig: int = 8, band: str = "quantile", line: str = "median", 
    *args: Any, **kwargs: Any,
) -> list[Figure]:
    plot_config: _PlotConfig = _PlotConfig(n_cols=n_cols, max_per_fig=max_per_fig, log_y=False, band=band, rolling_line=line)
    plot_config.y_label = "cumulative violations (-)"
    aggregated.set_attribute_id("c_violations")
    return _get_figures_for_curves(agg=aggregated, curves=aggregated.curves, config=plot_config)







 
def _algorithm_colors(algorithm_ids: list[str]) -> dict[str, Any]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    ordered = sorted(set(algorithm_ids))
    return {alg: cycle[i % len(cycle)] for i, alg in enumerate(ordered)}


def _choose_y_scale(values: NDArray[np.float64], log_y: bool) -> str:
    if not log_y:
        return "linear"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "linear"
    if np.any(finite <= 0):
        return "symlog"
    return "log"

def _plot_single_generic(
    aggregated: AggregatedResults,
    problem_id: str,
    dim: int,
    config: _PlotConfig,
    axes: Axes | None = None,
    colours: dict[str, Any] | None = None,
    show_warm_start: bool = True,
) -> Axes:
    _check_config_validity(config=config)
    curves: list[AggregatedCurve] = _get_curves_by_id_dim(aggregated.curves, id=problem_id, dim=dim)
    ax: Axes = axes or plt.subplots()[1]

    algorithm_ids: list[str] = sorted(c.algorithm_id for c in curves)
    colors: dict[str, Any] = colours or _algorithm_colors(algorithm_ids)

    stacked_rolling_central: list[NDArray[np.float64]] = []
    for curve in sorted(curves, key=lambda c: c.algorithm_id):
        curve: AggregatedCurve = _replace_nan(curve, config=config)
        _, central = _fill_curve_single(curve=curve, axes=ax, colors=colors, config=config)
        stacked_rolling_central.append(central)

    _rescale(stacked_rolling_central, axes=ax, config=config)

    if show_warm_start:
        warm_start = next(
            (
                s.warm_start for s in aggregated.auc_scores if s.problem_id == problem_id and s.n_x == dim), None,
            )
        if warm_start is not None and warm_start > 0:
            ax.axvline(warm_start, color="0.5", ls="--", lw=1.0, alpha=0.7)
            ax.text(
                warm_start,
                0.99,
                " counting starts ",
                transform=ax.get_xaxis_transform(),
                fontsize="x-small",
                color="0.4",
                ha="left",
                va="top",
                rotation=90,
            )

    return _display_config_on_axes(ax=ax, id=problem_id, dim=dim, config=config)

def _replace_nan(curve: AggregatedCurve, config: _PlotConfig) -> AggregatedCurve:
    return curve.fill_nans(val=0. if config.padding_at_zero else None)

def _fill_curve_single(curve: AggregatedCurve, axes: Axes, colors: dict[str, Any], config: _PlotConfig) -> tuple[Axes, NDArray[np.float64]]:
    color = colors.get(curve.algorithm_id)
    if curve.n_seeds == 0:
        axes.plot([], [], color=color, label=f"{curve.algorithm_id} (failed)")
        return (axes, np.array([]))

    central: NDArray[np.float64] = _determine_central(curve=curve, method=config.rolling_line)
    (lo, hi) = _determine_range(curve=curve, central=central, method=config.band)

    axes.plot(
        curve.evals,
        central,
        color=color,
        label=_write_label(curve=curve),
    )
    axes.fill_between(curve.evals, lo, hi, color=color, alpha=0.2)
    return (axes, central)

def _rescale(stacked_rolling_central: list[NDArray[np.float64]], axes: Axes, config: _PlotConfig):
    if stacked_rolling_central:
        stacked = np.concatenate(stacked_rolling_central)
        axes.set_yscale(_choose_y_scale(stacked, config.log_y))

def _write_label(curve: AggregatedCurve) -> str:
    return f"{curve.algorithm_id} (n={curve.n_seeds})"

def _determine_central(curve: AggregatedCurve, method: str | Central) -> NDArray[np.float64]:
    if not isinstance(method, Central):
        raise ValueError(f"Method is of type: {type(method)}, must be of type: {type(Central)}")
    match method:
        case Central.MEDIAN:
            return curve.median
        case Central.MEAN:
            return curve.mean

def _determine_range(curve: AggregatedCurve, central: NDArray[np.float64], method: str | Bands) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(method, Bands):
        raise ValueError(f"Method is of type: {type(method)}, must be of type: {type(Bands)}")
    match method:
        case Bands.QUANTILE:
            return (curve.q_low, curve.q_high)
        case Bands.STD:
            return (central - curve.std, central + curve.std)
        

def _display_config_on_axes(ax: Axes, id: str, dim: int, config: _PlotConfig) -> Axes:
    ax.set_xlabel(config.x_label)
    ax.set_ylabel(config.y_label)
    ax.set_title(f"{id} (n_x={dim})")
    ax.legend(fontsize="small")
    return ax

def _get_curves_by_id_dim(curves: list[AggregatedCurve], id: str, dim: int) -> list[AggregatedCurve]:
    curves_filtered: list[AggregatedCurve] = []
    for curve in curves:
        if curve.problem_id != id:
            continue
        if curve.n_x != dim:
            continue
        curves_filtered.append(curve)

    if not curves_filtered:
        raise KeyError(f"No curves found for problem_id: ({id}), and dim: ({dim})!")
    return curves_filtered

def _check_config_validity(config: _PlotConfig) -> None:
    if not isinstance(config.band, Bands) and config.band not in band_mapping:
        raise ValueError(f"band must be 'quantile' or 'std', got {config.band!r}")
    if not isinstance(config.rolling_line, Central) and config.rolling_line not in central_mapping:
        raise ValueError(f"line must be 'median' or 'mean', got {config.rolling_line!r}")

def _get_figures_for_curves(agg: AggregatedResults, curves: list[AggregatedCurve], config: _PlotConfig):
    algorithm_ids: list[str] = sorted({c.algorithm_id for c in curves})
    colors: dict[str, int] = _algorithm_colors(algorithm_ids=algorithm_ids)
    problem_pairs: list[tuple[str, int]] = sorted({(c.problem_id, c.n_x) for c in curves})

    figures: list[Figure] = []

    for page_start in range(0, len(problem_pairs), config.max_per_fig):
        page: list[tuple[str, int]] = problem_pairs[page_start : page_start + config.max_per_fig]
        figures.append(
            _curve_page(aggregated=agg, problem_pairs=page, algorithm_ids=algorithm_ids, colours=colors, config=config)
        )
    return figures

def _curve_page(
    aggregated: AggregatedResults,
    problem_pairs: list[tuple[str, int]],
    algorithm_ids: list[str],
    colours: dict[str, Any],
    config: _PlotConfig,
) -> Figure:
    n: int = len(problem_pairs)
    n_cols: int = min(config.n_cols, n)
    n_rows: int = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows), squeeze=False)

    for idx, (problem_id, dim) in enumerate(problem_pairs):
        r, c = divmod(idx, n_cols)
        ax: Axes = axes[r][c]
        _plot_single_generic(
            aggregated,
            problem_id=problem_id,
            dim=dim,
            axes=ax,
            config=config,
            colours=colours,
        )
        # Per-subplot legends are removed in favor of one shared legend.
        _remove_legend(ax)
        _prettify(ax)

    _hide_unused_axes(n, n_rows=n_rows, n_cols=n_cols, axes=axes)
    _instantiate_legend(fig, algorithm_ids=algorithm_ids, colours=colours).tight_layout(rect=(0, 0.05, 1, 1))
    return fig

def _instantiate_legend(figure: Figure, algorithm_ids: list[str], colours: dict[str, Any]) -> Figure:
    handles: list[Line2D] = [Line2D([], [], color=colours.get(a), label=a) for a in sorted(algorithm_ids)]
    figure.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(algorithm_ids), 4),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    return figure

def _hide_unused_axes(n_pairs: int, n_rows: int, n_cols: int, axes: NDArray[np.object_]) -> None:
    for idx in range(n_pairs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

def _remove_legend(axes: Axes) -> None:    
    if (legend := axes.get_legend()) is None:
        return
    legend.remove()

def _prettify(ax: Axes) -> None:
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_alpha(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Dimension-scaling plots (not changed)
# ─────────────────────────────────────────────────────────────────────────────


def plot_all_dimension_scaling(
    aggregated: AggregatedResults,
    *,
    n_cols: int = 2,
    max_per_fig: int = 8,
) -> list[Figure]:
    problems = sorted({s.problem_id for s in aggregated.auc_scores})
    if not problems:
        raise ValueError("No AUC scores to plot.")

    all_algos = sorted({s.algorithm_id for s in aggregated.auc_scores})
    colors = _algorithm_colors(all_algos)

    figures: list[Figure] = []
    for page_start in range(0, len(problems), max_per_fig):
        page = problems[page_start : page_start + max_per_fig]
        figures.append(_scaling_page(aggregated, page, all_algos, colors, n_cols))
    return figures



def plot_dimension_scaling(
    aggregated: AggregatedResults,
    problem_id: str,
    *,
    ax: Axes | None = None,
    colors: dict[str, Any] | None = None,
) -> Axes:
    scores = [s for s in aggregated.auc_scores if s.problem_id == problem_id]
    if not scores:
        raise KeyError(f"No AUC scores for problem_id={problem_id!r}")

    if ax is None:
        _, ax = plt.subplots()

    algo_ids = sorted({s.algorithm_id for s in scores})
    if colors is None:
        colors = _algorithm_colors(algo_ids)

    for alg in algo_ids:
        alg_scores = sorted((s for s in scores if s.algorithm_id == alg), key=lambda s: s.n_x)
        dims = [s.n_x for s in alg_scores]
        vals = [s.score for s in alg_scores]
        ax.plot(dims, vals, marker="o", color=colors.get(alg), label=alg)

    ax.set_xlabel("dimension (n_x)")
    ax.set_ylabel("normalized AUC score")
    ax.set_title(f"{problem_id}: ranking vs dimension")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize="small")
    return ax


def _scaling_page(
    aggregated: AggregatedResults,
    problems: list[str],
    all_algos: list[str],
    colors: dict[str, Any],
    n_cols: int,
) -> Figure:
    n = len(problems)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows), squeeze=False)

    for idx, problem_id in enumerate(problems):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        plot_dimension_scaling(aggregated, problem_id, ax=ax, colors=colors)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        _prettify(ax)

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    handles = [
        Line2D([], [], color=colors.get(a), marker="o", label=a) for a in sorted(all_algos)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(all_algos), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────


def save_figure(fig: Any, path: str, dpi: int = 150) -> None:
    """Write a figure to disk (format inferred from the file extension).

    Accepts any matplotlib figure-like object exposing ``savefig``
    (``Figure`` or ``SubFigure``).
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
