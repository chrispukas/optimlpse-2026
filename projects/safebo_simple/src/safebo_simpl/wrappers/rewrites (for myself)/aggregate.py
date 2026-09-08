from __future__ import annotations
import enum

from dataclasses import dataclass, field

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from typing import Any

from ddo_suite.experiments.runner import BenchmarkRecord, BenchmarkResults


class Central(enum.Enum):
    MEDIAN = 0
    MEAN = 1

central_mapping: dict[str, Central] = \
    {
        "median": Central.MEDIAN,
        "mean": Central.MEAN
    }


@dataclass(frozen=True, slots=True)
class AggregatedCurve:
    problem_id: str
    n_x: int
    algorithm_id: str
    evals: NDArray[np.float64]
    mean: NDArray[np.float64]
    median: NDArray[np.float64]
    q_low: NDArray[np.float64]
    q_high: NDArray[np.float64]
    std: NDArray[np.float64]
    n_seeds: int


@dataclass(frozen=True, slots=True)
class AUCScore:
    problem_id: str
    n_x: int
    algorithm_id: str
    score: float
    warm_start: int
    n_seeds: int

@dataclass
class AggregatedResults:
    results: BenchmarkResults
    _curve_attribute_map: dict[str, AggregatedResultCurve] = field(default_factory=dict)
    _valid_attributes: list[str] = field(default_factory=list)

    _curr_attribute: str = field(default_factory=str, init=False)

    def set_attribute_id(self, attribute_id: str) -> None:
        if attribute_id not in self._curve_attribute_map:
            raise ValueError(f"Attribute {attribute_id} not found in {list(self._curve_attribute_map.keys())}!")
        self._curr_attribute: str = attribute_id

    def _select_curve_by_attribute(self, attribute_id: str) -> AggregatedResultCurve:
        if attribute_id not in self._curve_attribute_map:
            raise ValueError(f"Attribute {attribute_id} not found in {list(self._curve_attribute_map.keys())}!")
        return self._curve_attribute_map[attribute_id]
    
    def __getitem__(self, key: tuple[str, int, str]) -> AggregatedCurve:
        return self._select_curve_by_attribute(self._curr_attribute).__getitem__(key=key)

    def __iter__(self):
        return self._select_curve_by_attribute(self._curr_attribute).__iter__()

    def __len__(self) -> int:
        return self._select_curve_by_attribute(self._curr_attribute).__len__()

    def __contains__(self, key: tuple[str, int, str]) -> bool:
        return key in self._select_curve_by_attribute(self._curr_attribute)._curve_index

    @property
    def curves(self) -> list[AggregatedCurve]:
        return self._select_curve_by_attribute(self._curr_attribute).curves

    @property
    def auc_scores(self) -> list[AUCScore]:
        return self._select_curve_by_attribute(self._curr_attribute)._auc

    def auc(self, key: tuple[str, int, str]) -> AUCScore:
        return self._select_curve_by_attribute(self._curr_attribute).auc(key=key)

    @property
    def by_problem(self) -> dict[str, list[AggregatedCurve]]:
        return self._select_curve_by_attribute(self._curr_attribute).by_problem

    @property
    def by_dimension(self) -> dict[int, list[AggregatedCurve]]:
        return self._select_curve_by_attribute(self._curr_attribute).by_dimension

    def ranking(self, problem_id: str, n_x: int) -> list[tuple[str, float]]:
        return self._select_curve_by_attribute(self._curr_attribute).ranking(problem_id=problem_id, n_x=n_x)

@dataclass 
class AggregatedResultCurve:
    _curves: list[AggregatedCurve] = field(default_factory=list)
    _auc: list[AUCScore] = field(default_factory=list)
    _curve_index: dict[tuple[str, int, str], AggregatedCurve] = field(default_factory=dict, init=False)
    _auc_index: dict[tuple[str, int, str], AUCScore] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._curve_index = {(c.problem_id, c.n_x, c.algorithm_id): c for c in self._curves}
        self._auc_index = {(a.problem_id, a.n_x, a.algorithm_id): a for a in self._auc}

    def __getitem__(self, key: tuple[str, int, str]) -> AggregatedCurve:
        try:
            return self._curve_index[key]
        except KeyError:
            raise KeyError(
                f"No curve for {key!r}. Available (first 10): "
                f"{sorted(self._curve_index.keys())[:10]}"
            ) from None
        
    def __iter__(self):
        return iter(self._curves)

    def __len__(self) -> int:
        return len(self._curves)

    def __contains__(self, key: tuple[str, int, str]) -> bool:
        return key in self._curve_index

    @property
    def curves(self) -> list[AggregatedCurve]:
        return list(self._curves)

    @property
    def auc_scores(self) -> list[AUCScore]:
        return list(self._auc)

    def auc(self, key: tuple[str, int, str]) -> AUCScore:
        try:
            return self._auc_index[key]
        except KeyError:
            raise KeyError(
                f"No AUC score for {key!r}. Available (first 10): "
                f"{sorted(self._auc_index.keys())[:10]}"
            ) from None

    @property
    def by_problem(self) -> dict[str, list[AggregatedCurve]]:
        out: dict[str, list[AggregatedCurve]] = {}
        for c in self._curves:
            out.setdefault(c.problem_id, []).append(c)
        return out

    @property
    def by_dimension(self) -> dict[int, list[AggregatedCurve]]:
        out: dict[int, list[AggregatedCurve]] = {}
        for c in self._curves:
            out.setdefault(c.n_x, []).append(c)
        return out

    def ranking(self, problem_id: str, n_x: int) -> list[tuple[str, float]]:
        scored = [
            (a.algorithm_id, a.score)
            for a in self._auc
            if a.problem_id == problem_id and a.n_x == n_x
        ]

        def sort_key(item: tuple[str, float]) -> tuple[int, float, str]:
            alg, score = item
            if np.isnan(score):
                return (1, 0.0, alg)
            return (0, -score, alg)

        return sorted(scored, key=sort_key)



def _pad_to_budget(curve: list[float], budget: int) -> NDArray[np.float64]:
    arr = np.asarray(curve, dtype=np.float64)
    if arr.size == 0:
        return np.full(budget, np.nan)
    if arr.size >= budget:
        return arr[:budget]
    pad = np.full(budget - arr.size, arr[-1])
    return np.concatenate([arr, pad])


def _resolve_warm_start(warm_start: int | float | None, budget: int) -> int:
    if warm_start is None:
        return int(0.3 * budget)
    if isinstance(warm_start, float) and 0.0 < warm_start < 1.0:
        return int(warm_start * budget)
    ws = int(warm_start)
    if ws < 0:
        raise ValueError(f"warm_start must be non-negative, got {warm_start}")
    if ws >= budget:
        raise ValueError(
            f"warm_start ({ws}) must be less than the budget ({budget}); "
            f"otherwise no evaluations are counted."
        )
    return ws


def _stack_successful_curves(records: list[BenchmarkRecord], budget: int, attribute_id: str) -> NDArray[np.float64]:
    curves: list[NDArray[np.float64]] = []
    for record in records:
        if not hasattr(record, attribute_id):
            raise ValueError(f"Provided attribute id (metric) {attribute_id} does not exist in the {record.__class__.__name__} class!")

        unsafe_attribute: list[float] | Any = getattr(record, attribute_id)
        if not isinstance(unsafe_attribute, list) and unsafe_attribute is not None: # Can only be either a list, or a Nonetype
            raise ValueError(f"Provide attribute id (metric) {attribute_id} is the incorrect expected type!",
                             f" is: ({type(unsafe_attribute).__name__}), expected a list or None!")
        if record.failed:
            continue
        if not unsafe_attribute:
            continue

        padded_rolling: NDArray[np.float64] = _pad_to_budget(
            curve=unsafe_attribute,
            budget=budget,
        )
        curves.append(padded_rolling)

    if not curves:
        return np.empty((0, budget), dtype=np.float64)
    return np.vstack(curves)


def aggregate_results(
    results: BenchmarkResults,
    quantiles: tuple[float, float] = (0.25, 0.75),
    warm_start: int | float | None = None,
    auc_curve: str = "median",
    auc_log: bool = False,
    attributes_to_check: list[str] = ["f_best"]
) -> AggregatedResults:
    _check_if_can_aggregate(quantiles=quantiles, auc_curve=auc_curve)
    records: list[BenchmarkRecord] = list(results)
    if not records:
        return AggregatedResults(results, {}, _valid_attributes=attributes_to_check)

    group_budget: dict[tuple[str, int], int] = _get_record_budget_map(records=records)
    groups: dict[tuple[str, int, str], list[BenchmarkRecord]] = _get_record_map(records=records)

    result_pairs: dict[str, AggregatedResultCurve] = {}

    for attribute_id in attributes_to_check:
        (curves, representative) = _aggregate_curve_scores(budget=group_budget, groups=groups, 
                                                        quantiles=quantiles, auc_curve=auc_curve, attribute_id=attribute_id)
        AUC_scores: list[AUCScore] = _aggregate_AUC_scores(representative=representative, budget=group_budget,
                                                        curves=curves, warm_start=warm_start, auc_log=auc_log)

        result_pairs[attribute_id] = AggregatedResultCurve(curves, AUC_scores)
    
    agg_results: AggregatedResults = AggregatedResults(results, _curve_attribute_map=result_pairs, _valid_attributes=attributes_to_check)
    agg_results.set_attribute_id(attribute_id=attributes_to_check[0])
    return agg_results


def _aggregate_curve_scores(
        budget: dict[tuple[str, int], int],
        groups: dict[tuple[str, int, str], list[BenchmarkRecord]], 
        quantiles: tuple[float, float], auc_curve: str, attribute_id: str, 
        ) -> tuple[list[AggregatedCurve], dict[tuple[str, int, str], NDArray[np.float64]]]:

    curves: list[AggregatedCurve] = []
    representative: dict[tuple[str, int, str], NDArray[np.float64]] = {}

    for (problem_id, n_x, algorithm_id), recs in groups.items():
        budget_single: int = budget[(problem_id, n_x)]
        evals_axis: NDArray[np.float64] = np.arange(1, budget_single + 1, dtype=np.float64)
        stacked_curves: NDArray[np.float64] = _stack_successful_curves(recs, budget_single, attribute_id=attribute_id)
        n_seeds: int = stacked_curves.shape[0]

        key: tuple[str, int, str] = (problem_id, n_x, algorithm_id)

        if n_seeds == 0:
            curve: AggregatedCurve = _default_agg_curve(evals_axis=evals_axis, key=key, budget_single=budget_single)
            curves.append(curve)
            continue

        (curve, rolling) = _get_agg_curve(evals_axis=evals_axis, key=key, auc_curve=auc_curve,
                                          stacked_curves=stacked_curves, quantiles=quantiles)
        curves.append(curve)
        representative[key] = rolling
    return (curves, representative)

def _default_agg_curve(evals_axis: NDArray[np.float64], key: tuple[str, int, str], budget_single: int) -> AggregatedCurve:
    (problem_id, dim, algorithm_id) = key

    nan: NDArray[np.float64] = np.full(budget_single, np.nan)
    curve: AggregatedCurve = AggregatedCurve(
            problem_id=problem_id,
            n_x=dim,
            algorithm_id=algorithm_id,
            evals=evals_axis,
            mean=nan,
            median=nan.copy(),
            q_low=nan.copy(),
            q_high=nan.copy(),
            std=nan.copy(),
            n_seeds=0,
        )
    return curve

def _get_agg_curve(evals_axis: NDArray[np.float64], key: tuple[str, int, str], auc_curve: str,
                   stacked_curves: NDArray[np.float64], quantiles: tuple[float, float]) -> tuple[AggregatedCurve, NDArray[np.float64]]:
    (q_low, q_high) = quantiles
    (problem_id, dim, algorithm_id) = key
    
    mean: NDArray[np.float64] = stacked_curves.mean(axis=0)
    median: NDArray[np.float64] = np.median(stacked_curves, axis=0)
    qh = np.quantile(stacked_curves, q_high, axis=0)
    ql: NDArray[np.float64] = np.quantile(stacked_curves, q_low, axis=0)
    std = stacked_curves.std(axis=0)

    curve: AggregatedCurve = AggregatedCurve(
        problem_id=problem_id,
        n_x=dim,
        algorithm_id=algorithm_id,
        evals=evals_axis,
        mean=mean,
        median=median,
        q_low=ql,
        q_high=qh,
        std=std,
        n_seeds=stacked_curves.shape[0],
    )
    rolling: NDArray[np.float64] = median if auc_curve == "median" else mean
    return (curve, rolling)


def _aggregate_AUC_scores(representative: dict[tuple[str, int, str], NDArray[np.float64]], budget: dict[tuple[str, int], int],
                          curves: list[AggregatedCurve], warm_start: int | float | None = None, auc_log: bool = False) -> list[AUCScore]:
    by_pd: dict[tuple[str, int], dict[str, NDArray[np.float64]]] = {}
    for (problem_id, n_x, algorithm_id), curve in representative.items():
        by_pd.setdefault((problem_id, n_x), {})[algorithm_id] = curve

    auc_scores: list[AUCScore] = []
    n_seeds_lookup = {(c.problem_id, c.n_x, c.algorithm_id): c.n_seeds for c in curves}

    for (problem_id, n_x), alg_curves in by_pd.items():
        algorithm_ids: list[str] = sorted(alg_curves.keys())
        ws: int = _resolve_warm_start(warm_start, budget[(problem_id, n_x)])

        if len(algorithm_ids) < 2:
            for algorithm_id in algorithm_ids:
                auc_scores.append(
                    AUCScore(
                        problem_id=problem_id,
                        n_x=n_x,
                        algorithm_id=algorithm_id,
                        score=float("nan"),
                        warm_start=ws,
                        n_seeds=n_seeds_lookup[(problem_id, n_x, algorithm_id)],
                    )
                )
            continue

        curves_vstack: NDArray[np.float64] = np.vstack([alg_curves[a] for a in algorithm_ids])
        mat: NDArray[np.float64] = np.log10(np.maximum(curves_vstack, 1e-12)) if auc_log else curves_vstack

        best: NDArray[np.float64] = mat.min(axis=0)  # lowest objective at each eval (best)
        worst: NDArray[np.float64] = mat.max(axis=0)  # highest objective at each eval (worst)
        denom: NDArray[np.float64] = (worst - best) + 1e-12

        for idx, algorithm_id in enumerate(algorithm_ids):
            perf: NDArray[np.float64] = (worst - mat[idx]) / denom
            score: float = float(np.mean(perf[ws:]))
            auc_scores.append(
                AUCScore(
                    problem_id=problem_id,
                    n_x=n_x,
                    algorithm_id=algorithm_id,
                    score=score,
                    warm_start=ws,
                    n_seeds=n_seeds_lookup[(problem_id, n_x, algorithm_id)],
                )
            )
    return auc_scores

def _check_if_can_aggregate(quantiles: tuple[float, float], auc_curve: str) -> None:
    (q_low, q_high) = quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"quantiles must satisfy 0 <= low < high <= 1, got {quantiles}")
    if auc_curve not in central_mapping:
        raise ValueError(f"auc_curve must be one of: ({list(central_mapping.keys())}), got {auc_curve!r}")

def _get_record_budget_map(records: list[BenchmarkRecord]) -> dict[tuple[str, int], int]:
    budget_map: dict[tuple[str, int], int] = defaultdict(int)
    for r in records:
        key = (r.problem_id, r.n_x)
        budget_map[key] = max(budget_map.get(key, 0), r.max_evals)
    return budget_map

def _get_record_map(records: list[BenchmarkRecord]) -> dict[tuple[str, int, str], list[BenchmarkRecord]]:
    groups: dict[tuple[str, int, str], list[BenchmarkRecord]] = defaultdict(list)
    for r in records:
        key: tuple[str, int, str] = (r.problem_id, r.n_x, r.algorithm_id)
        groups[key].append(r)
    return groups