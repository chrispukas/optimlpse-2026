from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from safebo_simpl.util import generics as su_safe

from safebo_simpl.objective_functions import ObjectiveFunction

from safebo_simpl.util.params import BOParams
from safebo_simpl.util.continuity import NormTensor
from safebo_simpl.constraints import SurrogateConstraint, NonSurrogateConstraint, Constraint
from safebo_simpl.util.s_math import LipschitzConstraints, LipschitzConstraintPair

import torch
from torch import Tensor

import numpy as np
import numpy.typing as npt

import botorch
from botorch import models as b_models
from botorch.posteriors import gpytorch as bp_gpytorch

import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import norm

import matplotlib.pyplot as plt

type BucketMetricCallable = Callable[[dict[int, Tensor]], dict[int, float]]
type MetricExtractionCallable[T_Constraint: Constraint] = Callable[[Tensor, float, list[T_Constraint]], Tensor]
type WeightingCallable = Callable[[Tensor], Tensor]
type UpdateCallable[T_Constraint: Constraint] = Callable[[Tensor, Tensor, Tensor, LipschitzConstraints], dict[int, Tensor]]
        
class VoronoiV1(su_safe.SafeBOAlgorithm):
    def __init__(self, X: Tensor, Y: Tensor, 
                 dtype: torch.dtype, device: torch.device, state: BOParams, objective_function: ObjectiveFunction,) -> None:
        super().__init__(X=X, Y=Y, dtype=dtype, device=device, state=state, objective_function=objective_function)

    def train(self) -> None:
        super()._train(single_pass=self.forward, metrics=False)

    def forward[T_Constraint: Constraint](self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        conf_level: float = self.state.convergence.confidence_level
        batch_size: int = self.state.sampling.batch_size
        constraint_objects: list[T_Constraint] = self.state.constraints.constraints or []

        lipschitz_constraint: LipschitzConstraints = LipschitzConstraints(conf_level=conf_level, surrogate_constraints=constraint_objects)
        w: Tensor = self.get_weighting(P=X, weighting_callable=self.pointwise_variance)
        (verticies, generators) = self.get_vertex_generator_pairs(P=X, w=w, tol=1e-3)

        self.plot_voronoi(X, w)

        rolling_bitmask_lipschitz_safety: dict[int, Tensor] = {}
        rolling_bitmask_constraint_violations: dict[int, Tensor] = {}

        def X_reward_func(X_input: Tensor) -> Tensor:
            nonlocal rolling_bitmask_lipschitz_safety, rolling_bitmask_constraint_violations
            (_, bitmask_pairs) = self.gradient_free_voronoi_walk(X=X_input, P=X, w=w,
                                    verts=verticies, generators=generators, lipschitz=lipschitz_constraint)
            rolling_bitmask_lipschitz_safety: dict[int, Tensor] = self.merge_rolling_pairs(rolling_bitmask_lipschitz_safety, bitmask_pairs)

            constraint_bitmask: dict[int, Tensor] = self.bucketed_violation_metrics(X=X_input, P=X, w=w, conf=conf_level, constraints=constraint_objects)
            rolling_bitmask_constraint_violations: dict[int, Tensor] = self.merge_rolling_pairs(rolling_bitmask_constraint_violations, constraint_bitmask)
            
            return self.X_topology_simplified(X_input, conf_level=conf_level, constraint_objects=constraint_objects)

        optimized: Tensor = self.de_sampler(X_reward_func, bounds=self.unit_bounds, batch_size=batch_size)
        grouped: dict[int, Tensor] = self.bucket_generator_pairs(self.extract_nearest_generators(X=optimized, P=X, w=w), optimized)

        lipschitz_probs: dict[int, float] = self.p_bitmask(rolling_bitmask_lipschitz_safety)
        constraint_violation_probs: dict[int, float] = self.p_bitmask(rolling_bitmask_constraint_violations)

        conditioned: dict[int, float] = self.condition_rolling_pairs((lipschitz_probs, constraint_violation_probs))
        (k_highest, k_lowest) = self.prob_ranking(prob=conditioned)

        # Ranking mechanism to chose safe sets, and expander sets, proposing a new candidate position based on probability metrics
        sel: Tensor = self.dist_select(candidate_pairs=grouped, k_safe=k_highest, k_unsafe=k_lowest)
        return sel

    # ==--=--== Topology Acquisition Methods ==--=--==

    def X_topology_simplified[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint]) -> Tensor:
        # smooth penalty mask to guarantee that this candidate is exploring in an uncertain point of the objective surrogate
        objective_uncertainty_mask: Tensor = self.objective_uncertainty_mask(X, conf_level=conf_level)
        
        # smooth penalty mask to guarantee that the x candidate is safe
        (safety_penalty_mask, safety_bitmask) = self.constraint_safe_mask(X, conf_level=conf_level, 
                                                                constraint_objects=constraint_objects, penalty_magnitude=1.0)

        penalty: Tensor = safety_penalty_mask
        # smooth reward mask to fill in the safe regions
        reward_mask: Tensor = self.surrogate.get_lcb(X, conf_level).squeeze(-1) #argmin the lcb
        return torch.where(safety_bitmask, reward_mask, penalty) - objective_uncertainty_mask

    # ==--=--== Estimators ==--=--==
    # 1. Monte-Carlo Estimators

    def mc_estimators[T_Constraints: Constraint](self, P: Tensor, w: Tensor, constraints: list[T_Constraints], conf: float, 
                                            bucket_metric_method: Optional[BucketMetricCallable], metric_extraction_method: Optional[MetricExtractionCallable], X: Optional[Tensor] = None, n: int = 10_000) -> dict[int, float]:
        r"""
            Monte-carlo based approach for generating candidate points, and then estimating via a method to apply on the outputted buckets.

            Inputs:
                P: Generator coordinates
                w: Voronoi distance offsets
                constraints: List of constraint objects
                conf: Confidence values for confidence bound calculations
                bucket_metric_method: Callable to compute on the already pre-generated generator: coordinate buckets (Defaults to `self.p_safety`)
                metric_extraction_method: Callable to compute bitmasks on specified requirements (Defaults to `self.violation_metric`)
                X: Input candidates, if not specified, then random initialization of size n is completed
                n: Initial sample size for the random monte-carlo approach
            Returns:
                dict[int, float]: (k: v) -> (unique generator index: computed probability value)
        """
        if not bucket_metric_method:
            bucket_metric_method = self.p_bitmask
        if not metric_extraction_method:
            metric_extraction_method = self.violation_metric

        dim: int = P.shape[-1]

        # Random point initialization
        X_samples: Tensor = self.randn((n * dim, dim), device=P.device, dtype=P.dtype) if not X else X

        # Grab the nearest verticies, bucketing each sample to a given vertex.
        generator_indicies: Tensor = self.extract_nearest_generators(X_samples, P, w)
        constraint_bitmask: Tensor = metric_extraction_method(X_samples, conf, constraints)

        # Filtering each constraint into their respective buckets
        grouped: dict[int, Tensor] = self.bucket_generator_pairs(generator_indicies, constraint_bitmask)
        return bucket_metric_method(grouped)

    def bucket_generator_pairs(self, X1: Tensor, X2: Tensor) -> dict[int, Tensor]:
        """
            Inputs:
                X1: (usually) Indicies of the respective generator points
                X2: The other tensor, which is perfectly aligned with X1
            Returns:
                dict: (k: v) -> (generator_idx: corresponding X2)
        """
        # Grabbing only the unique generators
        unique: Tensor = X1.unique()
        final: dict[int, Tensor] = {}

        # Remapping the ids of each unique generator, to the corresponding X2 (normally the constraint bitmask)
        for u in unique:
            final[int(u.item())] = X2[X1 == u]
        return final

    def p_bitmask(self, bucketed: dict[int, Tensor]) -> dict[int, float]:
        r"""
            MC estimator using simple bitmask counting:
                :math:`p(x) = \frac{1}{N} \sum_{i=1}^{N} f(x_i)`
            Inputs:
                bucketed: bucketed tensors by their corresponding unique generator point
            Returns:
                dict: (k: v) -> (generator_idx: `1/N \sum b_i \left( X \right)`)
        """
        final: dict[int, float] = {}
        for idx, bmask in bucketed.items():
            final[idx] = bmask.sum().item() / bmask.shape[0]
        return final

    # ==--=--== Other Safety Sampling Strategies ==--=--==
    # 2. Voronoi Walk Methods

    type GradfreeVorwalkResultType = tuple[Tensor, dict[int, Tensor]]

    def gradient_free_voronoi_walk(
            self, X: Tensor, P: Tensor, verts: Tensor, generators: Tensor, w: Tensor, lipschitz: LipschitzConstraints, 
            update_callable: Optional[UpdateCallable] = None, conf: float = 2.0, n_steps: int = 100, sensitivity: float = 0.05) -> GradfreeVorwalkResultType:
        """
            Inputs:
                X: Seed points of shape (n, dim)
                P: Generator points of shape (m, dim)
                verts: verticies
                generators: generators corresponding to each given vertex
                w: vorcand weighting
                lipschitz: lipschitz constraint object
                conf: confidence-level (specified in parameters)
                n_steps: maximum number of steps before the optimizer quits

            Returns:
                tuple[Tensor, final bitmasking pairs (int, Tensor)]: (n, dim) of final steps (placeholder)
        """
        if not update_callable:
            update_callable = self.lipschitz_checks

        # Initial step initialization
        X_step: Tensor = X # X.shape: (n, dim) != P.shape: (m, dim)
        pairs_final: dict[int, Tensor] = {}

        for _ in range(n_steps):
            # Grab (generator_index: (vertex, X_step) pairs)
            generator_indicies: Tensor = self.extract_nearest_generators(X_step, P=P, w=w, indicies=True)
            grouped_generators: dict[int, Tensor] = self.bucket_generator_pairs(X1=generator_indicies, X2=X_step)
            grouped_verticies: dict[int, Tensor] = self.bucket_generator_pairs(X1=generators, X2=verts)

            reached: bool = True
            X_buffer: list[Tensor] = []
            for (idx, X_g) in grouped_generators.items():
                if idx not in grouped_verticies:
                    raise ValueError(f"Generator index: {idx} not found in allowed verticies!")
                
                diff: Tensor = grouped_verticies[idx] - X_g
                diff_norm: Tensor = self.l2_normalization(diff)
                
                X_buffer.append(diff_norm)
                reached &= bool((diff_norm < sensitivity).all().item())

            # Over-write the original steps, and paint the voronoi landscape with a safety callable (for safety checks)
            X_step = torch.vstack(X_buffer)
            bitmask_pairs: dict[int, Tensor] = update_callable(X_step, P, w, lipschitz) # (k: v) -> (generator_index: ith_viol_bitmask)
            pairs_final = self.merge_rolling_pairs(pairs_final, bitmask_pairs)
            
            # Early escape sequence
            if reached:
                break

        return (X_step, pairs_final)

    def merge_rolling_pairs(self, p1: dict[int, Tensor], p2: dict[int, Tensor]) -> dict[int, Tensor]:
        """
            In-place operation.

            Inputs:
                p1: list pair 1 with (k: v) -> (generator_idx: Tensor of corresponding values)
                p1: list pair 2 with (k: v) -> (generator_idx: Tensor of corresponding values)
            Returns:
                dict[int, Tensor]: (k: v) -> (generator_idx: Tensor of corresponding values) new merged dictionary
        """
        common: set[int] = set(p1.keys()) & set(p2.keys())
        for k in common:
            if k not in p2 or k not in p1:
                continue
            p1[k] = torch.vstack((p1[k], p2[k]))
        return p1

    def condition_rolling_pairs(self, probs: tuple[dict[int, float], ...]) -> dict[int, float]:
        """
            Conditioning on each pair in the list of prob pairs.
            Inputs:
                probs: list of generator_idx: probability key-value pairs
            Outputs:
                dict[int, float]: a new multiplicative conditioned prob per provided idx
        """
        final: dict[int, float] = {}
        for p in probs:
            for (k, v) in p.items():
                if k in final:
                    final[k] *= v
                else:
                    final[k] = v
        return final
            
    def l2_normalization(self, X: Tensor) -> Tensor:
        """
            Rescales the input tensor X, ensuring that its Euclidian magnitude becomes one
            Inputs:
                X: Tensor of shape (n, X)
            Returns:
                X: Tensor of shape (n, X) where ||X|| == 1
        """
        return X / torch.sqrt(torch.sum(X**2, dim=1).unsqueeze(-1))
    
    # ==--=--== Safety guaranteeing methods ==--=--==

    def lipschitz_checks(self, X: Tensor, P: Tensor, w: Tensor, lipschitz: LipschitzConstraints) -> dict[int, Tensor]:
        violation: Tensor = lipschitz.get_violation_mask(X, P) # Shape: (n, dim), where n = n(X)
        nearest_generators: Tensor = self.extract_nearest_generators(X=X, P=P, w=w)
        return self.bucket_generator_pairs(X1=nearest_generators, X2=violation)
        
    # ==--=--== Metric extraction methods ==--=--==

    def bucketed_violation_metrics[T_Constraint: Constraint](self, X: Tensor, P: Tensor, w: Tensor, conf: float, constraints: list[T_Constraint]) -> dict[int, Tensor]:
        violation: Tensor = self.violation_metric(X, conf=conf, constraints=constraints) # Shape: (n, dim), where n = n(X)
        nearest_generators: Tensor = self.extract_nearest_generators(X=X, P=P, w=w)
        return self.bucket_generator_pairs(X1=nearest_generators, X2=violation)

    def violation_metric[T_Constraint: Constraint](self, X: Tensor, conf: float, constraints: list[T_Constraint]) -> Tensor:
        """

        """
        bitmask: Tensor = torch.ones((X.shape[0], ), device=X.device, dtype=torch.bool)
        for c in constraints:
            if not isinstance(c, SurrogateConstraint):
                continue
            bitmask &= (c.get_lcb(X, beta=conf) < 0).squeeze(-1)
        return bitmask

    # ==--=--== Random initialization methods ==--=--==

    def randn(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
        # I know this is a dumb wrapper of an already existing method at the moment, but I am keeping this here, so I can then easily write hot-swappable methods for Sobol sampling, and uniform grid sampling
        return torch.rand(shape, device=device, dtype=dtype)

    # ==--=--== Filters ==--=--==



    # ==--=--== Ranking methods ==--=--==

    def prob_ranking(self, prob: dict[int, float], k_highest: int = 1, k_lowest: int = 1) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """
            Top k, and lowest k ranking (can also implement min heap for O(n) t.c. vs O(nlogn))
            Inputs:
                prob: probability dictionary with (k: v) -> (generator_index: probability)
                k_highest: the number of highest elements to return
                k_highest: the number of lowest elements to return
            Returns:
                tuple[list[tuple[int, float]], list[tuple[int, float]]]: Returns two lists with paired (generator_index, probability) 1. top k highest, and 2. top k lowest elements
        """
        l: int = len(prob)
        paired: list[tuple[int, float]] = list(zip(prob.keys(), prob.values()))
        srt: list[tuple[int, float]] = sorted(paired, key=lambda x: x[1])
        return (srt[-min(k_highest, l):][::-1], srt[:min(k_lowest, l)])

    # ==--=--== Candidate selection methods ==--=--==

    def dist_select(self, candidate_pairs: dict[int, Tensor], k_safe: list[tuple[int, float]], k_unsafe: list[tuple[int, float]]) -> Tensor:
        k_safe_proposed: Tensor = self._merge_prob_pairs(candidate_pairs=candidate_pairs, prob_pairs=k_safe)
        k_unsafe_proposed: Tensor = self._merge_prob_pairs(candidate_pairs=candidate_pairs, prob_pairs=k_unsafe)

        dist: Tensor = torch.cdist(k_safe_proposed, k_unsafe_proposed).squeeze()
        return k_safe_proposed[torch.argmin(dist, dim=0)]

    def _merge_prob_pairs(self, candidate_pairs: dict[int, Tensor], prob_pairs: list[tuple[int, float]]) -> Tensor:
        k_safe_proposed_buffer: list[Tensor] = []
        for (idx, _) in prob_pairs:
            k_safe_proposed_buffer.append(candidate_pairs[idx])
        return torch.vstack(k_safe_proposed_buffer)

    # ==--=--== Weighting methods ==--=--==

    def get_weighting(self, P: Tensor, weighting_callable: Optional[WeightingCallable] = None) -> Tensor:
        w: Tensor = torch.zeros(P.shape[0], device=P.device)
        if weighting_callable:
            w: Tensor = weighting_callable(P)
        return w.T

    def pointwise_variance(self, X: Tensor) -> Tensor:
        return self.surrogate.get_ucb(X, self.state.convergence.confidence_level) - self.surrogate.get_lcb(X, self.state.convergence.confidence_level)

    # ==--=--== Voronoi-helper methods ==--=--==

    def extract_nearest_generators(self, X: Tensor, P: Tensor, w: Tensor, indicies: bool = True) -> Tensor:
        """
            Inputs:
                X: Input points
                P: Generator points (seeds for each vorcand)
            Returns:
                indicies, or generator points if the indicies flag is False
        """
        inds: Tensor = torch.argmin(torch.cdist(X, P) - w, dim=1) # Returns indicies
        return inds if indicies else P[inds]
    
    def get_vertex_generator_pairs(self, P: Tensor, w: Tensor, n: int = 1_000_000, tol: float = 1e-3) -> tuple[Tensor, Tensor]:  
        """
            Inputs:
                n: number of MC samples to estimate location of verticies
                P: Generator points (seeds for each vorcand)
                w: Weighting for boundary walls
                tol: Distance tolerance to detect verticies
            Returns:
                tuple[Verticies, Generator points]
        """
        dim: int = P.shape[1]
        # Scale by dim

        X_samples: Tensor = torch.rand((n * dim, dim), device=P.device, dtype=P.dtype)
        dist: Tensor = torch.cdist(X_samples, P) - w
        vertex_indicies, generator_indicies = torch.topk(dist, k=3, dim=1, largest=False)

        # Vertex detection bitmask
        vertex_bitmask: Tensor = (vertex_indicies[:, 2] - vertex_indicies[:, 0]) < tol

        # Vertex at edges detection bitmask
        edge_bitmask: Tensor = (vertex_indicies[:, 1] - vertex_indicies[:, 0]) < tol
        border_bitmask: Tensor = ((X_samples <= tol) | (X_samples >= 1.0 - tol)).any(dim=1) 
        border_vertex_bitmask: Tensor = edge_bitmask & border_bitmask

        # Merged bitmasking
        actual_verticies_bitmask: Tensor = vertex_bitmask | border_vertex_bitmask

        # Cleaning blotting of verticies
        cleaned_verticies_bitmask: Tensor = self._clean_vertex_blotting(actual_verticies_bitmask=actual_verticies_bitmask, 
                                                                        generator_indicies=generator_indicies)

        # Returns a tuple of [Verticies, Generator_points], paired together
        return (X_samples[cleaned_verticies_bitmask], generator_indicies[cleaned_verticies_bitmask])

    def _clean_vertex_blotting(self, actual_verticies_bitmask: Tensor, generator_indicies: Tensor) -> Tensor:
        """
            Inputs: 
                actual_verticies_bitmask: tensor boolean bitmask of the proposed vertices
                generator_indicies: tensor boolean bitmask of the corresponding indicies of generator points
            Outputs:
                Tensor[_bool]: boolean bitmask with no point clustering
        """
        _, device = actual_verticies_bitmask.dtype, actual_verticies_bitmask.device

        idx = actual_verticies_bitmask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return actual_verticies_bitmask

        triples = generator_indicies[idx].sort(dim=1).values
        _, inverse = torch.unique(triples, dim=0, return_inverse=True)

        rep = torch.empty(int(inverse.max()) + 1, dtype=torch.long, device=device)
        rep.scatter_(0, inverse, torch.arange(len(idx), device=device))

        clean_bitmask = torch.zeros_like(actual_verticies_bitmask)
        clean_bitmask[idx[rep]] = True
        return clean_bitmask

    # ==--=--== Plotting methods ==--=--==

    def plot_voronoi(self, P: Tensor, w: Tensor, res: int = 600) -> None:
        P, w = P.detach(), w.detach()
        xs = torch.linspace(0, 1, res, device=P.device, dtype=P.dtype)

        # slice through the cube at 0.5, varying only the first two dims
        grid = torch.full((res * res, P.shape[1]), 0.5, device=P.device, dtype=P.dtype)
        grid[:, :2] = torch.cartesian_prod(xs, xs)

        labels = self.extract_nearest_generators(grid, P=P, w=w).reshape(res, res)
        vertices, _ = self.get_vertex_generator_pairs(P=P, w=w)

        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

        ax.contour(xs.cpu(), xs.cpu(), labels.T.cpu(), levels=np.arange(len(P) - 1) + 0.5,
                colors="#1f2937", linewidths=2.2)
        ax.scatter(*vertices[:, :2].cpu().T, s=16, c="#e03131", marker="D", linewidths=0,
                alpha=0.6, zorder=2, label="vertices")
        ax.scatter(*P[:, :2].cpu().T, s=100, c="red", edgecolors="white", linewidths=1.8,
                zorder=3, label="generators")

        ax.set(xlim=(0.7, 1), ylim=(0.2, 0.5), aspect="equal", xlabel="$x_0$", ylabel="$x_1$",
            title=f"Weighted Voronoi diagram ({len(P)} generators)")
        ax.grid(color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

        fig.tight_layout()
        plt.show()

    def plot_fitted_normal(self, probs: dict[int, float]):
        x_data = np.array(list(probs.keys()))
        p_data = np.array(list(probs.values()))
        
        mean = np.sum(x_data * p_data)
        std = np.sqrt(np.sum(p_data * (x_data - mean)**2))
        x_smooth = np.linspace(mean - 3*std, mean + 3*std, 300)
        y_normal = norm.pdf(x_smooth, mean, std)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_smooth, y_normal, color='crimson', linewidth=2.5, 
                label=f'Fitted Normal ($\mu$={mean:.2f}, $\sigma$={std:.2f})')
        plt.fill_between(x_smooth, y_normal, color='crimson', alpha=0.15)
        
        plt.scatter(x_data, p_data, color='black', alpha=0.5, zorder=3, label='Actual Data Points')
        
        plt.title("Normal Distribution Fit from Probability Dict", fontsize=12, fontweight='bold')
        plt.xlabel("Outcome")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    # ==--=--== Helper methods ==--=--==

    @staticmethod
    def _argmin_rewards(X: Tensor, reward_method: Callable[[Tensor], Tensor]) -> Tensor:
        return X[torch.argmin(reward_method(X))]
    @staticmethod
    def _default_penalty_mask(X: Tensor) -> Tensor:
        N: int = X.shape[0]
        return torch.full((N, ), 0, dtype=X.dtype, device=X.device)
    @staticmethod
    def _default_bitmask(X: Tensor, true: bool = True) -> Tensor:
        N: int = X.shape[0]
        return torch.full((N, ), int(true), dtype=torch.bool, device=X.device)
    @staticmethod
    def _validate_result(X: Tensor) -> Tensor:
        if torch.isneginf(X).all():
            raise ValueError("No valid constraint evaluations were found!")
        return X

    def objective_uncertainty_mask(self, X: Tensor, conf_level: float, penalty_magnitude: float = 1.) -> Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(X=X)

        lcb: Tensor = self.surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        ucb: Tensor = self.surrogate.get_ucb(X=X, beta=conf_level).squeeze(-1) # (N_z, )
        violation: Tensor = torch.clamp(0. - (ucb-lcb), 0.)

        penalty_mask: Tensor = torch.maximum(penalty_mask, violation)
        
        return self._validate_result((penalty_mask) * penalty_magnitude) # D \ S_t
    
    def constraint_safe_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Returns: penalty_mask, safety_bitmask
        """
        penalty_mask: Tensor = self._default_penalty_mask(X=X)
        bitmask: Tensor = self._default_bitmask(X=X)

        for constraint_surrogate in constraint_objects:
            if not isinstance(constraint_surrogate, SurrogateConstraint):
                continue
            lcb: Tensor = constraint_surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1)
            this_bitmask: Tensor = (lcb >= 0.) # bitmask with AND operator
            bitmask &= this_bitmask

            violation: Tensor = torch.where(~this_bitmask, -lcb, torch.zeros_like(lcb)) # Guarantees safety
            penalty_mask: Tensor = torch.maximum(penalty_mask, violation) # stack on violations to ensure that any non-0 region becomes invalid

        return (self._validate_result((penalty_mask) * penalty_magnitude), bitmask)

    def constraint_uncertainty_mask[T_Constraint: Constraint](self, X: Tensor, conf_level: float, constraint_objects: list[T_Constraint], penalty_magnitude: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
        pessimistic_penalty_mask: Tensor = self._default_penalty_mask(X=X)
        optimistic_penalty_mask: Tensor = torch.clone(pessimistic_penalty_mask)

        single_uncertain: Tensor = self._default_bitmask(X, True)

        for constraint_surrogate in constraint_objects:
            if not isinstance(constraint_surrogate, SurrogateConstraint):
                continue
            lcb: Tensor = constraint_surrogate.get_lcb(X=X, beta=conf_level).squeeze(-1)
            ucb: Tensor = constraint_surrogate.get_ucb(X=X, beta=conf_level).squeeze(-1)

            this_uncertain: Tensor = (lcb < 0.) & (ucb > 0.)
            single_uncertain &= this_uncertain

            pessimistic_violation: Tensor = torch.where(~this_uncertain, lcb, torch.zeros_like(lcb)) # Guarantees safety
            optimistic_violation: Tensor = torch.where(~this_uncertain, -ucb, torch.zeros_like(ucb)) # Guarantees safety

            pessimistic_penalty_mask: Tensor = torch.maximum(pessimistic_penalty_mask, pessimistic_violation)
            optimistic_penalty_mask: Tensor = torch.maximum(optimistic_penalty_mask, optimistic_violation)

        return (self._validate_result((pessimistic_penalty_mask + optimistic_penalty_mask) * penalty_magnitude), single_uncertain)

    def get_lipschitz_bitmask(self, X: Tensor, Z: Tensor, constraints: list[tuple[Tensor, Tensor]], penalty_magnitude: float = 1.) -> Tensor:
        penalty_mask: Tensor = self._default_penalty_mask(Z) # (N_z, )
        eucl_distance: Tensor = torch.cdist(x1=X, x2=Z) # (N_x, N_z)

        for (L_i, u_i) in constraints:
            u_i_col: Tensor = u_i.reshape(-1, 1)
            lipschitz: Tensor = u_i_col - L_i * eucl_distance
            violation: Tensor = torch.clamp(-lipschitz.max(dim=0).values, 0.0)
            penalty_mask: Tensor = torch.maximum(penalty_mask, violation)
        return penalty_mask * penalty_magnitude

    def hypercube_bounds(self, X: Tensor, padding: float = 1.) -> Tensor:
        min_bounds: Tensor = torch.clamp(torch.amin(X, dim=0) - padding, 0.0, 1.0)
        max_bounds: Tensor = torch.clamp(torch.amax(X, dim=0) + padding, 0.0, 1.0)

        return torch.permute(torch.vstack((min_bounds, max_bounds)), (1, 0))

    def enforce_safety[T_Constraint: Constraint](self, X: Tensor, X_safe: Tensor, conf_level: float, 
                                                 constraint_objects: list[T_Constraint], penalty_magnitude: float) -> Tensor:
        (_, bitmask) = self.constraint_safe_mask(
                    X, 
                    conf_level=conf_level, 
                    constraint_objects=constraint_objects, 
                    penalty_magnitude=penalty_magnitude
                )
        safety: Tensor = X[bitmask]
        return self.safety_fallback(X_prop=X, X_safe=X_safe, beta=conf_level) if safety.shape[0] == 0 else safety

    def get_lcb_matrix[T_Constraint: Constraint](self, X: Tensor, beta: float, constraint_objects: list[T_Constraint]) -> Tensor:
        final: Tensor = torch.zeros_like(X, device=X.device, dtype=X.dtype)

        for constraint_surrogate in constraint_objects:
            if not isinstance(constraint_surrogate, SurrogateConstraint):
                continue
            lcb: Tensor = constraint_surrogate.get_lcb(X=X, beta=beta)
            final: Tensor = torch.hstack((final, lcb))
        return final


    def safety_fallback(self, X_prop: Tensor, X_safe: Tensor, beta: float) -> Tensor:
        (X_safe_lcb, X_safe_idx) = torch.min(self.surrogate.get_lcb(X_safe, beta=beta))
        (X_proposed_lcb, X_prop_idx) = torch.min(self.surrogate.get_lcb(X_prop, beta=beta))

        gamma: float = (X_safe_lcb / (X_safe_lcb - X_proposed_lcb)).item()

        X_safe_single: Tensor = X_safe[X_safe_idx]
        X_prop_single: Tensor = X_prop[X_prop_idx]

        return X_safe_single + gamma * (X_prop_single - X_safe_single)

    def dist_selection(self, X: Tensor, Z: Tensor) -> Tensor:
        dist: Tensor = torch.cdist(x1=X, x2=Z.unsqueeze(0)).squeeze()   
        return X[torch.argmin(dist, dim=0)]

    def recalculate_magnitude(self, X: Tensor) -> float:
        (_, std) = self.surrogate.get_properties(X=X)
        return torch.sqrt(std).min().item()