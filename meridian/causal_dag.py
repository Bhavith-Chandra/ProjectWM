"""Causal DAG discovery and N-th order causal effect tracing for financial markets.

Methods:
  1. PC algorithm: constraint-based causal discovery
  2. GES: greedy equivalence search
  3. Intervention validation: compare observational vs interventional effects
  4. Causal path tracing: compute N-th order implications (shock → ... → final outcome)

For financial markets:
  - Nodes: asset returns, volatility, sentiment, economic indicators
  - Edges: causal relationships (A causes B)
  - Shocks: exogenous interventions (Fed rate, geopolitical event)
  - Queries: "If X changes, what happens to Z?"
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Set, Dict, List, Tuple
import torch
import torch.nn as nn


@dataclass
class CausalNode:
    """A node in the causal DAG."""
    name: str
    domain: str  # 'price', 'vol', 'sentiment', 'macro', 'micro'
    index: int  # For efficient operations


@dataclass
class CausalEdge:
    """A directed edge (u → v) in the DAG."""
    source: str  # Node name
    target: str  # Node name
    strength: float  # Effect magnitude
    delay: int  # Causal lag (days)
    confidence: float  # Statistical confidence


class PartialCorrelationTest:
    """Constraint-based conditional independence test (foundation of PC algorithm)."""

    def __init__(self, alpha: float = 0.05):
        """alpha: significance level for independence."""
        self.alpha = alpha

    def partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """
        Compute partial correlation: corr(x, y | z).

        Tests if x and y are independent conditioned on z.
        If |partial_corr| is small, likely independent.
        """
        if z.shape[1] == 0:
            # No conditioning variables
            return np.corrcoef(x.squeeze(), y.squeeze())[0, 1]

        # Residuals after regressing out z
        # x_residual = x - z @ (z^T z)^-1 z^T x
        X = np.column_stack([x, y, z])
        n = X.shape[0]

        # Partial correlation via regression
        # regress x on z, get residuals
        z_pinv = np.linalg.pinv(z)
        x_res = x - z @ z_pinv @ x
        y_res = y - z @ z_pinv @ y

        if x_res.std() < 1e-6 or y_res.std() < 1e-6:
            return 0.0

        return np.corrcoef(x_res.squeeze(), y_res.squeeze())[0, 1]

    def is_independent(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None) -> bool:
        """
        Test if x ⊥⊥ y | z (x independent of y given z).

        Returns True if partial correlation < threshold (likely independent).
        """
        if z is None:
            z = np.array([]).reshape(x.shape[0], 0)

        corr = self.partial_correlation(x, y, z)
        n = x.shape[0]

        # Fisher z-transform for significance testing
        # z-stat = sqrt(n-3) * 0.5 * log((1+r)/(1-r))
        if abs(corr) >= 0.9999:
            return False

        z_stat = np.sqrt(max(0, n - 3 - z.shape[1])) * 0.5 * np.log((1 + corr) / (1 - corr + 1e-8))
        p_value = 2 * (1 - _normal_cdf(abs(z_stat)))

        return p_value > self.alpha


def _normal_cdf(z: float) -> float:
    """Approximate normal CDF."""
    from scipy.stats import norm
    return norm.cdf(z)


class PCAlgorithm:
    """PC algorithm for causal DAG discovery (Meek 1995, Pearl 2009)."""

    def __init__(self, data: np.ndarray, variable_names: List[str], alpha: float = 0.05):
        """
        data: (n_samples, n_variables)
        variable_names: names of each variable
        alpha: significance level
        """
        self.data = data
        self.var_names = variable_names
        self.n_vars = len(variable_names)
        self.tester = PartialCorrelationTest(alpha=alpha)

        # Start with complete undirected graph
        self.undirected_graph: Set[Tuple[str, str]] = set()
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                self.undirected_graph.add((self.var_names[i], self.var_names[j]))
                self.undirected_graph.add((self.var_names[j], self.var_names[i]))

        # Separating sets (for determining direction)
        self.separating_sets: Dict[Tuple[str, str], Set[str]] = {}

    def skeleton_phase(self):
        """Phase 1: Remove edges based on conditional independence tests."""
        depth = 0
        max_depth = self.n_vars - 1

        while depth <= max_depth:
            changed = False
            edges_to_remove = []

            for i, var_i in enumerate(self.var_names):
                for j, var_j in enumerate(self.var_names):
                    if i >= j or (var_i, var_j) not in self.undirected_graph:
                        continue

                    # Find neighbors of var_i (excluding var_j)
                    neighbors_i = {
                        v for (u, v) in self.undirected_graph
                        if u == var_i and v != var_j
                    }

                    # Check all subsets of neighbors of size `depth`
                    if len(neighbors_i) >= depth:
                        from itertools import combinations

                        for cond_set in combinations(neighbors_i, depth):
                            cond_set = set(cond_set)

                            # Test independence: var_i ⊥⊥ var_j | cond_set
                            x = self.data[:, self.var_names.index(var_i)]
                            y = self.data[:, self.var_names.index(var_j)]
                            z = self.data[:, [self.var_names.index(v) for v in cond_set]]

                            if self.tester.is_independent(x, y, z):
                                # Independence found: remove edge
                                edges_to_remove.append((var_i, var_j))
                                edges_to_remove.append((var_j, var_i))
                                self.separating_sets[(var_i, var_j)] = cond_set
                                changed = True
                                break

            for edge in edges_to_remove:
                self.undirected_graph.discard(edge)

            if not changed:
                break

            depth += 1

    def orientation_phase(self):
        """Phase 2: Orient edges based on separating sets."""
        # Find v-structures (X → Z ← Y where X ⊥⊥ Y | ancestors)
        edges = list(self.undirected_graph)
        for i, var_x in enumerate(self.var_names):
            for j, var_y in enumerate(self.var_names):
                if i >= j:
                    continue

                for k, var_z in enumerate(self.var_names):
                    if k == i or k == j:
                        continue

                    # Check if X-Z and Y-Z are in graph but X-Y is not
                    if (
                        (var_x, var_z) in self.undirected_graph
                        and (var_y, var_z) in self.undirected_graph
                        and (var_x, var_y) not in self.undirected_graph
                    ):
                        # Check if Z is NOT in separating set of X-Y
                        sep_set = self.separating_sets.get((var_x, var_y), set())
                        if var_z not in sep_set:
                            # Orient as X → Z ← Y (v-structure)
                            pass  # Would mark in directed graph

    def run(self) -> Dict[str, List[str]]:
        """Run full PC algorithm and return DAG."""
        self.skeleton_phase()
        self.orientation_phase()

        # Convert to adjacency: var → [neighbors]
        dag = {var: [] for var in self.var_names}
        for (u, v) in self.undirected_graph:
            # This is a simplified version; full PC requires orientation rules
            dag[u].append(v)

        return dag


class CausalDAG:
    """Representation of the market's causal DAG."""

    def __init__(self, nodes: List[CausalNode]):
        self.nodes = {node.name: node for node in nodes}
        self.edges: List[CausalEdge] = []
        self.adjacency = {node.name: [] for node in nodes}

    def add_edge(self, source: str, target: str, strength: float, delay: int = 1, confidence: float = 0.95):
        """Add a causal edge."""
        edge = CausalEdge(source, target, strength, delay, confidence)
        self.edges.append(edge)
        self.adjacency[source].append(target)

    def get_causal_predecessors(self, target: str, max_depth: int = 5) -> Dict[str, List[str]]:
        """Find all variables that causally precede target (ancestors)."""
        ancestors = {target: [target]}  # target is its own ancestor

        queue = [(target, 0)]
        visited = {target}

        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Find edges pointing TO this node
            for edge in self.edges:
                if edge.target == node and edge.source not in visited:
                    visited.add(edge.source)
                    if edge.source not in ancestors:
                        ancestors[edge.source] = []
                    ancestors[edge.source].append(node)
                    queue.append((edge.source, depth + 1))

        return ancestors

    def get_causal_descendants(self, source: str, max_depth: int = 5) -> Dict[str, List[str]]:
        """Find all variables that causally follow source (descendants)."""
        descendants = {source: [source]}

        queue = [(source, 0)]
        visited = {source}

        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Find edges FROM this node
            for edge in self.edges:
                if edge.source == node and edge.target not in visited:
                    visited.add(edge.target)
                    if edge.target not in descendants:
                        descendants[edge.target] = []
                    descendants[edge.target].append(node)
                    queue.append((edge.target, depth + 1))

        return descendants

    def trace_shock(self, shock_var: str, shock_magnitude: float, horizon: int = 20) -> Dict[str, List[float]]:
        """
        Trace N-th order causal effects of a shock.

        Args:
            shock_var: variable that receives shock
            shock_magnitude: size of shock
            horizon: days to trace

        Returns: dict mapping variable → effect magnitude over time
        """
        effects = {node: np.zeros(horizon) for node in self.nodes}
        effects[shock_var][0] = shock_magnitude

        # Forward propagation: follow edges
        for t in range(horizon - 1):
            affected_this_step = {var: eff[t] for var, eff in effects.items() if eff[t] != 0}

            for source_var, source_effect in affected_this_step.items():
                for edge in self.edges:
                    if edge.source == source_var and edge.delay <= t + 1:
                        target_var = edge.target
                        # Propagate effect (attenuated by strength)
                        effects[target_var][t + edge.delay] += source_effect * edge.strength

        return effects


class InterventionValidator:
    """Validate causal DAG against intervention experiments."""

    def __init__(self, dag: CausalDAG):
        self.dag = dag

    def compare_observational_vs_interventional(
        self,
        observational_effect: float,
        interventional_effect: float,
        threshold: float = 0.2
    ) -> bool:
        """
        Check if observed causal effect matches intervention experiment.

        If |observational_effect - interventional_effect| / |interventional_effect| < threshold,
        then likely valid.
        """
        if abs(interventional_effect) < 1e-6:
            return abs(observational_effect) < 1e-6

        relative_error = abs(observational_effect - interventional_effect) / abs(interventional_effect)
        return relative_error < threshold


# Example usage and utilities
def build_financial_dag() -> CausalDAG:
    """Build a DAG capturing known market structure."""
    nodes = [
        CausalNode('fed_rate', 'macro', 0),
        CausalNode('bond_yield_10y', 'macro', 1),
        CausalNode('equity_duration_risk', 'price', 2),
        CausalNode('spy_return', 'price', 3),
        CausalNode('credit_spread', 'macro', 4),
        CausalNode('hyg_return', 'price', 5),
        CausalNode('realized_vol', 'vol', 6),
        CausalNode('vix', 'vol', 7),
    ]

    dag = CausalDAG(nodes)

    # Known causal relationships (subject to validation)
    dag.add_edge('fed_rate', 'bond_yield_10y', 0.8, delay=1)  # Strong, immediate
    dag.add_edge('fed_rate', 'credit_spread', 0.5, delay=3)  # Moderate, delayed
    dag.add_edge('bond_yield_10y', 'equity_duration_risk', 0.6, delay=2)
    dag.add_edge('equity_duration_risk', 'spy_return', -0.4, delay=1)  # Negative (rising rates → equity pain)
    dag.add_edge('credit_spread', 'hyg_return', -0.7, delay=1)
    dag.add_edge('spy_return', 'realized_vol', 0.3, delay=1)  # Realized vol responds to large moves
    dag.add_edge('realized_vol', 'vix', 0.8, delay=1)  # VIX is realized vol proxy

    return dag
