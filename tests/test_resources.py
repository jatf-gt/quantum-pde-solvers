"""
Tests for core.resources.

Two things need protecting here, and they are different in kind:

1.  The transpilation mechanics (basis gates used, two-qubit counting,
    depth reporting) should behave the way the docstrings say.
2.  The *safe-upper-bound* property of the composed estimate is an
    empirical finding, not a mathematical guarantee, and empirical
    findings drift silently if nothing keeps checking them. The
    ``TestComposabilityBound`` class below re-runs the validation for
    every (N, degree) shape the rest of the codebase actually relies on,
    so a future Qiskit version that changes transpiler heuristics enough
    to break the bound is caught here rather than in a thesis figure.
"""
from __future__ import annotations

import pytest

qiskit = pytest.importorskip("qiskit")

from core.resources import (                                       # noqa: E402
    HERON_R2_BASIS_GATES,
    HERON_R2_TWO_QUBIT_GATE_BUDGET,
    ResourceReport,
    block_encoding_unit_cost,
    qsvt_resource_estimate,
    state_prep_cost,
    transpile_report,
    validate_composability,
)


# -- Hardware target sanity ----------------------------------------------------

class TestHardwareTargetConstants:
    """
    These constants were confirmed against current IBM documentation and
    independent hardware papers (August 2026). Pin them so a future edit
    doesn't silently swap in the Eagle-generation gate set (ECR) or an
    outdated budget figure without the change being visible in a diff.
    """

    def test_basis_gates_use_cz_not_ecr(self):
        """
        Validates that the hardware target correctly specifies CZ as the native two-qubit gate.
        Ensures resource estimates are predicated on the Heron architecture rather than the legacy Eagle generation.
        """
        # Heron r1/r2 native two-qubit gate is CZ. ECR belongs to the
        # earlier Eagle generation; conflating the two silently mis-prices
        # every resource estimate in this module.
        assert "cz" in HERON_R2_BASIS_GATES
        assert "ecr" not in HERON_R2_BASIS_GATES

    def test_basis_gates_are_single_and_two_qubit_only(self):
        """
        Confirms that the specified basis gates consist strictly of single- and two-qubit operations.
        """
        assert set(HERON_R2_BASIS_GATES) == {"rz", "sx", "x", "cz"}

    def test_budget_is_positive_and_documented_order_of_magnitude(self):
        """
        Ensures that the defined two-qubit gate budget aligns with established hardware constraints.
        Prevents silent degradation of feasibility thresholds by enforcing expected magnitudes.
        """
        # IBM's reported Heron r2 circuit capacity is on the order of
        # thousands of two-qubit gates, not hundreds or millions.
        assert 1_000 <= HERON_R2_TWO_QUBIT_GATE_BUDGET <= 20_000


# -- Transpilation mechanics ---------------------------------------------------

@pytest.mark.quantum
class TestTranspileReport:

    def test_reports_only_target_basis_gates(self):
        """
        Validates that the transpilation process correctly decomposes circuits strictly into the designated basis gates.
        """
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        report = transpile_report(qc)
        assert set(report.gate_counts) <= set(HERON_R2_BASIS_GATES)

    def test_post_depth_is_positive_for_nontrivial_circuit(self):
        """
        Confirms that non-trivial circuits yield a positive post-transpilation depth and two-qubit gate count.
        """
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        report = transpile_report(qc)
        assert report.post_depth > 0
        assert report.two_qubit_count >= 1

    def test_identity_circuit_needs_no_two_qubit_gates(self):
        """
        Validates that an identity circuit requires zero two-qubit gates post-transpilation.
        Ensures the transpiler correctly optimises trivial operations.
        """
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3)  # no gates at all
        report = transpile_report(qc)
        assert report.two_qubit_count == 0


# -- Block-encoding unit cost --------------------------------------------------

@pytest.mark.quantum
class TestBlockEncodingUnitCost:

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_unit_cost_grows_with_N(self, N):
        """
        Confirms that the resource cost for block encoding increases as the problem size scales.
        """
        import numpy as np
        report = block_encoding_unit_cost(N)
        assert report.two_qubit_count > 0
        # n data qubits + 1 block-encoding ancilla
        assert report.n_qubits == int(np.log2(N)) + 1

    def test_unit_cost_increases_monotonically(self):
        """
        Validates that the block encoding unit cost exhibits monotonic growth with respect to problem size.
        """
        costs = [block_encoding_unit_cost(N).two_qubit_count for N in (4, 8, 16)]
        assert costs == sorted(costs)
        assert costs[0] < costs[-1]


# -- The empirical safe-upper-bound property -----------------------------------

@pytest.mark.quantum
class TestComposabilityBound:
    """
    Re-derives the finding stated in core.resources' module docstring: the
    composed estimate (unit_cost * degree + prep_cost) is not exact, but has
    never been observed to fall below a direct full-circuit transpilation.

    If this test starts failing, the module docstring's overshoot-margin
    numbers are stale and every FeasibilityReport produced since is suspect
    until re-validated.
    """

    @pytest.mark.parametrize("N,degree", [
        (4, 5), (4, 11), (4, 21),
        (8, 5), (8, 11), (8, 21),
        (16, 5), (16, 11),
    ])
    def test_composed_is_a_safe_upper_bound(self, N, degree):
        """
        Confirms that the composed resource estimate provides a rigorous upper bound compared to direct full-circuit transpilation.
        Ensures the validity of subsequent feasibility reports by preventing underestimation of gate counts.
        """
        result = validate_composability(N, degree)
        assert result["is_safe_upper_bound"], (
            f"composed estimate ({result['composed_two_qubit_count']}) fell "
            f"below the directly-transpiled count "
            f"({result['direct_two_qubit_count']}) at N={N}, degree={degree} "
            f"-- the safe-upper-bound property has broken, likely due to a "
            f"transpiler version change. Every FeasibilityReport should be "
            f"treated as unverified until this is understood."
        )

    def test_overshoot_shrinks_as_N_grows(self):
        """
        Validates that the conservatism of the upper bound diminishes as problem size increases.
        Confirms the scaling assumption that estimates for larger systems are comparatively tight.
        """
        # The module docstring claims the margin tightens with N; confirm it
        # rather than merely asserting it, since this is the basis for
        # treating N>=8 estimates as tight and N=4 estimates as loose.
        overshoot_4  = validate_composability(4, 11)["overshoot_fraction"]
        overshoot_16 = validate_composability(16, 11)["overshoot_fraction"]
        assert overshoot_16 < overshoot_4


# -- End-to-end feasibility estimate -------------------------------------------

@pytest.mark.quantum
class TestQSVTResourceEstimate:

    def test_small_problem_is_feasible(self):
        """
        Validates that constrained problem instances are appropriately classified as feasible within the designated hardware budget.
        """
        # N=4, production degree 63 (per QSVTConfig1D's own recommended
        # max_degree table): should fit comfortably inside the Heron r2
        # two-qubit budget.
        report = qsvt_resource_estimate(N=4, degree=63, kappa=9.0)
        assert report.feasible
        assert report.total_two_qubit_count < HERON_R2_TWO_QUBIT_GATE_BUDGET

    def test_large_problem_is_not_feasible(self):
        """
        Confirms that problem instances exceeding hardware constraints are correctly identified as infeasible.
        """
        # N=32, production degree 511: expected to badly exceed the budget --
        # this is the "explicitly not feasible" result from the earlier
        # hardware-scoping discussion, now backed by a transpiled number.
        report = qsvt_resource_estimate(N=32, degree=511, kappa=441.0)
        assert not report.feasible
        assert report.total_two_qubit_count > 10 * HERON_R2_TWO_QUBIT_GATE_BUDGET

    def test_feasible_implies_bound_holds(self):
        """
        Ensures that a classification of feasibility implies the composed estimate strictly bounds the direct transpilation count.
        """
        # A "feasible" verdict should be trustworthy precisely because the
        # estimate is a safe upper bound: if composed <= budget, then the
        # (smaller or equal) directly-transpiled circuit is also <= budget.
        report = qsvt_resource_estimate(N=4, degree=63, kappa=9.0)
        assert report.feasible
        direct = validate_composability(4, 63)
        assert direct["direct_two_qubit_count"] <= report.total_two_qubit_count