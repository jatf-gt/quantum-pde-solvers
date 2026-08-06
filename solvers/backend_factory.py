"""
Centralised factory for Qiskit Aer simulation backends, providing
transparent GPU/CPU dispatch based on runtime environment detection.

Mathematical context
--------------------
Statevector simulation of an n-qubit circuit requires storing and
manipulating a complex vector of dimension 2ⁿ, with each gate
application constituting a dense matrix–vector product of cost
O(2ⁿ). For the QSVT circuits encountered in this project
(n_total = n_data + 1, depth ~ κ/ε), the dominant cost is:

    T_sim ∝ depth × 2^n_total

At N=8 (n_total=4, depth≈6,479) this is tractable on CPU (~222 s).
At N=16 (n_total=5, depth≈44,567) CPU simulation requires several
hours; GPU acceleration via NVIDIA cuStateVec reduces this by a
factor of 5–50× depending on circuit depth and qubit count.

GPU support requires the `qiskit-aer-gpu` package (CUDA 12) in
place of the standard `qiskit-aer`. On CX3 Phase 2, the L40S
(48 GB GDDR6, CUDA compute capability 8.9) and A100 (40 GB,
capability 8.0) nodes are available via the gpu72 queue.

References
----------
NVIDIA cuStateVec: https://docs.nvidia.com/cuda/cuquantum/latest/
Qiskit Aer GPU:    https://pypi.org/project/qiskit-aer-gpu/
Imperial CX3 GPU:  https://icl-rcs-user-guide.readthedocs.io/
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# -- Public Interface ---------------------------------------------------------


def get_aer_backend(
    prefer_gpu: bool = True,
    custatevec: bool = True,
    precision: str = "double",
    max_parallel_threads: int = 0,
) -> "AerSimulator":  # noqa: F821  (type hint only; import deferred below)
    """
    Construct and return an ``AerSimulator`` statevector backend, selecting
    GPU acceleration when available and requested.

    The function attempts GPU initialisation first; if the GPU device is
    unavailable (no CUDA driver, no ``qiskit-aer-gpu`` installation, or
    ``prefer_gpu=False``), it falls back to a CPU backend configured for
    maximum OpenMP thread utilisation.

    GPU selection logic
    -------------------
    1. ``prefer_gpu=False``  → CPU backend unconditionally.
    2. ``CUDA_VISIBLE_DEVICES`` not set or empty → CPU fallback (no GPU
       allocated by the PBS scheduler).
    3. ``AerSimulator`` with ``device='GPU'`` raises ``AerError`` → CPU
       fallback with a warning.
    4. Otherwise → GPU backend with optional cuStateVec acceleration.

    cuStateVec is beneficial for circuits with n_qubits ≥ 15 and provides
    the largest speedup for very deep circuits (depth ≫ 1000), which is
    precisely the QSVT regime encountered in this project.

    Parameters
    ----------
    prefer_gpu : bool
        If ``True``, attempt GPU initialisation before falling back to CPU.
        Set to ``False`` to force CPU execution (e.g., for debugging or
        when running on a non-GPU node).
    custatevec : bool
        If ``True`` and a GPU backend is successfully initialised, enable
        NVIDIA cuStateVec acceleration via the cuQuantum library.
        Requires ``qiskit-aer-gpu`` built against CUDA 12.
    precision : str
        Floating-point precision for statevector storage and arithmetic.
        ``'double'`` (64-bit complex) is required for the numerical
        accuracy targets of this project; ``'single'`` halves memory
        consumption but introduces ~1e-7 rounding errors.
    max_parallel_threads : int
        Maximum number of OpenMP threads for CPU simulation. ``0`` sets
        this automatically to the number of available CPU cores, which is
        the correct choice on a dedicated HPC node.

    Returns
    -------
    AerSimulator
        Configured Qiskit Aer statevector simulator instance.

    Raises
    ------
    ImportError
        If ``qiskit_aer`` is not installed in the active environment.
    """
    # Deferred import: qiskit_aer is a heavy dependency not required at
    # module import time; deferring avoids circular dependency issues in
    # test environments that mock the backend.
    from qiskit_aer import AerSimulator
    from qiskit_aer.backends.aerbackend import AerError

    # -- GPU path -------------------------------------------------------------
    if prefer_gpu:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            log.debug(
                "GPU backend requested but CUDA_VISIBLE_DEVICES is unset "
                "or empty; this indicates no GPU was allocated by the PBS "
                "scheduler. Falling back to CPU backend."
            )
        else:
            try:
                backend = AerSimulator(
                    method="statevector",
                    device="GPU",
                    precision=precision,
                    cuStateVec_enable=custatevec,
                )
                # Probe the backend to confirm GPU initialisation succeeded.
                available = backend.available_devices()
                if "GPU" not in available:
                    raise AerError(
                        "GPU device not listed in available_devices(); "
                        "qiskit-aer-gpu may not be installed."
                    )
                log.info(
                    "GPU backend initialised successfully. "
                    "cuStateVec=%s, precision=%s, CUDA_VISIBLE_DEVICES=%s.",
                    custatevec,
                    precision,
                    cuda_visible,
                )
                return backend

            except (AerError, Exception) as exc:
                log.warning(
                    "GPU backend initialisation failed (%s). "
                    "Falling back to CPU backend.",
                    exc,
                )

    # -- CPU fallback ---------------------------------------------------------
    backend = AerSimulator(
        method="statevector",
        device="CPU",
        precision=precision,
        max_parallel_threads=max_parallel_threads,
        # Enable OpenMP parallelisation for matrix multiplication when
        # n_qubits exceeds this threshold. The default of 14 is appropriate
        # for the circuit sizes encountered in this project (n_total ≤ 9).
        statevector_parallel_threshold=8,
    )
    log.debug(
        "CPU backend initialised. precision=%s, max_parallel_threads=%s.",
        precision,
        max_parallel_threads if max_parallel_threads > 0 else "auto",
    )
    return backend


def log_backend_info(backend: "AerSimulator") -> None:  # noqa: F821
    """
    Emit a structured diagnostic log entry describing the active backend.

    Parameters
    ----------
    backend : AerSimulator
        The backend instance returned by :func:`get_aer_backend`.
    """
    try:
        options = backend.options
        device   = getattr(options, "device",    "unknown")
        prec     = getattr(options, "precision", "unknown")
        csv_flag = getattr(options, "cuStateVec_enable", False)
        log.info(
            "Active Aer backend — device: %-4s | precision: %-6s | "
            "cuStateVec: %s",
            device, prec, csv_flag,
        )
    except Exception:
        log.info("Active Aer backend — (options unavailable)")