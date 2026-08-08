import time
import numpy as np

# We'll use the existing inner solver registry
from solvers.outer.inner import get_inner

# ==============================================================================
# 4th Order Operators & Assembly
# ==============================================================================

def build_strip_matrix_4th(N: int, dx: float) -> np.ndarray:
    """4th order implicit strip matrix, identical to the debug_2d_4th logic."""
    A = np.zeros((N, N))
    diag = -30.0 * np.ones(N)
    sub1 = 16.0 * np.ones(N - 1)
    sub2 = -1.0 * np.ones(N - 2)
    
    A += np.diag(diag, k=0)
    A += np.diag(sub1, k=1) + np.diag(sub1, k=-1)
    A += np.diag(sub2, k=2) + np.diag(sub2, k=-2)
    
    if N > 1:
        A[0, 1] += -1.0
        A[-1, -2] += -1.0
    return A

def _build_rhs_strip(j: int, phi: np.ndarray, f_vals: np.ndarray, dx: float, dy: float,
                     bc_x0, bc_x1, bc_y0, bc_y1) -> np.ndarray:
    kappa_aniso = (dx / dy)**2
    N = phi.shape[0]
    b = 12.0 * dx**2 * f_vals[:, j].copy()

    # X-boundary corrections (implicit 4th order direction)
    ax0 = bc_x0[j] if isinstance(bc_x0, np.ndarray) else bc_x0
    ax1 = bc_x1[j] if isinstance(bc_x1, np.ndarray) else bc_x1
    b[0] -= 18.0 * ax0
    if N > 1:
        b[1] += ax0
    b[-1] -= 18.0 * ax1
    if N > 1:
        b[-2] += ax1

    # Y-boundary corrections (explicit 2nd order direction)
    if j > 0:
        b -= 12.0 * kappa_aniso * phi[:, j - 1]
    else:
        ay0 = bc_y0 if isinstance(bc_y0, np.ndarray) else np.full(N, bc_y0)
        b -= 12.0 * kappa_aniso * ay0

    if j < N - 1:
        b -= 12.0 * kappa_aniso * phi[:, j + 1]
    else:
        ay1 = bc_y1 if isinstance(bc_y1, np.ndarray) else np.full(N, bc_y1)
        b -= 12.0 * kappa_aniso * ay1

    return b

def compute_residual_2d_4th(phi: np.ndarray, f_vals: np.ndarray, dx: float, dy: float,
                            bc_x0, bc_x1, bc_y0, bc_y1, A_strip) -> np.ndarray:
    N = phi.shape[0]
    res = np.zeros_like(phi)
    for j in range(N):
        b_j = _build_rhs_strip(j, phi, f_vals, dx, dy, bc_x0, bc_x1, bc_y0, bc_y1)
        res[:, j] = b_j - A_strip @ phi[:, j]
    return res

def relax_2d_4th(phi: np.ndarray, f_vals: np.ndarray, dx: float, dy: float,
                 bc_x0, bc_x1, bc_y0, bc_y1, A_strip, solve_strip, nu: int) -> np.ndarray:
    """Line Gauss-Seidel smoothing sweep."""
    N = phi.shape[0]
    for _ in range(nu):
        for j in range(N):
            b_j = _build_rhs_strip(j, phi, f_vals, dx, dy, bc_x0, bc_x1, bc_y0, bc_y1)
            phi[:, j] = solve_strip(A_strip, b_j)
    return phi

# ==============================================================================
# Multigrid Transfer Operators (from solvers/outer/multigrid.py)
# ==============================================================================

def interpolation_1d(n_fine: int, n_coarse: int, L: float) -> np.ndarray:
    x_f = np.arange(1, n_fine + 1) * (L / (n_fine + 1))
    x_c = np.arange(1, n_coarse + 1) * (L / (n_coarse + 1))
    x_ext = np.concatenate(([0.0], x_c, [L]))

    P = np.zeros((n_fine, n_coarse))
    for i, x in enumerate(x_f):
        k = int(np.clip(np.searchsorted(x_ext, x) - 1, 0, len(x_ext) - 2))
        t = (x - x_ext[k]) / (x_ext[k + 1] - x_ext[k])
        for idx, w in ((k, 1.0 - t), (k + 1, t)):
            if 1 <= idx <= n_coarse:
                P[i, idx - 1] += w
    return P

def restriction_from(P: np.ndarray) -> np.ndarray:
    R = P.T.copy()
    s = R.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return R / s

def _apply_axis_ops(mats: list[np.ndarray], arr: np.ndarray) -> np.ndarray:
    out = arr
    for ax, M in enumerate(mats):
        out = np.moveaxis(np.tensordot(M, out, axes=([1], [ax])), 0, ax)
    return out

# ==============================================================================
# Multigrid FMG 4th Order Implementation
# ==============================================================================

class Level4th:
    def __init__(self, N, dx, dy, f_vals, bc_x0, bc_x1, bc_y0, bc_y1, Lx, Ly):
        self.N = N
        self.dx = dx
        self.dy = dy
        self.f_vals = f_vals
        self.bc_x0 = bc_x0
        self.bc_x1 = bc_x1
        self.bc_y0 = bc_y0
        self.bc_y1 = bc_y1
        self.Lx = Lx
        self.Ly = Ly
        
        kappa_aniso = (dx / dy)**2
        self.A_strip = build_strip_matrix_4th(N, dx) - 24.0 * kappa_aniso * np.eye(N)
        
        self.N_coarse = N // 2
        if self.N_coarse >= 4:
            self.Px = interpolation_1d(N, self.N_coarse, Lx)
            self.Py = interpolation_1d(N, self.N_coarse, Ly)
            self.Rx = restriction_from(self.Px)
            self.Ry = restriction_from(self.Py)
        else:
            self.Px = self.Py = self.Rx = self.Ry = None

    def restrict_field(self, arr):
        return _apply_axis_ops([self.Rx, self.Ry], arr)

    def prolong_field(self, arr):
        return _apply_axis_ops([self.Px, self.Py], arr)

def build_hierarchy_2d_4th(N_fine, dx_fine, dy_fine, f_vals, bc_x0, bc_x1, bc_y0, bc_y1):
    Lx = dx_fine * (N_fine + 1)
    Ly = dy_fine * (N_fine + 1)
    
    levels = []
    lvl = Level4th(N_fine, dx_fine, dy_fine, f_vals, bc_x0, bc_x1, bc_y0, bc_y1, Lx, Ly)
    levels.append(lvl)
    
    while lvl.N_coarse >= 4:
        f_coarse = lvl.restrict_field(lvl.f_vals)
        
        def restrict_bc(bc, R):
            if isinstance(bc, np.ndarray):
                return R @ bc
            return bc

        bc_x0_c = restrict_bc(lvl.bc_x0, lvl.Ry)
        bc_x1_c = restrict_bc(lvl.bc_x1, lvl.Ry)
        bc_y0_c = restrict_bc(lvl.bc_y0, lvl.Rx)
        bc_y1_c = restrict_bc(lvl.bc_y1, lvl.Rx)
        
        dx_c = Lx / (lvl.N_coarse + 1)
        dy_c = Ly / (lvl.N_coarse + 1)
        
        lvl = Level4th(lvl.N_coarse, dx_c, dy_c, f_coarse, bc_x0_c, bc_x1_c, bc_y0_c, bc_y1_c, Lx, Ly)
        levels.append(lvl)
        
    return levels

def v_cycle_4th(level_idx, levels, u, f, bc_x0, bc_x1, bc_y0, bc_y1, solve_strip, nu1=2, nu2=2):
    lvl = levels[level_idx]
    
    # Pre-smoothing
    u = relax_2d_4th(u, f, lvl.dx, lvl.dy, bc_x0, bc_x1, bc_y0, bc_y1, lvl.A_strip, solve_strip, nu1)
    
    if lvl.Px is not None:
        # Residual
        r = compute_residual_2d_4th(u, f, lvl.dx, lvl.dy, bc_x0, bc_x1, bc_y0, bc_y1, lvl.A_strip)
        r_c = lvl.restrict_field(r)
        
        # Coarse grid error equation (A_c e_c = r_c, homogeneous BCs)
        e_c = np.zeros((lvl.N_coarse, lvl.N_coarse))
        e_c = v_cycle_4th(level_idx + 1, levels, e_c, r_c, 0.0, 0.0, 0.0, 0.0, solve_strip, nu1, nu2)
        
        # Prolong & Correct
        u += lvl.prolong_field(e_c)
        
        # Post-smoothing
        u = relax_2d_4th(u, f, lvl.dx, lvl.dy, bc_x0, bc_x1, bc_y0, bc_y1, lvl.A_strip, solve_strip, nu2)
    else:
        # Exact solve at coarsest level (just do a lot of smoothing for now, Thomas will solve strips perfectly)
        u = relax_2d_4th(u, f, lvl.dx, lvl.dy, bc_x0, bc_x1, bc_y0, bc_y1, lvl.A_strip, solve_strip, 20)
        
    return u

def fmg_2d_4th(N, f_vals, dx, dy=None, bc_x0=0.0, bc_x1=0.0, bc_y0=0.0, bc_y1=0.0,
               inner="thomas", inner_kwargs=None, tol=1e-6, max_cycles=15, nu1=2, nu2=2):
    if inner_kwargs is None: inner_kwargs = {}
    if dy is None: dy = dx
    
    levels = build_hierarchy_2d_4th(N, dx, dy, f_vals, bc_x0, bc_x1, bc_y0, bc_y1)
    
    # Wrapper for inner solvers (Thomas, QSVT, etc)
    raw_solve = get_inner(inner, fallback_to_thomas=True, **inner_kwargs)
    def solve_strip(A, b):
        x = raw_solve(A, b)
        return x

    # --- FMG Start (Coarse to Fine) ---
    u = np.zeros((levels[-1].N, levels[-1].N))
    
    for l in range(len(levels) - 1, -1, -1):
        if l < len(levels) - 1:
            u = levels[l].prolong_field(u)
        lvl = levels[l]
        # At each level on the way up, run a full V-cycle to solve the exact problem
        u = v_cycle_4th(l, levels, u, lvl.f_vals, lvl.bc_x0, lvl.bc_x1, lvl.bc_y0, lvl.bc_y1, solve_strip, nu1, nu2)
        
    # --- V-cycles on Fine Grid ---
    history = []
    lvl = levels[0]
    # Compute initial un-smoothed RHS norm for residual scaling
    init_r = compute_residual_2d_4th(np.zeros_like(u), lvl.f_vals, lvl.dx, lvl.dy,
                                     lvl.bc_x0, lvl.bc_x1, lvl.bc_y0, lvl.bc_y1, lvl.A_strip)
    b_norm = np.linalg.norm(init_r) + 1e-300
    
    for i in range(max_cycles):
        u = v_cycle_4th(0, levels, u, lvl.f_vals, lvl.bc_x0, lvl.bc_x1, lvl.bc_y0, lvl.bc_y1, solve_strip, nu1, nu2)
        
        r = compute_residual_2d_4th(u, lvl.f_vals, lvl.dx, lvl.dy,
                                    lvl.bc_x0, lvl.bc_x1, lvl.bc_y0, lvl.bc_y1, lvl.A_strip)
        res = np.linalg.norm(r) / b_norm
        history.append(res)
        
        if res < tol:
            break
            
    return u, history

def sor_2d_4th(N, f_vals, dx, dy=None, bc_x0=0.0, bc_x1=0.0, bc_y0=0.0, bc_y1=0.0,
               inner="thomas", inner_kwargs=None, tol=1e-6, max_iter=5000, omega=1.0):
    if inner_kwargs is None: inner_kwargs = {}
    if dy is None: dy = dx
    kappa_aniso = (dx / dy)**2
    
    A_strip = build_strip_matrix_4th(N, dx) - 24.0 * kappa_aniso * np.eye(N)
    
    raw_solve = get_inner(inner, fallback_to_thomas=True, **inner_kwargs)
    def solve_strip(A, b):
        return raw_solve(A, b)

    phi = np.zeros((N, N))
    history = []
    
    init_r = compute_residual_2d_4th(np.zeros_like(phi), f_vals, dx, dy, bc_x0, bc_x1, bc_y0, bc_y1, A_strip)
    b_norm = np.linalg.norm(init_r) + 1e-300
    
    for i in range(max_iter):
        for j in range(N):
            b_j = _build_rhs_strip(j, phi, f_vals, dx, dy, bc_x0, bc_x1, bc_y0, bc_y1)
            x_new = solve_strip(A_strip, b_j)
            phi[:, j] = omega * x_new + (1.0 - omega) * phi[:, j]
            
        r = compute_residual_2d_4th(phi, f_vals, dx, dy, bc_x0, bc_x1, bc_y0, bc_y1, A_strip)
        res = np.linalg.norm(r) / b_norm
        history.append(res)
        
        if res < tol:
            break
            
    return phi, history

# ── 3D FMG Implementation ─────────────────────────────────────────────────────

def _build_rhs_strip_3d(j, k, phi, f_vals, dx, dy, dz, bc_lo, bc_hi, periodic=(False,False,False)):
    kappa_y = (dx / dy)**2
    kappa_z = (dx / dz)**2
    N = phi.shape[0]
    b = 12.0 * dx**2 * f_vals[:, j, k].copy()

    def _extract_bc(bc_array, idx_j, idx_k):
        if isinstance(bc_array, np.ndarray):
            return bc_array[idx_j, idx_k]
        return bc_array

    val0 = _extract_bc(bc_lo[0], j, k)
    val1 = _extract_bc(bc_hi[0], j, k)
    b[0] -= 18.0 * val0
    if N > 1: b[1] += val0
    b[-1] -= 18.0 * val1
    if N > 1: b[-2] += val1

    if j > 0:
        b -= 12.0 * kappa_y * phi[:, j-1, k]
    elif periodic[1]:
        b -= 12.0 * kappa_y * phi[:, -1, k]
    else:
        ay0 = bc_lo[1]
        v_ay0 = ay0[:, k] if isinstance(ay0, np.ndarray) else np.full(N, ay0)
        b -= 12.0 * kappa_y * v_ay0

    if j < N - 1:
        b -= 12.0 * kappa_y * phi[:, j+1, k]
    elif periodic[1]:
        b -= 12.0 * kappa_y * phi[:, 0, k]
    else:
        ay1 = bc_hi[1]
        v_ay1 = ay1[:, k] if isinstance(ay1, np.ndarray) else np.full(N, ay1)
        b -= 12.0 * kappa_y * v_ay1

    if k > 0:
        b -= 12.0 * kappa_z * phi[:, j, k-1]
    elif periodic[2]:
        b -= 12.0 * kappa_z * phi[:, j, -1]
    else:
        az0 = bc_lo[2]
        v_az0 = az0[:, j] if isinstance(az0, np.ndarray) else np.full(N, az0)
        b -= 12.0 * kappa_z * v_az0

    if k < N - 1:
        b -= 12.0 * kappa_z * phi[:, j, k+1]
    elif periodic[2]:
        b -= 12.0 * kappa_z * phi[:, j, 0]
    else:
        az1 = bc_hi[2]
        v_az1 = az1[:, j] if isinstance(az1, np.ndarray) else np.full(N, az1)
        b -= 12.0 * kappa_z * v_az1

    return b

def compute_residual_3d_4th(phi, f_vals, dx, dy, dz, bc_lo, bc_hi, A_strip, periodic=(False,False,False)):
    N = phi.shape[0]
    r = np.zeros_like(phi)
    for k in range(N):
        for j in range(N):
            b_j = _build_rhs_strip_3d(j, k, phi, f_vals, dx, dy, dz, bc_lo, bc_hi, periodic)
            r[:, j, k] = b_j - A_strip @ phi[:, j, k]
    return r

def relax_3d_4th(phi, f_vals, dx, dy, dz, bc_lo, bc_hi, A_strip, solve_strip, nu, periodic=(False,False,False)):
    N = phi.shape[0]
    for _ in range(nu):
        for k in range(N):
            for j in range(N):
                b_j = _build_rhs_strip_3d(j, k, phi, f_vals, dx, dy, dz, bc_lo, bc_hi, periodic)
                phi[:, j, k] = solve_strip(A_strip, b_j)
    return phi

class Level4th3D:
    def __init__(self, N, dx, dy, dz, bc_lo, bc_hi, periodic=(False,False,False), f_vals=None):
        self.N = N
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.bc_lo = bc_lo
        self.bc_hi = bc_hi
        self.periodic = periodic
        
        kappa_y = (dx / dy)**2
        kappa_z = (dx / dz)**2
        self.A_strip = build_strip_matrix_4th(N, dx) - 24.0 * kappa_y * np.eye(N) - 24.0 * kappa_z * np.eye(N)
        
        if f_vals is None:
            self.f_vals = np.zeros((N, N, N))
        else:
            self.f_vals = f_vals

def build_hierarchy_3d_4th(N_fine, dx_fine, dy_fine, dz_fine, bc_lo, bc_hi, periodic, f_fine):
    levels = []
    levels.append(Level4th3D(N_fine, dx_fine, dy_fine, dz_fine, bc_lo, bc_hi, periodic, f_fine))
    
    N = N_fine
    while N > 4:
        N_coarse = N // 2
        dx_c = levels[-1].dx * 2
        dy_c = levels[-1].dy * 2
        dz_c = levels[-1].dz * 2
        
        bcl_c = (
            levels[-1].bc_lo[0],
            levels[-1].bc_lo[1][1::2] if isinstance(levels[-1].bc_lo[1], np.ndarray) else levels[-1].bc_lo[1],
            levels[-1].bc_lo[2][1::2] if isinstance(levels[-1].bc_lo[2], np.ndarray) else levels[-1].bc_lo[2],
        )
        bch_c = (
            levels[-1].bc_hi[0],
            levels[-1].bc_hi[1][1::2] if isinstance(levels[-1].bc_hi[1], np.ndarray) else levels[-1].bc_hi[1],
            levels[-1].bc_hi[2][1::2] if isinstance(levels[-1].bc_hi[2], np.ndarray) else levels[-1].bc_hi[2],
        )
        levels.append(Level4th3D(N_coarse, dx_c, dy_c, dz_c, bcl_c, bch_c, periodic))
        N = N_coarse
        
    return levels

def v_cycle_3d_4th(level_idx, levels, u, f, bc_lo, bc_hi, solve_strip, nu1, nu2):
    lvl = levels[level_idx]
    
    u = relax_3d_4th(u, f, lvl.dx, lvl.dy, lvl.dz, bc_lo, bc_hi, lvl.A_strip, solve_strip, nu1, lvl.periodic)
    
    if level_idx == len(levels) - 1:
        return relax_3d_4th(u, f, lvl.dx, lvl.dy, lvl.dz, bc_lo, bc_hi, lvl.A_strip, solve_strip, nu2, lvl.periodic)
        
    r = compute_residual_3d_4th(u, f, lvl.dx, lvl.dy, lvl.dz, bc_lo, bc_hi, lvl.A_strip, lvl.periodic)
    
    lvl_c = levels[level_idx + 1]
    r_c = np.zeros((lvl_c.N, lvl_c.N, lvl_c.N))
    for k in range(lvl_c.N):
        for j in range(lvl_c.N):
            r_c[:, j, k] = R_1d @ r[:, 2*j+1, 2*k+1]
            
    e_c = np.zeros((lvl_c.N, lvl_c.N, lvl_c.N))
    zero_bcl = (0.0, 0.0, 0.0)
    zero_bch = (0.0, 0.0, 0.0)
    
    e_c = v_cycle_3d_4th(level_idx + 1, levels, e_c, r_c, zero_bcl, zero_bch, solve_strip, nu1, nu2)
    
    for k in range(lvl_c.N):
        for j in range(lvl_c.N):
            u[:, 2*j+1, 2*k+1] += P_1d @ e_c[:, j, k]
            
    u = relax_3d_4th(u, f, lvl.dx, lvl.dy, lvl.dz, bc_lo, bc_hi, lvl.A_strip, solve_strip, nu2, lvl.periodic)
    return u

def fmg_3d_4th(N, f_vals, dx, dy=None, dz=None, bc_lo=(0.,0.,0.), bc_hi=(0.,0.,0.), periodic=(False,False,False),
               inner="thomas", inner_kwargs=None, tol=1e-6, max_cycles=30, nu1=2, nu2=2):
    if inner_kwargs is None: inner_kwargs = {}
    if dy is None: dy = dx
    if dz is None: dz = dx
    
    if any(periodic):
        raise ValueError("FMG 3D 4th order does not currently support periodic boundaries. Fall back to SOR.")
    
    levels = build_hierarchy_3d_4th(N, dx, dy, dz, bc_lo, bc_hi, periodic, f_vals)
    
    raw_solve = get_inner(inner, fallback_to_thomas=True, **inner_kwargs)
    def solve_strip(A, b):
        return raw_solve(A, b)
        
    u = np.zeros((levels[-1].N, levels[-1].N, levels[-1].N))
    
    for lvl_idx in reversed(range(len(levels) - 1)):
        lvl = levels[lvl_idx]
        u_fine = np.zeros((lvl.N, lvl.N, lvl.N))
        for k in range(levels[lvl_idx+1].N):
            for j in range(levels[lvl_idx+1].N):
                u_fine[:, 2*j+1, 2*k+1] = P_1d @ u[:, j, k]
        u = u_fine
        
        u = v_cycle_3d_4th(lvl_idx, levels, u, lvl.f_vals, lvl.bc_lo, lvl.bc_hi, solve_strip, nu1, nu2)
        
    lvl = levels[0]
    history = []
    
    init_r = compute_residual_3d_4th(np.zeros_like(u), f_vals, dx, dy, dz, bc_lo, bc_hi, lvl.A_strip, periodic)
    b_norm = np.linalg.norm(init_r) + 1e-300
    
    for i in range(max_cycles):
        u = v_cycle_3d_4th(0, levels, u, lvl.f_vals, lvl.bc_lo, lvl.bc_hi, solve_strip, nu1, nu2)
        
        r = compute_residual_3d_4th(u, lvl.f_vals, lvl.dx, lvl.dy, lvl.dz,
                                    lvl.bc_lo, lvl.bc_hi, lvl.A_strip, periodic)
        res = np.linalg.norm(r) / b_norm
        history.append(res)
        if res < tol:
            break
            
    return u, history

def sor_3d_4th(N, f_vals, dx, dy=None, dz=None, bc_lo=(0.,0.,0.), bc_hi=(0.,0.,0.), periodic=(False,False,False),
               inner="thomas", inner_kwargs=None, tol=1e-6, max_iter=5000, omega=1.0):
    if inner_kwargs is None: inner_kwargs = {}
    if dy is None: dy = dx
    if dz is None: dz = dx
    
    kappa_y = (dx / dy)**2
    kappa_z = (dx / dz)**2
    A_strip = build_strip_matrix_4th(N, dx) - 24.0 * kappa_y * np.eye(N) - 24.0 * kappa_z * np.eye(N)
    
    raw_solve = get_inner(inner, fallback_to_thomas=True, **inner_kwargs)
    def solve_strip(A, b):
        return raw_solve(A, b)

    phi = np.zeros((N, N, N))
    history = []
    
    init_r = compute_residual_3d_4th(np.zeros_like(phi), f_vals, dx, dy, dz, bc_lo, bc_hi, A_strip, periodic)
    b_norm = np.linalg.norm(init_r) + 1e-300
    
    for i in range(max_iter):
        for k in range(N):
            for j in range(N):
                b_j = _build_rhs_strip_3d(j, k, phi, f_vals, dx, dy, dz, bc_lo, bc_hi, periodic)
                x_new = solve_strip(A_strip, b_j)
                phi[:, j, k] = omega * x_new + (1.0 - omega) * phi[:, j, k]
            
        r = compute_residual_3d_4th(phi, f_vals, dx, dy, dz, bc_lo, bc_hi, A_strip, periodic)
        res = np.linalg.norm(r) / b_norm
        history.append(res)
        
        if res < tol:
            break
            
    return phi, history

if __name__ == "__main__":
    # Test FMG 4th order on a simple problem
    N = 32
    L = 1.0
    dx = L / (N + 1)
    
    x = np.arange(1, N + 1) * dx
    y = np.arange(1, N + 1) * dx
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Analytical solution: u = sin(pi x) sin(pi y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # f = Laplacian u = -2 pi^2 sin(pi x) sin(pi y)
    f_vals = -2 * np.pi**2 * u_exact
    
    print(f"Running FMG 4th Order on {N}x{N} grid...")
    t0 = time.perf_counter()
    u_fmg, hist = fmg_2d_4th(N, f_vals, dx, dy=dx, inner="thomas", max_cycles=15)
    t1 = time.perf_counter()
    
    print(f"FMG converged in {len(hist)} cycles. Time: {t1 - t0:.4f}s")
    for i, res in enumerate(hist):
        print(f"  Cycle {i+1}: {res:.3e}")
        
    err = np.max(np.abs(u_fmg - u_exact)) / np.max(np.abs(u_exact))
    print(f"Max Relative Error (vs Analytical): {err * 100:.4f}%")
