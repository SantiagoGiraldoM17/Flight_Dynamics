import numpy as np

# ── RCAM Constants ──────────────────────────────────────────
g = 9.81
rho = 1.225
S = 260.0
St = 64.0
lt = 24.8
l = 6.6
b = 44.8
cbar = 6.6
XAPT1 = 0.0; YAPT1 = 7.94;  ZAPT1 = 1.9
XAPT2 = 0.0; YAPT2 = -7.94; ZAPT2 = 1.9
m = 120000.0
alpha0 = -11.5 * np.pi / 180.0
deps_dalpha = 0.25

# Inertia tensor [kg·m²]
Inertia = m * np.array([
    [40.07, 0.0,   2.098],
    [0.0,   64.0,  0.0  ],
    [2.098, 0.0,   99.92]
])


def rcam_derivatives(X, U):
    """
    RCAM 6-DOF nonlinear aircraft model.

    State X (9): [uB, vB, wB, p, q, r, phi, theta, psi]
        velocities m/s, rates rad/s, angles rad

    Control U (5): [dA, dE, dR, dTH1, dTH2]
        surfaces rad, throttles 0-1

    Returns: Xdot (9)
    """
    uB, vB, wB = X[0], X[1], X[2]
    p, q, r    = X[3], X[4], X[5]
    phi, theta, psi = X[6], X[7], X[8]

    dA, dE, dR = U[0], U[1], U[2]
    dTH1, dTH2 = U[3], U[4]

    cphi   = np.cos(phi);   sphi   = np.sin(phi)
    ctheta = np.cos(theta); stheta = np.sin(theta)
    ttheta = np.tan(theta)

    # ── Airspeed (no wind) ──────────────────────────────────
    ua, va, wa = uB, vB, wB
    V = np.sqrt(ua**2 + va**2 + wa**2)
    if V < 0.1:
        V = 0.1

    alpha = np.arctan2(wa, ua)
    beta  = np.arcsin(np.clip(va / V, -1.0, 1.0))
    sa = np.sin(alpha); ca = np.cos(alpha)
    sb = np.sin(beta);  cb = np.cos(beta)

    # ── Aero coefficients ───────────────────────────────────
    eps     = deps_dalpha * (alpha - alpha0)
    alpha_t = alpha - eps + dE + 1.3 * q * lt / V

    CL_wb = 5.5 * (alpha - alpha0)
    CL_t  = (St / S) * 3.1 * alpha_t
    CL    = CL_wb + CL_t
    CD    = 0.13 + 0.07 * (CL_wb - 0.45)**2
    CY    = -1.6 * beta + 0.24 * dR

    Cl_c = (-1.4 * beta
            + (l / V) * (-11.0 * p + 5.0 * r)
            + (+0.6 * dA + 0.22 * dR))   # +dA → right roll (standard sign convention)

    Cm_c = (-0.59
            - 3.1 * (St * lt / (S * l)) * (alpha - eps)
            + (l / V) * (-4.03 * (St * lt**2 / (S * l**2)) * q)
            + (-3.1 * (St * lt / (S * l)) * dE))

    Cn_c = ((1.0 - alpha * 180.0 / (15.0 * np.pi)) * beta
            + (l / V) * (1.7 * p - 11.5 * r)
            + (-0.63 * dR))

    # ── Aero forces (body axes) ─────────────────────────────
    qbar = 0.5 * rho * V**2
    D = CD * qbar * S
    Y = CY * qbar * S
    L = CL * qbar * S

    FxA =  L * sa - D * ca * cb - Y * ca * sb
    FyA = -D * sb + Y * cb
    FzA = -L * ca - D * sa * cb - Y * sa * sb

    # ── Aero moments (body axes) ────────────────────────────
    LA = Cl_c * qbar * S * b
    MA = Cm_c * qbar * S * cbar
    NA = Cn_c * qbar * S * b

    # ── Engine forces & moments ─────────────────────────────
    F1 = dTH1 * m * g
    F2 = dTH2 * m * g
    Fx_eng = F1 + F2
    M_eng = (np.cross([XAPT1, YAPT1, ZAPT1], [F1, 0.0, 0.0])
           + np.cross([XAPT2, YAPT2, ZAPT2], [F2, 0.0, 0.0]))

    # ── Gravity (body axes) ─────────────────────────────────
    Fx_grav = m * g * (-stheta)
    Fy_grav = m * g * ctheta * sphi
    Fz_grav = m * g * ctheta * cphi

    # ── Total forces & moments ──────────────────────────────
    Fx = FxA + Fx_eng + Fx_grav
    Fy = FyA + Fy_grav
    Fz = FzA + Fz_grav
    Mx = LA + M_eng[0]
    My = MA + M_eng[1]
    Mz = NA + M_eng[2]

    # ── Translational EoM ───────────────────────────────────
    duB = Fx / m + r * vB - q * wB
    dvB = Fy / m + p * wB - r * uB
    dwB = Fz / m + q * uB - p * vB

    # ── Rotational EoM ──────────────────────────────────────
    omega   = np.array([p, q, r])
    M_total = np.array([Mx, My, Mz])
    omega_dot = np.linalg.solve(Inertia, M_total - np.cross(omega, Inertia @ omega))

    # ── Euler kinematics ────────────────────────────────────
    ctheta_safe = ctheta if abs(ctheta) > 1e-10 else 1e-10
    dphi   = p + sphi * ttheta * q + cphi * ttheta * r
    dtheta = cphi * q - sphi * r
    dpsi   = (sphi / ctheta_safe) * q + (cphi / ctheta_safe) * r

    return np.array([duB, dvB, dwB,
                     omega_dot[0], omega_dot[1], omega_dot[2],
                     dphi, dtheta, dpsi])


def position_derivatives(X):
    """
    Earth-frame position derivatives: [dX, dY, dZ] = R_BV^T @ [uB, vB, wB].
    Z positive down; altitude h = -Z.
    """
    uB, vB, wB = X[0], X[1], X[2]
    phi, theta, psi = X[6], X[7], X[8]

    cphi   = np.cos(phi);   sphi   = np.sin(phi)
    ctheta = np.cos(theta); stheta = np.sin(theta)
    cpsi   = np.cos(psi);   spsi   = np.sin(psi)

    dXe = (cpsi * ctheta * uB
         + (cpsi * stheta * sphi - spsi * cphi) * vB
         + (cpsi * stheta * cphi + spsi * sphi) * wB)
    dYe = (spsi * ctheta * uB
         + (spsi * stheta * sphi + cpsi * cphi) * vB
         + (spsi * stheta * cphi - cpsi * sphi) * wB)
    dZe = (-stheta * uB
         + ctheta * sphi * vB
         + ctheta * cphi * wB)

    return np.array([dXe, dYe, dZe])


def rcam_full(X12, U):
    """
    12-state wrapper: X12 = [uB,vB,wB, p,q,r, phi,theta,psi, Xe,Ye,Ze].
    Returns 12 derivatives.
    """
    Xdot9   = rcam_derivatives(X12[:9], U)
    pos_dot = position_derivatives(X12[:9])
    return np.concatenate([Xdot9, pos_dot])
