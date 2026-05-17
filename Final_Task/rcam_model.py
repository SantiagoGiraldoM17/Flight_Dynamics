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
    RCAM 6-DOF nonlinear aircraft model — xdot(X, U).

    Computes the 9-state derivative vector for the RCAM benchmark aircraft,
    following the 10-step structure required by the assignment rubric.

    Inputs:
        X (9,) state vector:
            x1=u, x2=v, x3=w   : body-axis velocities          [m/s]
            x4=p, x5=q, x6=r   : body-axis angular rates       [rad/s]
            x7=phi, x8=theta, x9=psi : Euler angles            [rad]

        U (5,) control vector:
            u1=dA   : aileron        [rad]
            u2=dE   : stabilizer     [rad]
            u3=dR   : rudder         [rad]
            u4=dTH1 : throttle 1     [-]   (0..1, right engine, YAPT1=+7.94)
            u5=dTH2 : throttle 2     [-]   (0..1, left  engine, YAPT2=-7.94)

    Returns:
        X_dot (9,) : time derivative of X
    """
    # ───────────────── State and Control vector ─────────────────
    x1 = X[0]   # u   - body x velocity (forward) [m/s]
    x2 = X[1]   # v   - body y velocity (lateral) [m/s]
    x3 = X[2]   # w   - body z velocity (down)    [m/s]
    x4 = X[3]   # p   - roll rate                 [rad/s]
    x5 = X[4]   # q   - pitch rate                [rad/s]
    x6 = X[5]   # r   - yaw rate                  [rad/s]
    x7 = X[6]   # phi   - roll angle              [rad]
    x8 = X[7]   # theta - pitch angle             [rad]
    x9 = X[8]   # psi   - yaw angle               [rad]

    u1 = U[0]   # dA   - aileron     [rad]
    u2 = U[1]   # dE   - stabilizer  [rad]
    u3 = U[2]   # dR   - rudder      [rad]
    u4 = U[3]   # dTH1 - throttle 1  [-]
    u5 = U[4]   # dTH2 - throttle 2  [-]

    # Descriptive aliases (match RCAM spec notation, used below)
    uB, vB, wB = x1, x2, x3
    p,  q,  r  = x4, x5, x6
    phi, theta, psi = x7, x8, x9
    dA, dE, dR      = u1, u2, u3
    dTH1, dTH2      = u4, u5

    # ───────────────── Constant definitions ─────────────────────
    # (aircraft constants live at module level: g, rho, S, St, lt,
    #  l, b, cbar, alpha0, deps_dalpha, m, Inertia, XAPT*/YAPT*/ZAPT*)
    sphi,   cphi   = np.sin(phi),   np.cos(phi)
    stheta, ctheta = np.sin(theta), np.cos(theta)
    ttheta         = np.tan(theta)

    # ════════════════════════════════════════════════════════════
    #  STEP 1 — Airspeed and aerodynamic angles
    # ════════════════════════════════════════════════════════════
    V = np.sqrt(uB**2 + vB**2 + wB**2)
    if V < 0.1:
        V = 0.1                                  # avoid div-by-zero
    alpha = np.arctan2(wB, uB)
    beta  = np.arcsin(np.clip(vB / V, -1.0, 1.0))
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta),  np.cos(beta)

    # ════════════════════════════════════════════════════════════
    #  STEP 2 — Aerodynamic coefficients (wing-body + tail)
    # ════════════════════════════════════════════════════════════
    eps     = deps_dalpha * (alpha - alpha0)
    alpha_t = alpha - eps + dE + 1.3 * q * lt / V    # tail angle of attack

    CL_wb = 5.5 * (alpha - alpha0)
    CL_t  = (St / S) * 3.1 * alpha_t
    CL    = CL_wb + CL_t
    CD    = 0.13 + 0.07 * (CL_wb - 0.45)**2
    CY    = -1.6 * beta + 0.24 * dR

    Cl_c = (-1.4 * beta
            + (l / V) * (-11.0 * p + 5.0 * r)
            + (+0.6 * dA + 0.22 * dR))     # +dA → right roll (standard convention)

    Cm_c = (-0.59
            - 3.1 * (St * lt / (S * l)) * (alpha - eps)
            + (l / V) * (-4.03 * (St * lt**2 / (S * l**2)) * q)
            + (-3.1 * (St * lt / (S * l)) * dE))

    Cn_c = ((1.0 - alpha * 180.0 / (15.0 * np.pi)) * beta
            + (l / V) * (1.7 * p - 11.5 * r)
            + (-0.63 * dR))

    # ════════════════════════════════════════════════════════════
    #  STEP 3 — Aerodynamic forces in body axes
    # ════════════════════════════════════════════════════════════
    qbar = 0.5 * rho * V**2                  # dynamic pressure
    D = CD * qbar * S                        # drag
    Y = CY * qbar * S                        # side force
    L = CL * qbar * S                        # lift

    FxA =  L * sa - D * ca * cb - Y * ca * sb
    FyA =          - D * sb     + Y * cb
    FzA = -L * ca - D * sa * cb - Y * sa * sb

    # ════════════════════════════════════════════════════════════
    #  STEP 4 — Aerodynamic moments in body axes
    # ════════════════════════════════════════════════════════════
    LA = Cl_c * qbar * S * b                 # roll
    MA = Cm_c * qbar * S * cbar              # pitch
    NA = Cn_c * qbar * S * b                 # yaw

    # ════════════════════════════════════════════════════════════
    #  STEP 5 — Engine forces and moments
    # ════════════════════════════════════════════════════════════
    F1 = dTH1 * m * g
    F2 = dTH2 * m * g
    Fx_eng = F1 + F2
    M_eng  = (np.cross([XAPT1, YAPT1, ZAPT1], [F1, 0.0, 0.0])
            + np.cross([XAPT2, YAPT2, ZAPT2], [F2, 0.0, 0.0]))

    # ════════════════════════════════════════════════════════════
    #  STEP 6 — Gravity components in body axes
    # ════════════════════════════════════════════════════════════
    Fx_grav = m * g * (-stheta)
    Fy_grav = m * g *  ctheta * sphi
    Fz_grav = m * g *  ctheta * cphi

    # ════════════════════════════════════════════════════════════
    #  STEP 7 — Total forces and moments
    # ════════════════════════════════════════════════════════════
    Fx = FxA + Fx_eng + Fx_grav
    Fy = FyA          + Fy_grav
    Fz = FzA          + Fz_grav
    Mx = LA + M_eng[0]
    My = MA + M_eng[1]
    Mz = NA + M_eng[2]

    # ════════════════════════════════════════════════════════════
    #  STEP 8 — Translational equations of motion
    # ════════════════════════════════════════════════════════════
    duB = Fx / m + r * vB - q * wB
    dvB = Fy / m + p * wB - r * uB
    dwB = Fz / m + q * uB - p * vB

    # ════════════════════════════════════════════════════════════
    #  STEP 9 — Rotational equations of motion
    #          ω̇ = I⁻¹ · (M − ω × I·ω)
    # ════════════════════════════════════════════════════════════
    omega     = np.array([p, q, r])
    M_total   = np.array([Mx, My, Mz])
    omega_dot = np.linalg.solve(Inertia, M_total - np.cross(omega, Inertia @ omega))
    dp, dq, dr = omega_dot

    # ════════════════════════════════════════════════════════════
    #  STEP 10 — Euler-angle kinematics
    # ════════════════════════════════════════════════════════════
    ctheta_safe = ctheta if abs(ctheta) > 1e-10 else 1e-10   # gimbal-lock guard
    dphi   = p + sphi * ttheta * q + cphi * ttheta * r
    dtheta =     cphi          * q - sphi          * r
    dpsi   =    (sphi / ctheta_safe) * q + (cphi / ctheta_safe) * r

    # ───────────────── Assemble derivative vector ───────────────
    X_dot = np.array([duB, dvB, dwB,
                      dp,  dq,  dr,
                      dphi, dtheta, dpsi])
    return X_dot


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
