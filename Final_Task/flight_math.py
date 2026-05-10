import numpy as np

# Aircraft parameters (example values, need to be adjusted)
m = 1000.0  # mass kg
Ixx = 1000.0  # moment of inertia kg m^2
Iyy = 2000.0
Izz = 1500.0
Ixz = 100.0
S = 20.0  # wing area m^2
b = 10.0  # wing span m
c = 2.0   # mean chord m
rho = 1.225  # air density kg/m^3

# Aerodynamic coefficients (simplified)
C_L0 = 0.1
C_La = 5.0
C_D0 = 0.02
C_Da2 = 0.1
C_Ybeta = -0.5
C_Yp = 0.0
C_Yr = 0.1
C_lbeta = -0.1
C_lp = -0.5
C_lr = 0.1
C_lda = 0.1  # aileron
C_m0 = 0.0
C_ma = -1.0
C_mq = -5.0
C_mde = -1.0  # elevator
C_nbeta = 0.1
C_np = -0.1
C_nr = -0.5
C_ndr = -0.1  # rudder

# Thrust positions (example)
thrust1_pos = np.array([0.0, -1.0, 0.0])  # left engine
thrust2_pos = np.array([0.0, 1.0, 0.0])   # right engine

def build_dcm(euler_vector, degrees=True):
    """
    Builds the Body-to-NED Direction Cosine Matrix (DCM).
    """
    phi, theta, psi = euler_vector
    
    # Convert to radians if necessary
    if degrees:
        phi, theta, psi = np.radians([phi, theta, psi])
        
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    # Construct and return the matrix
    C_b_n = np.array([
        [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
        [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
        [-s_theta,      s_phi*c_theta,                     c_phi*c_theta]
    ])
    
    return C_b_n


def transform_flight_data(euler_vector, v_body_vector):
    """
    Takes Euler angles and a Body velocity vector, returns NED velocity and Euler angles.
    
    Inputs:
    euler_vector  : [phi, theta, psi] (Roll, Pitch, Yaw)
    v_body_vector : [u, v, w] (Velocity in body frame)

    
    Outputs:
    v_ned_vec     : Velocity vector in NED frame [V_North, V_East, V_Down]
    euler_vector  : The original [phi, theta, psi] vector
    """
    
    # 1. Get the DCM for the given Euler angles
    C_b_n = build_dcm(euler_vector, True)  # We know our input is in degrees, so set degrees=True explicitly
    
    # 2. Keep a copy of the original angles for the output
    output_euler = np.array(euler_vector)
    
    # 3. Convert body velocity to column vector and multiply
    v_body = np.array(v_body_vector).reshape(3, 1)
    v_ned = np.dot(C_b_n, v_body)
    
    return v_ned.flatten(), output_euler

def compute_aero_angles(euler_vector, v_body, v_ned=None):
    """
    Computes aerodynamic and flight path angles.
    
    Inputs:
    euler_vector : [phi, theta, psi] (degrees)
    v_body       : Velocity vector in body frame [u, v, w]
    v_ned        : Velocity vector in NED frame [V_N, V_E, V_D] (optional)
    
    Outputs:
    alpha : Angle of Attack (degrees)
    beta  : Sideslip Angle (degrees)  — from V_NED if available
    gamma : Climb/Flight Path Angle (degrees) — from V_NED if available
    """
    
    u, v, w = v_body
    
    # Calculate total airspeed magnitude
    V_tot = np.linalg.norm(v_body)
    
    # Handle stationary case to prevent division by zero
    if V_tot < 1e-6:
        return np.array([0.0, 0.0, 0.0])
        
    # Angle of Attack (Alpha) — always from body frame
    alpha_rad = np.arctan2(w, u)
    
    if v_ned is not None:
        V_N, V_E, V_D = v_ned
        V_ned_mag = np.linalg.norm(v_ned)
        
        if V_ned_mag < 1e-6:
            beta_rad = 0.0
            gamma_rad = 0.0
        else:
            # Sideslip (Beta) from NED: lateral deviation
            V_horiz = np.sqrt(V_N**2 + V_E**2)
            beta_rad = np.arctan2(V_E, V_N) - np.radians(euler_vector[2])
            # Wrap to [-pi, pi]
            beta_rad = (beta_rad + np.pi) % (2 * np.pi) - np.pi
            
            # Climb Angle (Gamma) from NED: γ = arcsin(-V_D / |V|)
            gamma_rad = np.arcsin(np.clip(-V_D / V_ned_mag, -1.0, 1.0))
    else:
        # Fallback: original formulas
        beta_rad = np.arcsin(v / V_tot)
        theta_rad = np.radians(euler_vector[1])
        gamma_rad = theta_rad - alpha_rad
    
    # Package the angles into a vector and convert to degrees
    aero_angles = np.degrees([alpha_rad, beta_rad, gamma_rad])
    
    return aero_angles

def aircraft_state(v_body, v_ned, euler, angular_rates, aero_angles, degrees=True):
    """
    Returns aircraft state values in the structured dictionary format 
    required by the assignment.
    """
    
    # 1. Build the exact dictionary structure the assignment requests
    state_values = {
        "angles": np.array(aero_angles),           # [alpha, beta, gamma] in degrees
        "velocities_body": np.array(v_body),     # [u, v, w] in body frame [m/s]
        "velocities_ned": np.array(v_ned),          # [V_N, V_E, V_D] in NED frame [m/s]
        "angular_rates": np.array(angular_rates),       # [p, q, r] roll, pitch, yaw rates [rad/s]
        "attitude": np.array(euler)     # [phi, theta, psi] Euler angles
    }
    
    return state_values


# ============================================================
# Phase 1 — Angular Rates → Euler Angles
# ============================================================

def calculate_H(phi, theta):
    """
    Builds the 3×3 kinematic transformation matrix H that relates
    body angular rates [p, q, r] to Euler angle rates [φ̇, θ̇, ψ̇].

    Φ̇ = H · ω_B

    Inputs (radians):
        phi   : roll angle  (φ)
        theta : pitch angle (θ)

    Returns:
        H : 3×3 numpy array
    """
    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    t_theta = np.tan(theta)
    sec_theta = 1.0 / np.cos(theta)

    H = np.array([
        [1.0,  s_phi * t_theta,    c_phi * t_theta],
        [0.0,  c_phi,             -s_phi           ],
        [0.0,  s_phi * sec_theta,  c_phi * sec_theta]
    ])
    return H


def euler_rate(H, omega_B):
    """
    Computes Euler angle rates: Φ̇ = H × ω_B

    Inputs:
        H       : 3×3 kinematic matrix from calculate_H
        omega_B : [p, q, r] angular rates in body frame (rad/s)

    Returns:
        phi_dot, theta_dot, psi_dot  as a 1-D array (rad/s)
    """
    omega = np.array(omega_B).flatten()
    return H @ omega


def integrate(derivative, prev_value, dt):
    """
    Generic first-order (Euler) numerical integrator.
    new_value = prev_value + derivative * dt

    Works for scalars and arrays alike.
    """
    return np.asarray(prev_value) + np.asarray(derivative) * dt


def terminator(euler_angles_rad):
    """
    Wraps / clamps Euler angles to valid ranges and checks for gimbal lock.

    φ (roll)  →  wrapped to [-π, π]
    θ (pitch) →  clamped to [-π/2, π/2]
    ψ (yaw)   →  wrapped to [-π, π]

    Also returns a boolean flag indicating if gimbal lock is detected
    (|θ| within 1° of 90°).

    Inputs:
        euler_angles_rad : [phi, theta, psi] in radians

    Returns:
        wrapped : numpy array [phi, theta, psi] in radians
        gimbal_lock : bool, True if |θ| ≈ 90°
    """
    phi, theta, psi = euler_angles_rad

    # Wrap φ and ψ to [-π, π]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    psi = (psi + np.pi) % (2 * np.pi) - np.pi

    # Clamp θ to [-π/2, π/2]
    theta = np.clip(theta, -np.pi / 2, np.pi / 2)

    # Gimbal lock detection: |θ| within ~1° of ±90°
    gimbal_lock = abs(abs(theta) - np.pi / 2) < np.radians(1.0)

    return np.array([phi, theta, psi]), gimbal_lock


# ============================================================
# Phase 2 — Body Accelerations → NED Velocity & Position
# ============================================================

def accel_body_to_ned(C_b_e, accel_body):
    """
    Transforms body-frame accelerations to NED frame.

    Inputs:
        C_b_e      : 3×3 DCM (body-to-NED)
        accel_body : [ax, ay, az] accelerations in body frame (m/s²)

    Returns:
        accel_ned : [aN, aE, aD] accelerations in NED frame (m/s²)
    """
    return C_b_e @ np.array(accel_body).flatten()


def add_gravity(accel_ned, g=9.81):
    """
    Adds gravity to NED-frame accelerations.
    In NED, gravity acts in the +Down direction: [0, 0, +g]

    Inputs:
        accel_ned : [aN, aE, aD] (m/s²)
        g         : gravitational acceleration (default 9.81 m/s²)

    Returns:
        corrected : [aN, aE, aD + g] (m/s²)
    """
    gravity = np.array([0.0, 0.0, g])
    return np.asarray(accel_ned) + gravity


# ============================================================
# Phase 3 — Quaternion & Euler Axis
# ============================================================

def dcm_to_quaternion(C):
    """
    Converts a 3×3 DCM to a unit quaternion [q_s, q_x, q_y, q_z]
    using Shepperd's method to avoid numerical issues.

    Input:
        C : 3×3 DCM (body-to-NED)

    Returns:
        q : [q_s, q_x, q_y, q_z]  (scalar-first convention)
    """
    tr = np.trace(C)

    # Find the largest diagonal element to determine best pivot
    diag = [tr, C[0, 0], C[1, 1], C[2, 2]]
    idx = np.argmax(diag)

    if idx == 0:
        # tr is largest
        q_s = 0.5 * np.sqrt(1.0 + tr)
        k = 0.25 / q_s
        q_x = k * (C[2, 1] - C[1, 2])
        q_y = k * (C[0, 2] - C[2, 0])
        q_z = k * (C[1, 0] - C[0, 1])
    elif idx == 1:
        q_x = 0.5 * np.sqrt(1.0 + 2 * C[0, 0] - tr)
        k = 0.25 / q_x
        q_s = k * (C[2, 1] - C[1, 2])
        q_y = k * (C[0, 1] + C[1, 0])
        q_z = k * (C[0, 2] + C[2, 0])
    elif idx == 2:
        q_y = 0.5 * np.sqrt(1.0 + 2 * C[1, 1] - tr)
        k = 0.25 / q_y
        q_s = k * (C[0, 2] - C[2, 0])
        q_x = k * (C[0, 1] + C[1, 0])
        q_z = k * (C[1, 2] + C[2, 1])
    else:
        q_z = 0.5 * np.sqrt(1.0 + 2 * C[2, 2] - tr)
        k = 0.25 / q_z
        q_s = k * (C[1, 0] - C[0, 1])
        q_x = k * (C[0, 2] + C[2, 0])
        q_y = k * (C[1, 2] + C[2, 1])

    q = np.array([q_s, q_x, q_y, q_z])
    # Normalize to ensure unit quaternion
    q = q / np.linalg.norm(q)
    # Convention: keep scalar part positive
    if q[0] < 0:
        q = -q
    return q


def angle2quaternion(phi, theta, psi, degrees=True):
    """
    Converts Euler angles → quaternion, going through DCM as intermediate step.
    This is the 'angle 2 quaternion' block from the flow diagram.

    Φ → DCM → DCM to quaternion → q

    Inputs:
        phi, theta, psi : Euler angles (roll, pitch, yaw)
        degrees         : if True, inputs are in degrees

    Returns:
        q : [q_s, q_x, q_y, q_z]
    """
    euler = [phi, theta, psi]
    C = build_dcm(euler, degrees=degrees)
    q = dcm_to_quaternion(C)
    return q


def quaternion_angle(q):
    """
    Extracts the rotation angle from a quaternion.
    θ = 2 · arccos(q_s)

    Input:
        q : [q_s, q_x, q_y, q_z]

    Returns:
        theta : rotation angle in radians
    """
    q_s = np.clip(q[0], -1.0, 1.0)  # numerical safety
    theta = 2.0 * np.arccos(abs(q_s))
    return theta


def quaternion_axis(q, theta):
    """
    Extracts the Euler axis (unit rotation axis) from a quaternion.
    ê = q_v / sin(θ/2)

    Inputs:
        q     : [q_s, q_x, q_y, q_z]
        theta : rotation angle in radians (from quaternion_angle)

    Returns:
        e_hat : [ex, ey, ez] unit vector, or [0,0,0] if θ ≈ 0
    """
    sin_half = np.sin(theta / 2.0)
    if abs(sin_half) < 1e-10:
        # No rotation — axis is undefined, return zero vector
        return np.array([0.0, 0.0, 0.0])

    q_v = q[1:4]
    e_hat = q_v / sin_half
    # Normalize for safety
    norm = np.linalg.norm(e_hat)
    if norm > 1e-10:
        e_hat = e_hat / norm
    return e_hat


def compute_rcam_derivatives(state, control):
    """
    Computes the derivatives of the state using RCAM model.

    State: [u, v, w, p, q, r, x, y, z, phi, theta, psi]
    Control: [aileron_deg, elevator_deg, rudder_deg, thrust1_N, thrust2_N]

    Returns: state_dot
    """
    u, v, w, p, q, r, x, y, z, phi, theta, psi = state
    da, de, dr, T1, T2 = control

    # Convert angles to radians
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)
    psi_rad = np.radians(psi)
    da_rad = np.radians(da)
    de_rad = np.radians(de)
    dr_rad = np.radians(dr)

    # Airspeed
    V = np.sqrt(u**2 + v**2 + w**2)
    if V < 1e-6:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V, -1, 1))

    # Dynamic pressure
    q_dyn = 0.5 * rho * V**2

    # Aerodynamic coefficients
    C_L = C_L0 + C_La * alpha
    C_D = C_D0 + C_Da2 * alpha**2
    C_Y = C_Ybeta * beta + C_Yp * (p * b / (2 * V) if V > 0 else 0) + C_Yr * (r * b / (2 * V) if V > 0 else 0)

    C_l = C_lbeta * beta + C_lp * (p * b / (2 * V) if V > 0 else 0) + C_lr * (r * b / (2 * V) if V > 0 else 0) + C_lda * da_rad
    C_m = C_m0 + C_ma * alpha + C_mq * (q * c / (2 * V) if V > 0 else 0) + C_mde * de_rad
    C_n = C_nbeta * beta + C_np * (p * b / (2 * V) if V > 0 else 0) + C_nr * (r * b / (2 * V) if V > 0 else 0) + C_ndr * dr_rad

    # Forces in body frame (aerodynamics)
    F_aero_x = q_dyn * S * (-C_D * np.cos(alpha) + C_L * np.sin(alpha))
    F_aero_y = q_dyn * S * C_Y
    F_aero_z = q_dyn * S * (-C_D * np.sin(alpha) - C_L * np.cos(alpha))

    # Thrust in body frame (assume along x)
    F_thrust_x = T1 + T2
    F_thrust_y = 0.0
    F_thrust_z = 0.0

    # Gravity in body frame
    g = 9.81
    C_n_b = build_dcm([phi, theta, psi], degrees=True).T  # NED to body
    gravity_body = C_n_b @ np.array([0, 0, g])  # gravity in NED is [0,0,g], transform to body

    F_x = F_aero_x + F_thrust_x + gravity_body[0]
    F_y = F_aero_y + F_thrust_y + gravity_body[1]
    F_z = F_aero_z + F_thrust_z + gravity_body[2]

    # Moments in body frame
    L = q_dyn * S * b * C_l
    M = q_dyn * S * c * C_m
    N = q_dyn * S * b * C_n

    # Thrust moments (assume engines at y = ±1 m)
    d = 1.0  # distance from CG
    M += (T2 - T1) * d  # torque from thrust difference

    # Equations of motion
    # Linear accelerations
    u_dot = (F_x / m) - q * w + r * v
    v_dot = (F_y / m) - r * u + p * w
    w_dot = (F_z / m) - p * v + q * u

    # Angular accelerations
    I = np.array([[Ixx, 0, -Ixz],
                  [0, Iyy, 0],
                  [-Ixz, 0, Izz]])
    omega = np.array([p, q, r])
    M_vec = np.array([L, M, N])
    omega_dot = np.linalg.solve(I, M_vec - np.cross(omega, I @ omega))

    # Position derivatives (in NED)
    C_b_n = build_dcm([phi, theta, psi], degrees=True)
    vel_ned = C_b_n @ np.array([u, v, w])
    x_dot, y_dot, z_dot = vel_ned

    # Euler angle derivatives
    H = calculate_H(phi_rad, theta_rad)
    euler_dot = euler_rate(H, [p, q, r])

    # Convert euler_dot to degrees
    euler_dot_deg = np.degrees(euler_dot)

    state_dot = np.array([u_dot, v_dot, w_dot,
                          omega_dot[0], omega_dot[1], omega_dot[2],
                          x_dot, y_dot, z_dot,
                          euler_dot_deg[0], euler_dot_deg[1], euler_dot_deg[2]])

    return state_dot
