import numpy as np


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

# ============================================================
# Euler-angle housekeeping
# ============================================================

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
# Quaternion & Euler axis
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


