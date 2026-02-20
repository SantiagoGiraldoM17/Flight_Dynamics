import numpy as np

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
    
    # Unpack the input vectors
    phi, theta, psi = euler_vector
    
    
    # Convert angles to radians

    phi = np.radians(phi)
    theta = np.radians(theta)
    psi = np.radians(psi)
    
    # Pre-compute sines and cosines
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    # Construct the Direction Cosine Matrix (Body to NED)
    C11 = c_theta * c_psi
    C12 = s_phi * s_theta * c_psi - c_phi * s_psi
    C13 = c_phi * s_theta * c_psi + s_phi * s_psi
    
    C21 = c_theta * s_psi
    C22 = s_phi * s_theta * s_psi + c_phi * c_psi
    C23 = c_phi * s_theta * s_psi - s_phi * c_psi
    
    C31 = -s_theta
    C32 = s_phi * c_theta
    C33 = c_phi * c_theta
    
    C_b_n = np.array([
        [C11, C12, C13],
        [C21, C22, C23],
        [C31, C32, C33]
    ])
    
    # Convert body velocity to column vector and multiply
    v_body = np.array(v_body_vector).reshape(3, 1)
    v_ned = np.dot(C_b_n, v_body)
    
    # Return flattened NED velocity vector and the Euler angles vector
    return v_ned.flatten(), euler_vector

def compute_aero_angles(v_body, v_ned):
    """
    Computes aerodynamic and flight path angles.
    
    Inputs:
    v_body : Velocity vector in body frame [u, v, w]
    v_ned  : Velocity vector in NED frame [V_N, V_E, V_D]
    
    Outputs:
    alpha : Angle of Attack (degrees)
    beta  : Sideslip Angle (degrees)
    gamma : Climb/Flight Path Angle (degrees)
    """
    
    u, v, w = v_body
    v_n, v_e, v_d = v_ned
    
    # Calculate total airspeed magnitude
    V_tot = np.linalg.norm(v_body)
    
    # Handle stationary case to prevent division by zero
    if V_tot == 0.0:
        return 0.0, 0.0, 0.0
        
    # Angle of Attack (Alpha)
    alpha_rad = np.arctan2(w, u)
    
    # Sideslip Angle (Beta)
    beta_rad = np.arcsin(v / V_tot)
    
    # Climb Angle (Gamma)
    # Using -v_d because negative Down means we are climbing
    gamma_rad = np.arcsin(-v_d / V_tot)
    
    # Convert radians to degrees for output
    alpha = np.degrees(alpha_rad)
    beta = np.degrees(beta_rad)
    gamma = np.degrees(gamma_rad)
    
    return alpha, beta, gamma