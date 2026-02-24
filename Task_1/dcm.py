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

def compute_aero_angles(euler_vector, v_body):
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
    phi, theta, psi = euler_vector
    
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
    theta_rad = np.radians(theta)  
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