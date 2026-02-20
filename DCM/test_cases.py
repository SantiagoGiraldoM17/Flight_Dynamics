# Import both functions from your math file
from dcm import transform_flight_data, compute_aero_angles

def print_results(case_name, euler_in, v_body_in, euler_out, v_ned_out, alpha, beta, gamma):
    """Helper function to print the results neatly."""
    print(f"### {case_name} ###")
    print("INPUTS:")
    print(f"  Euler [Roll, Pitch, Yaw] : {euler_in} deg")
    print(f"  Body Velocity [u, v, w]  : {v_body_in} m/s")
    print("OUTPUTS (NED Frame):")
    print(f"  North Velocity (V_N)     : {v_ned_out[0]:.2f} m/s")
    print(f"  East Velocity  (V_E)     : {v_ned_out[1]:.2f} m/s")
    print(f"  Down Velocity  (V_D)     : {v_ned_out[2]:.2f} m/s")
    print("AERODYNAMIC ANGLES:")
    print(f"  Angle of Attack (Alpha)  : {alpha:.2f}°")
    print(f"  Sideslip Angle (Beta)    : {beta:.2f}°")
    print(f"  Flight Path / Climb (Gamma): {gamma:.2f}°")
    print("-" * 50 + "\n")

# ==========================================
# Case A: Straight-and-Level Flight
# ==========================================
euler_A = [0.0, 0.0, 45]
v_body_A = [100.0, 0.0, 0.0]

v_ned_A, euler_out_A = transform_flight_data(euler_A, v_body_A)
alpha_A, beta_A, gamma_A = compute_aero_angles(v_body_A, v_ned_A)

print_results("Case A: Straight-and-Level Flight", 
              euler_A, v_body_A, euler_out_A, v_ned_A, alpha_A, beta_A, gamma_A)


# ==========================================
# Case B: Aircraft Climb (Perfectly Coordinated)
# ==========================================
# Nose is pitched up 15 degrees, moving purely forward through the air.
euler_B = [0.0, 15.0, 0.0]
v_body_B = [100.0, 0.0, 0.0]

v_ned_B, euler_out_B = transform_flight_data(euler_B, v_body_B)
alpha_B, beta_B, gamma_B = compute_aero_angles(v_body_B, v_ned_B)

print_results("Case B: Aircraft Climb", 
              euler_B, v_body_B, euler_out_B, v_ned_B, alpha_B, beta_B, gamma_B)


# ==========================================
# Case C: Aircraft Turn (Uncoordinated with AoA)
# ==========================================
# Heading East, banked right 30 degrees, pitched up 5 degrees.
# Added v=5.0 (slipping right) and w=8.0 (generating Angle of Attack).
euler_C = [30.0, 5.0, 90.0]
v_body_C = [100.0, 5.0, 8.0]

v_ned_C, euler_out_C = transform_flight_data(euler_C, v_body_C)
alpha_C, beta_C, gamma_C = compute_aero_angles(v_body_C, v_ned_C)

print_results("Case C: Aircraft Turn", 
              euler_C, v_body_C, euler_out_C, v_ned_C, alpha_C, beta_C, gamma_C)