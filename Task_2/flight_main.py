import numpy as np
import csv
import sys

# ── Playback speed multiplier (1 = real-time, 3 = 3× faster, etc.) ──
SPEED_MULT = 3

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

from flight_math import (
    calculate_H, euler_rate, integrate, terminator,
    build_dcm, accel_body_to_ned, add_gravity,
    angle2quaternion, quaternion_angle, quaternion_axis,
    compute_aero_angles
)
from flight_gui import FlightDataViewer


def load_csv(filepath):
    """
    Reads the Tello IMU CSV file.

    Returns:
        times     : 1-D array of time stamps (s)
        gyro_data : N×3 array of [p, q, r] (rad/s)
        accel_data: N×3 array of [ax, ay, az] (m/s²)
    """
    times = []
    gyro_data = []
    accel_data = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            gyro_data.append([
                float(row['gyro_p_rad_s']),
                float(row['gyro_q_rad_s']),
                float(row['gyro_r_rad_s'])
            ])
            accel_data.append([
                float(row['accel_x_m_s2']),
                float(row['accel_y_m_s2']),
                float(row['accel_z_m_s2'])
            ])

    return np.array(times), np.array(gyro_data), np.array(accel_data)


def run_ahrs(csv_file):
    """
    Runs the full AHRS/INS loop:
    1. Reads IMU data from CSV
    2. Pre-computes all state histories
    3. Launches the GUI with real-time animation playback
    """
    # ── Load data ───────────────────────────────────────────
    times, gyro_data, accel_data = load_csv(csv_file)
    N = len(times)

    # ── Initial conditions (all zeros) ──────────────────────
    euler_rad = np.array([0.0, 0.0, 0.0])      # φ, θ, ψ in radians
    vel_body  = np.array([0.0, 0.0, 0.0])       # u, v, w in m/s
    vel_ned   = np.array([0.0, 0.0, 0.0])       # V_N, V_E, V_D
    pos_ned   = np.array([0.0, 0.0])             # P_N, P_E
    altitude  = 0.0                               # h (metres)

    # ── History arrays ──────────────────────────────────────
    euler_hist      = np.zeros((N, 3))
    vel_body_hist   = np.zeros((N, 3))
    vel_ned_hist    = np.zeros((N, 3))
    pos_hist        = np.zeros((N, 2))
    alt_hist        = np.zeros(N)
    quat_hist       = np.zeros((N, 4))
    rot_angle_hist  = np.zeros(N)
    euler_axis_hist = np.zeros((N, 3))
    aero_hist       = np.zeros((N, 3))
    gimbal_hist     = np.zeros(N, dtype=bool)

    # ── Main loop ───────────────────────────────────────────
    for i in range(N):
        dt = (times[1] - times[0]) if i == 0 else (times[i] - times[i - 1])

        omega_B = gyro_data[i]     # [p, q, r] rad/s
        accel_B = accel_data[i]    # [ax, ay, az] m/s²

        # LEFT BRANCH: Angular rates → Euler angles
        H         = calculate_H(euler_rad[0], euler_rad[1])
        euler_dot = euler_rate(H, omega_B)
        euler_rad = integrate(euler_dot, euler_rad, dt)
        euler_rad, gimbal = terminator(euler_rad)
        euler_deg = np.degrees(euler_rad)

        # TOP BRANCH: Body accel → NED velocity & position
        C_b_n        = build_dcm(euler_deg, degrees=True)
        accel_ned    = accel_body_to_ned(C_b_n, accel_B)
        accel_ned    = add_gravity(accel_ned)
        vel_body     = integrate(accel_ned, vel_body, dt)
        vel_ned      = C_b_n @ vel_body
        pos_ned      = integrate(vel_ned[:2], pos_ned, dt)
        altitude     = integrate(-vel_ned[2], altitude, dt)

        # RIGHT BRANCH: Quaternion & Euler axis
        q         = angle2quaternion(euler_deg[0], euler_deg[1], euler_deg[2], degrees=True)
        theta_rot = quaternion_angle(q)
        e_hat     = quaternion_axis(q, theta_rot)

        # Aerodynamic angles
        aero = compute_aero_angles(euler_deg, vel_body, v_ned=vel_ned)

        # Store
        euler_hist[i]       = euler_deg
        vel_body_hist[i]    = vel_body
        vel_ned_hist[i]     = vel_ned
        pos_hist[i]         = pos_ned
        alt_hist[i]         = altitude
        quat_hist[i]        = q
        rot_angle_hist[i]   = np.degrees(theta_rot)
        euler_axis_hist[i]  = e_hat
        aero_hist[i]        = aero
        gimbal_hist[i]      = gimbal

    history = {
        'times':       times,
        'gyro':        gyro_data,
        'euler':       euler_hist,
        'vel_body':    vel_body_hist,
        'vel_ned':     vel_ned_hist,
        'position':    pos_hist,
        'altitude':    alt_hist,
        'quaternion':  quat_hist,
        'rot_angle':   rot_angle_hist,
        'euler_axis':  euler_axis_hist,
        'aero_angles': aero_hist,
        'gimbal_lock': gimbal_hist,
    }

    launch_gui(history)


def launch_gui(history):
    """Creates the Qt window and animates through pre-computed AHRS history."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background: #1a1a1a; }")

    euler_0  = history['euler'][0]
    v_body_0 = history['vel_body'][0]

    viewer = FlightDataViewer(euler_0, v_body_0, history=history)

    window = QMainWindow()
    window.setWindowTitle("AHRS — Attitude & Navigation Reference System")
    window.resize(1200, 700)
    window.setCentralWidget(viewer)
    window.show()

    N = len(history['times'])
    idx = [0]

    FRAME_INTERVAL_MS = 50   # ~20 FPS display
    dt_data   = history['times'][1] - history['times'][0] if N > 1 else 0.02
    STEP_SIZE = max(1, int(SPEED_MULT * FRAME_INTERVAL_MS / 1000 / dt_data))

    timer = QTimer()

    def animate():
        i = idx[0]
        if i >= N:
            timer.stop()
            return

        viewer.update_state(
            new_euler    = history['euler'][i],
            new_v_body   = history['vel_body'][i],
            v_ned        = history['vel_ned'][i],
            aero_angles  = history['aero_angles'][i],
            quaternion   = history['quaternion'][i],
            rot_angle    = history['rot_angle'][i],
            euler_axis   = history['euler_axis'][i],
            position     = history['position'][i],
            altitude     = history['altitude'][i],
            gimbal_lock  = history['gimbal_lock'][i],
            time         = history['times'][i],
            frame_idx    = i,
        )
        idx[0] = min(idx[0] + STEP_SIZE, N)

    timer.timeout.connect(animate)
    timer.start(FRAME_INTERVAL_MS)

    sys.exit(app.exec_())


if __name__ == '__main__':
    import os

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, 'tello_imu_example.csv')
    csv_path    = sys.argv[1] if len(sys.argv) > 1 else default_csv

    print(f"Running AHRS with: {csv_path}")
    run_ahrs(csv_path)
