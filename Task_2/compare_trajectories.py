"""
compare_trajectories.py
-----------------------
Compares the dead-reckoned (IMU-integrated) trajectory from flight_main.py
against the ground-truth positions in tello_ground_truth.csv.

Produces:
  1. 2-D top-down trajectory overlay  (X vs Y)
  2. Per-axis position vs time        (X, Y each in a subplot)
  3. Position error magnitude vs time
  4. Summary statistics printed to console
"""

import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

# ── Re-use the math helpers from the existing project ────────────────
from flight_math import (
    calculate_H, euler_rate, integrate, terminator,
    build_dcm, accel_body_to_ned, add_gravity,
)


# =====================================================================
# Data loading
# =====================================================================

def load_imu(filepath):
    """Load the IMU CSV (gyro + accel)."""
    times, gyro, accel = [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            gyro.append([float(row['gyro_p_rad_s']),
                         float(row['gyro_q_rad_s']),
                         float(row['gyro_r_rad_s'])])
            accel.append([float(row['accel_x_m_s2']),
                          float(row['accel_y_m_s2']),
                          float(row['accel_z_m_s2'])])
    return np.array(times), np.array(gyro), np.array(accel)


def load_ground_truth(filepath):
    """Load the ground-truth CSV."""
    times, x, y, z = [], [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            x.append(float(row['gt_pos_x_m']))
            y.append(float(row['gt_pos_y_m']))
            z.append(float(row['gt_pos_z_m']))
    return np.array(times), np.array(x), np.array(y), np.array(z)


# =====================================================================
# Dead-reckoning loop (mirrors flight_main.run_ahrs, position only)
# =====================================================================

def dead_reckon(times, gyro_data, accel_data):
    """
    Run the same AHRS/INS integration as flight_main.py
    and return position history (N, E) plus altitude.
    """
    N = len(times)

    euler_rad = np.array([0.0, 0.0, 0.0])
    vel_body  = np.array([0.0, 0.0, 0.0])
    pos_ned   = np.array([0.0, 0.0])
    altitude  = 0.0

    pos_hist = np.zeros((N, 2))
    alt_hist = np.zeros(N)

    for i in range(N):
        dt = (times[1] - times[0]) if i == 0 else (times[i] - times[i - 1])

        omega_B = gyro_data[i]
        accel_B = accel_data[i]

        # Angular rates → Euler angles
        H         = calculate_H(euler_rad[0], euler_rad[1])
        euler_dot = euler_rate(H, omega_B)
        euler_rad = integrate(euler_dot, euler_rad, dt)
        euler_rad, _ = terminator(euler_rad)
        euler_deg = np.degrees(euler_rad)

        # Body accel → NED velocity & position
        C_b_n     = build_dcm(euler_deg, degrees=True)
        accel_ned = accel_body_to_ned(C_b_n, accel_B)
        accel_ned = add_gravity(accel_ned)
        vel_body  = integrate(accel_ned, vel_body, dt)
        vel_ned   = C_b_n @ vel_body
        pos_ned   = integrate(vel_ned[:2], pos_ned, dt)
        altitude  = integrate(-vel_ned[2], altitude, dt)

        pos_hist[i] = pos_ned
        alt_hist[i] = altitude

    return pos_hist, alt_hist


# =====================================================================
# Plotting
# =====================================================================

def plot_comparison(t_gt, gt_x, gt_y, t_imu, est_x, est_y):
    """Create comparison plots."""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Trajectory Comparison — Dead Reckoning vs Ground Truth',
                 fontsize=15, fontweight='bold')

    # ── 1. Top-down 2-D trajectory ──
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(gt_x,  gt_y,  'b-',  linewidth=1.5, label='Ground Truth')
    ax1.plot(est_x, est_y, 'r--', linewidth=1.2, label='Dead Reckoning (IMU)')
    ax1.set_xlabel('X  (North)  [m]')
    ax1.set_ylabel('Y  (East)  [m]')
    ax1.set_title('Top-Down Trajectory')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, alpha=0.3)

    # ── 2. X position vs time ──
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_gt,  gt_x,  'b-',  linewidth=1.5, label='GT  X (North)')
    ax2.plot(t_imu, est_x, 'r--', linewidth=1.2, label='Est X (North)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('X Position [m]')
    ax2.set_title('X (North) Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3. Y position vs time ──
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_gt,  gt_y,  'b-',  linewidth=1.5, label='GT  Y (East)')
    ax3.plot(t_imu, est_y, 'r--', linewidth=1.2, label='Est Y (East)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Y Position [m]')
    ax3.set_title('Y (East) Position vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 4. Position error over time ──
    # Interpolate ground truth to match IMU timestamps
    gt_x_interp = np.interp(t_imu, t_gt, gt_x)
    gt_y_interp = np.interp(t_imu, t_gt, gt_y)
    err = np.sqrt((est_x - gt_x_interp)**2 + (est_y - gt_y_interp)**2)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t_imu, err, 'm-', linewidth=1.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Position Error [m]')
    ax4.set_title('Euclidean Position Error vs Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    imu_csv = os.path.join(script_dir, 'tello_imu_example.csv')
    gt_csv  = os.path.join(script_dir, 'tello_ground_truth.csv')

    print("Loading IMU data …")
    t_imu, gyro, accel = load_imu(imu_csv)

    print("Loading ground truth …")
    t_gt, gt_x, gt_y, gt_z = load_ground_truth(gt_csv)

    print(f"IMU samples : {len(t_imu)}   |   GT samples : {len(t_gt)}")
    print(f"IMU time    : {t_imu[0]:.2f} – {t_imu[-1]:.2f} s")
    print(f"GT  time    : {t_gt[0]:.2f} – {t_gt[-1]:.2f} s")

    print("\nRunning dead-reckoning integration …")
    pos_hist, alt_hist = dead_reckon(t_imu, gyro, accel)
    est_x = pos_hist[:, 0]   # North
    est_y = pos_hist[:, 1]   # East

    # ── Summary statistics ──
    gt_x_interp = np.interp(t_imu, t_gt, gt_x)
    gt_y_interp = np.interp(t_imu, t_gt, gt_y)
    err = np.sqrt((est_x - gt_x_interp)**2 + (est_y - gt_y_interp)**2)

    print("\n══════════════════════════════════════════")
    print("         Position Error Statistics        ")
    print("══════════════════════════════════════════")
    print(f"  Max error   : {err.max():.3f} m")
    print(f"  Mean error  : {err.mean():.3f} m")
    print(f"  Final error : {err[-1]:.3f} m")
    print(f"  RMS error   : {np.sqrt(np.mean(err**2)):.3f} m")
    print("══════════════════════════════════════════")

    print(f"\n  GT  final pos : ({gt_x[-1]:.3f}, {gt_y[-1]:.3f}) m")
    print(f"  Est final pos : ({est_x[-1]:.3f}, {est_y[-1]:.3f}) m")

    print("\nPlotting …")
    plot_comparison(t_gt, gt_x, gt_y, t_imu, est_x, est_y)
