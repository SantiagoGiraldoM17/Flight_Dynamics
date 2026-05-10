import numpy as np
import sys

SPEED_MULT = 1

from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog,
                              QVBoxLayout, QHBoxLayout, QListWidget,
                              QPushButton, QLabel)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

from rcam_model import rcam_derivatives, rcam_full
from flight_math import (
    build_dcm, terminator,
    angle2quaternion, quaternion_angle, quaternion_axis,
    compute_aero_angles,
)
from flight_gui import FlightDataViewer


# ═══════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════

def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(X0_9, U, dt=1.0, N=180, h0=3000.0):
    """RK4 integration of the 12-state RCAM model. Returns history dict.

    U may be a fixed (5,) array or a callable U(t) -> (5,) array for
    time-varying control schedules.
    """
    X12 = np.zeros(12)
    X12[:9] = X0_9
    X12[11] = -h0

    state_hist = np.zeros((N, 12))
    ctrl_hist  = np.zeros((N, 5))
    for i in range(N):
        state_hist[i] = X12
        u = U(i * dt) if callable(U) else U
        ctrl_hist[i]  = u
        X12 = rk4_step(rcam_full, X12, u, dt)

    times         = np.arange(N) * dt
    vel_body_hist = state_hist[:, 0:3]
    gyro_hist     = state_hist[:, 3:6]
    euler_rad     = state_hist[:, 6:9]
    euler_deg     = np.degrees(euler_rad)
    pos_hist      = state_hist[:, 9:12]
    alt_hist      = -state_hist[:, 11]

    vel_ned_hist    = np.zeros((N, 3))
    aero_hist       = np.zeros((N, 3))
    quat_hist       = np.zeros((N, 4))
    rot_angle_hist  = np.zeros(N)
    euler_axis_hist = np.zeros((N, 3))
    gimbal_hist     = np.zeros(N, dtype=bool)

    for i in range(N):
        ed = euler_deg[i]
        vb = vel_body_hist[i]
        vel_ned_hist[i] = build_dcm(ed, degrees=True) @ vb
        aero_hist[i]    = compute_aero_angles(ed, vb, vel_ned_hist[i])
        _, gimbal = terminator(euler_rad[i])
        gimbal_hist[i] = gimbal
        q  = angle2quaternion(ed[0], ed[1], ed[2], degrees=True)
        tr = quaternion_angle(q)
        quat_hist[i]       = q
        rot_angle_hist[i]  = np.degrees(tr)
        euler_axis_hist[i] = quaternion_axis(q, tr)

    return {
        'times': times, 'state': state_hist,
        'gyro': gyro_hist, 'euler': euler_deg,
        'vel_body': vel_body_hist, 'vel_ned': vel_ned_hist,
        'position': pos_hist[:, :2], 'pos_3d': pos_hist,
        'altitude': alt_hist, 'aero_angles': aero_hist,
        'quaternion': quat_hist, 'rot_angle': rot_angle_hist,
        'euler_axis': euler_axis_hist, 'gimbal_lock': gimbal_hist,
        'controls': ctrl_hist,
    }


# ═══════════════════════════════════════════════════════════
#  PSO Trim Finder
# ═══════════════════════════════════════════════════════════

def pso_trim(V_target=78.0, psi_target=np.pi/4,
             n_particles=80, n_iterations=400):
    """
    Particle Swarm Optimization for straight, level, steady trim.

    Searches for [alpha, dE, dTH] that minimise ||xdot||^2
    with uB = V*cos(alpha), wB = V*sin(alpha), vB=p=q=r=phi=0,
    theta=alpha (gamma=0), psi=psi_target.
    """
    dim = 3
    lb = np.array([-0.10, -0.436, 0.001])
    ub = np.array([ 0.35,  0.175, 1.000])

    def cost(params):
        alpha, dE, dTH = params
        uB = V_target * np.cos(alpha)
        wB = V_target * np.sin(alpha)
        X = np.array([uB, 0.0, wB, 0.0, 0.0, 0.0, 0.0, alpha, psi_target])
        U = np.array([0.0, dE, 0.0, dTH, dTH])
        xdot = rcam_derivatives(X, U)
        return np.sum(xdot**2)

    pos = lb + (ub - lb) * np.random.rand(n_particles, dim)
    vel = (ub - lb) * 0.1 * (np.random.rand(n_particles, dim) - 0.5)
    pb      = pos.copy()
    pb_cost = np.array([cost(p) for p in pos])
    gi      = np.argmin(pb_cost)
    gb      = pb[gi].copy()
    gb_cost = pb_cost[gi]

    w_max, w_min, c1, c2 = 0.9, 0.4, 2.0, 2.0

    for it in range(n_iterations):
        w = w_max - (w_max - w_min) * it / n_iterations
        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            vel[i] = (w * vel[i]
                      + c1 * r1 * (pb[i] - pos[i])
                      + c2 * r2 * (gb   - pos[i]))
            pos[i] = np.clip(pos[i] + vel[i], lb, ub)
            c = cost(pos[i])
            if c < pb_cost[i]:
                pb[i]      = pos[i].copy()
                pb_cost[i] = c
                if c < gb_cost:
                    gb      = pos[i].copy()
                    gb_cost = c
        if it % 100 == 0:
            print(f"  PSO iter {it:4d}  cost = {gb_cost:.4e}")

    print(f"  PSO iter {n_iterations:4d}  cost = {gb_cost:.4e}  (final)")
    return gb, gb_cost


# ═══════════════════════════════════════════════════════════
#  Scenario definitions
# ═══════════════════════════════════════════════════════════

def scenario_untrimmed():
    print("Running untrimmed flight scenario...\n")
    X0 = np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    U  = np.array([0.0, -0.1, 0.0, 0.08, 0.08])
    history = simulate(X0, U, dt=1.0, N=180, h0=3000.0)
    sf = history['state'][-1]
    print(f"  Final: uB={sf[0]:.2f}  wB={sf[2]:.2f}  "
          f"theta={np.degrees(sf[7]):.2f}deg  h={-sf[11]:.0f}m")
    return history


def scenario_aileron_pulse():
    """
    Same initial state as the untrimmed scenario.
    At t=30 s a +5 deg aileron step is applied for 2 s (t=30..32),
    then ailerons return to zero.

    Uses dt=0.1 s (N=1800) because the RCAM roll-damping eigenvalue is
    ~-9.2 rad/s (time constant 0.11 s).  RK4 requires |λ·dt| < 2.79,
    so dt must be < 0.30 s.  With dt=1 s the roll mode is numerically
    unstable and blows up the moment any aileron is applied.
    """
    print("Running aileron-pulse scenario...\n")
    print("  X0 = [85, 0, 0, 0, 0, 0, 0, 0.1, 0]")
    print("  U  = [0, -0.1, 0, 0.08, 0.08]  (baseline)")
    print("  Pulse: dA = +5 deg  at t=30 s  for 2 s")
    print("  dt = 0.1 s  (required for lateral-directional stability)\n")

    X0     = np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    U_base = np.array([0.0, -0.1, 0.0, 0.08, 0.08])
    dA_rad = 5.0 * np.pi / 180.0   # +5 deg in radians

    def U_schedule(t):
        u = U_base.copy()
        if 30.0 <= t < 32.0:
            u[0] = dA_rad
        return u

    history = simulate(X0, U_schedule, dt=0.1, N=1800, h0=3000.0)
    sf = history['state'][-1]
    print(f"  Final: uB={sf[0]:.2f}  wB={sf[2]:.2f}  "
          f"phi={np.degrees(sf[6]):.2f}deg  h={-sf[11]:.0f}m")
    return history


def scenario_aileron_engine_out():
    """
    Same baseline as scenario 2:  +5 deg aileron pulse at t=30 s for 2 s.
    At t=60 s the LEFT engine (dTH1) is shut down for the rest of the sim.

    The right engine alone produces an asymmetric thrust moment about the
    z-body axis (engine 2 is mounted on the right wing at +YAPT2),
    yawing the aircraft to the left.  Yaw drives sideslip, sideslip drives
    a rolling moment via the dihedral term in Cl_c, so the aircraft
    diverges in a coupled spiral / Dutch-roll mode.
    """
    print("Running aileron + engine-failure scenario...\n")
    print("  X0 = [85, 0, 0, 0, 0, 0, 0, 0.1, 0]")
    print("  U  = [0, -0.1, 0, 0.08, 0.08]  (baseline)")
    print("  Pulse:        dA   = +5 deg  at t=30..32 s")
    print("  Engine fail:  dTH2 = 0       at t>=60 s  (left engine, YAPT2=-7.94)\n")

    X0     = np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    U_base = np.array([0.0, -0.1, 0.0, 0.08, 0.08])
    dA_rad = 5.0 * np.pi / 180.0

    def U_schedule(t):
        u = U_base.copy()
        if 30.0 <= t < 32.0:
            u[0] = dA_rad           # aileron pulse
        if t >= 60.0:
            u[4] = 0.0              # LEFT engine off (dTH2 sits at YAPT2=-7.94)
        return u

    history = simulate(X0, U_schedule, dt=0.1, N=1800, h0=3000.0)
    sf = history['state'][-1]
    print(f"  Final: uB={sf[0]:.2f}  vB={sf[1]:.2f}  wB={sf[2]:.2f}  "
          f"phi={np.degrees(sf[6]):.2f}deg  psi={np.degrees(sf[8]):.2f}deg  "
          f"h={-sf[11]:.0f}m")
    return history


def scenario_trim_ne_78():
    V   = 78.0
    psi = np.pi / 4

    print(f"Finding trim: V={V} m/s, psi=45 deg (NE)...\n")
    params, cost_val = pso_trim(V_target=V, psi_target=psi)
    alpha, dE, dTH = params

    print(f"\n  -- Trim result (cost = {cost_val:.2e}) --")
    print(f"  alpha  = {np.degrees(alpha):+.4f} deg  ({alpha:+.6f} rad)")
    print(f"  dE     = {np.degrees(dE):+.4f} deg  ({dE:+.6f} rad)")
    print(f"  dTH    = {dTH:.6f}")
    print(f"  theta  = {np.degrees(alpha):+.4f} deg  (= alpha, level flight)")

    uB = V * np.cos(alpha)
    wB = V * np.sin(alpha)
    X0 = np.array([uB, 0.0, wB, 0.0, 0.0, 0.0, 0.0, alpha, psi])
    U  = np.array([0.0, dE, 0.0, dTH, dTH])

    xdot = rcam_derivatives(X0, U)
    print(f"  ||xdot|| = {np.linalg.norm(xdot):.2e}\n")

    print("Simulating from trim state (3 min, dt=1s)...")
    history = simulate(X0, U, dt=1.0, N=180, h0=3000.0)
    return history


SCENARIOS = [
    {
        "name": "1 — Untrimmed Flight",
        "desc": ("X0 = [85, 0, 0, 0, 0, 0, 0, 0.1, 0]\n"
                 "U  = [0, -0.1, 0, 0.08, 0.08]\n\n"
                 "Far-from-trim initial condition.\n"
                 "Phugoid oscillation and gradual descent."),
        "func": scenario_untrimmed,
    },
    {
        "name": "2 — Aileron Pulse (+5 deg, t=30..32 s)",
        "desc": ("Same initial state as Scenario 1.\n\n"
                 "At t = 30 s: dA = +5 deg\n"
                 "At t = 32 s: dA = 0  (ailerons neutral)\n\n"
                 "Shows roll response, Dutch-roll coupling,\n"
                 "and lateral-directional dynamics.\n"
                 "(dt=0.1 s for roll-mode stability)"),
        "func": scenario_aileron_pulse,
    },
    {
        "name": "3 — Aileron Pulse + Left Engine Failure",
        "desc": ("Baseline = Scenario 2.\n\n"
                 "At t = 30..32 s: dA = +5 deg pulse\n"
                 "At t = 60 s    : LEFT engine shut down\n"
                 "                 (dTH2 = 0, dTH1 = 0.08)\n\n"
                 "Asymmetric thrust → yaw left → sideslip\n"
                 "→ roll via dihedral. Aircraft enters\n"
                 "a coupled spiral / Dutch-roll divergence."),
        "func": scenario_aileron_engine_out,
    },
    {
        "name": "4 — Trimmed Level Flight NE @ 78 m/s",
        "desc": ("PSO finds equilibrium for:\n"
                 "  V = 78 m/s,  psi = 45 deg  (NE)\n"
                 "  Straight, level, steady flight\n"
                 "  gamma = 0,  phi = 0,  beta = 0\n\n"
                 "Optimises [alpha, dE, dTH] so xdot = 0."),
        "func": scenario_trim_ne_78,
    },
]


# ═══════════════════════════════════════════════════════════
#  Scenario selector dialog
# ═══════════════════════════════════════════════════════════

class ScenarioDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RCAM Simulation")
        self.setMinimumSize(480, 340)
        self.setStyleSheet("""
            QDialog      { background: #1e1e1e; }
            QLabel       { color: #e0e0e0; }
            QListWidget  { background: #2d2d2d; color: white;
                           border: 1px solid #555; font-size: 13px; }
            QListWidget::item          { padding: 6px; }
            QListWidget::item:selected { background: #0078d4; }
            QPushButton  { background: #0078d4; color: white; border: none;
                           padding: 8px 28px; font-size: 13px;
                           border-radius: 4px; }
            QPushButton:hover { background: #1a8ad4; }
        """)

        lay = QVBoxLayout(self)
        title = QLabel("Select Scenario")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(title)

        self.list = QListWidget()
        for s in SCENARIOS:
            self.list.addItem(s["name"])
        self.list.setCurrentRow(0)
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list)

        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet(
            "color: #aaa; background: #252525; padding: 8px; "
            "border: 1px solid #444; font-family: Consolas; font-size: 12px;")
        self._on_row(0)
        lay.addWidget(self.desc)

        row = QHBoxLayout()
        row.addStretch()
        btn = QPushButton("Run")
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        lay.addLayout(row)

        self.selected = 0

    def _on_row(self, r):
        if 0 <= r < len(SCENARIOS):
            self.desc.setText(SCENARIOS[r]["desc"])
            self.selected = r

    def accept(self):
        self.selected = self.list.currentRow()
        super().accept()


# ═══════════════════════════════════════════════════════════
#  GUI launcher
# ═══════════════════════════════════════════════════════════

def launch_gui(history):
    viewer = FlightDataViewer(
        history['euler'][0], history['vel_body'][0], history=history)

    win = QMainWindow()
    win.setWindowTitle("RCAM Flight Dynamics Simulation")
    win.resize(1200, 700)
    win.setCentralWidget(viewer)
    win.show()

    # Mutable state shared by all closures
    ctx       = {'history': history, 'N': len(history['times'])}
    idx       = [0]
    step_size = [1]
    timer     = QTimer()

    def _dt():
        h = ctx['history']
        return (h['times'][1] - h['times'][0]) if len(h['times']) > 1 else 1.0

    def _set_speed(mult):
        """Adjust timer interval and step size for the requested speed multiplier."""
        dt = _dt()
        # At <= 20x: keep step=1, slow/fast the timer interval.
        # Above 20x: pin interval at 50 ms and increase steps per tick.
        if mult <= 20.0:
            new_interval = max(16, int(dt * 1000.0 / mult))
            step_size[0] = 1
        else:
            new_interval = 50
            step_size[0] = max(1, round(mult * new_interval / (dt * 1000.0)))
        timer.setInterval(new_interval)

    def animate():
        h = ctx['history']
        i = idx[0]
        if i >= ctx['N']:
            timer.stop()
            return
        pe = h['pos_3d'][i]
        viewer.update_state(
            new_euler   = h['euler'][i],
            new_v_body  = h['vel_body'][i],
            v_ned       = h['vel_ned'][i],
            aero_angles = h['aero_angles'][i],
            quaternion  = h['quaternion'][i],
            rot_angle   = h['rot_angle'][i],
            euler_axis  = h['euler_axis'][i],
            position    = pe[:2],
            altitude    = -pe[2],
            gimbal_lock = h['gimbal_lock'][i],
            time        = h['times'][i],
            frame_idx   = i,
            controls    = h['controls'][i],
        )
        idx[0] = min(i + step_size[0], ctx['N'])

    def on_new_scenario():
        timer.stop()
        dlg = ScenarioDialog(win)
        if dlg.exec_() == QDialog.Accepted:
            sc = SCENARIOS[dlg.selected]
            print(f"\n{'='*50}")
            print(f" {sc['name']}")
            print(f"{'='*50}\n")
            new_hist = sc['func']()
            ctx['history'] = new_hist
            ctx['N']       = len(new_hist['times'])
            idx[0]         = 0
            viewer.load_history(new_hist)
        # Resume playback (whether new scenario or cancelled)
        _set_speed(viewer.speed_combo.currentData())
        timer.start()

    viewer.scenario_requested.connect(on_new_scenario)
    viewer.speed_changed.connect(_set_speed)

    timer.timeout.connect(animate)
    _set_speed(1.0)   # start at 1x real-time
    timer.start()

    win._timer  = timer
    win._viewer = viewer
    return win


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background: #1a1a1a; }")

    dlg = ScenarioDialog()
    if dlg.exec_() == QDialog.Accepted:
        sc = SCENARIOS[dlg.selected]
        print(f"\n{'='*50}")
        print(f" {sc['name']}")
        print(f"{'='*50}\n")
        history = sc['func']()
        print("\nLaunching GUI...\n")
        win = launch_gui(history)
        sys.exit(app.exec_())
    else:
        sys.exit(0)
