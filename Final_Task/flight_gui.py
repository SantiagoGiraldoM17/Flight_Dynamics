import numpy as np
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
                              QLabel, QTabWidget, QSplitter, QPushButton, QComboBox)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (QPainter, QColor, QPen, QBrush, QPainterPath,
                          QFont, QPolygonF)
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from flight_math import transform_flight_data, compute_aero_angles, build_dcm

pg.setConfigOptions(antialias=True, background='#1a1a1a', foreground='w')


# ═══════════════════════════════════════════════════════════
#  Attitude Indicator — custom QPainter widget
# ═══════════════════════════════════════════════════════════

class AttitudeIndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roll_deg  = 0.0
        self.pitch_deg = 0.0
        self.setMinimumSize(200, 200)

    def update_angles(self, roll_deg, pitch_deg):
        self.roll_deg  = roll_deg
        self.pitch_deg = pitch_deg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r  = min(w, h) * 0.43          # instrument radius in px
        cx = w / 2
        cy = h / 2 + r * 0.08          # shift down slightly to leave room for title

        painter.translate(cx, cy)

        # ── Circular clip ──────────────────────────────────
        clip = QPainterPath()
        clip.addEllipse(QRectF(-r, -r, 2 * r, 2 * r))
        painter.setClipPath(clip)

        # ── Roll + pitch transform for sky/ground/ladder ──
        painter.save()
        painter.rotate(-self.roll_deg)          # negative: CCW screen = CW roll left

        pitch_offset = (self.pitch_deg / 45.0) * r   # px per 45°

        # Sky
        painter.setBrush(QBrush(QColor('#1f77b4')))
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(-2 * r, -2 * r + pitch_offset, 4 * r, 2 * r))

        # Ground
        painter.setBrush(QBrush(QColor('#8c564b')))
        painter.drawRect(QRectF(-2 * r, pitch_offset, 4 * r, 2 * r))

        # Horizon line
        painter.setPen(QPen(QColor('white'), 2))
        painter.drawLine(QPointF(-r, pitch_offset), QPointF(r, pitch_offset))

        # Pitch ladder (±10, ±20, ±30 degrees)
        for p in range(-30, 31, 10):
            if p == 0:
                continue
            y_pos = pitch_offset - (p / 45.0) * r
            if -r * 1.05 < y_pos < r * 1.05:
                half_w = r * (0.20 if p % 20 == 0 else 0.12)
                painter.setPen(QPen(QColor('white'), 1.5))
                painter.drawLine(QPointF(-half_w, y_pos), QPointF(half_w, y_pos))

        # Bank-angle tick marks (rotate each tick individually)
        bank_marks = [0, 10, 20, 30, 45, 60, -10, -20, -30, -45, -60]
        for angle in bank_marks:
            painter.save()
            painter.rotate(angle)
            inner = 0.85 if abs(angle) in (0, 30, 60) else 0.92
            painter.setPen(QPen(QColor('white'), 2))
            painter.drawLine(QPointF(0, -inner * r), QPointF(0, -r))
            painter.restore()

        painter.restore()   # end roll context

        # ── Static elements (drawn after removing roll transform) ──
        
        # Bezel ring
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor('dimgray'), 6))
        painter.drawEllipse(QRectF(-r, -r, 2 * r, 2 * r))
        
        # Orange aircraft reference crosshair
        painter.setPen(QPen(QColor('orange'), 2))
        painter.drawLine(QPointF(-0.40 * r, 0), QPointF(-0.10 * r, 0))
        painter.drawLine(QPointF( 0.10 * r, 0), QPointF( 0.40 * r, 0))
        painter.setBrush(QBrush(QColor('orange')))
        painter.setPen(Qt.NoPen)
        dot_r = 0.025 * r
        painter.drawEllipse(QRectF(-dot_r, -dot_r, 2 * dot_r, 2 * dot_r))

        # ── Remove clip for bezel/pointer ──────────────────
        painter.setClipping(False)

        # Roll pointer triangle (static, at top)
        tri = QPolygonF([
            QPointF(-0.06 * r, -(r + 0.08 * r)),
            QPointF( 0.06 * r, -(r + 0.08 * r)),
            QPointF(0,          -(r - 0.08 * r))
        ])
        painter.setBrush(QBrush(QColor('orange')))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(tri)



        # Title
        painter.translate(-cx, -cy)   # back to widget origin
        painter.setPen(QPen(QColor('white')))
        painter.setFont(QFont('Arial', 9, QFont.Bold))
        painter.drawText(QRectF(0, 2, w, 20), Qt.AlignHCenter, "Attitude Indicator")

        painter.end()


# ═══════════════════════════════════════════════════════════
#  Heading Indicator — custom QPainter widget
# ═══════════════════════════════════════════════════════════

class HeadingIndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.yaw_deg = 0.0
        self.setMinimumSize(200, 200)

    def update_yaw(self, yaw_deg):
        self.yaw_deg = yaw_deg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r  = min(w, h) * 0.43
        cx = w / 2
        cy = h / 2 + r * 0.08

        painter.translate(cx, cy)

        # Black face
        painter.setBrush(QBrush(QColor('#111111')))
        painter.setPen(QPen(QColor('dimgray'), 6))
        painter.drawEllipse(QRectF(-r, -r, 2 * r, 2 * r))

        # Ticks and cardinal labels
        cardinals = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
        for angle in range(0, 360, 30):
            rad   = np.radians(angle)
            # Aviation: 0°=North=up → screen sx = sin, sy = -cos (up is -Y on screen)
            sx    = np.sin(rad)
            sy    = -np.cos(rad)
            in_r  = 0.85 if angle % 90 == 0 else 0.90

            painter.setPen(QPen(QColor('white'), 2))
            painter.drawLine(
                QPointF(sx * in_r * r, sy * in_r * r),
                QPointF(sx * r,        sy * r)
            )

            if angle % 90 == 0:
                tx = sx * 0.70 * r
                ty = sy * 0.70 * r
                painter.setPen(QPen(QColor('white')))
                painter.setFont(QFont('Arial', max(8, int(r * 0.17)), QFont.Bold))
                sz = r * 0.30
                painter.drawText(QRectF(tx - sz / 2, ty - sz / 2, sz, sz),
                                 Qt.AlignCenter, cardinals[angle])

        # Heading arrow
        yaw_rad   = np.radians(self.yaw_deg)
        ax        = np.sin(yaw_rad)
        ay        = -np.cos(yaw_rad)
        body_end  = 0.50 * r          # shaft end (before arrowhead)
        head_len  = 0.22 * r
        head_half = 0.10 * r

        # Shaft
        painter.setPen(QPen(QColor('orange'), 4))
        painter.drawLine(QPointF(0, 0), QPointF(ax * body_end, ay * body_end))

        # Arrowhead
        tip = np.array([ax, ay]) * (body_end + head_len)
        base = np.array([ax, ay]) * body_end
        perp = np.array([-ay, ax])   # perpendicular unit vector
        lp = base + perp * head_half
        rp = base - perp * head_half

        head = QPolygonF([
            QPointF(tip[0], tip[1]),
            QPointF(lp[0],  lp[1]),
            QPointF(rp[0],  rp[1])
        ])
        painter.setBrush(QBrush(QColor('orange')))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(head)

        # Center pin
        pin_r = r * 0.05
        painter.setBrush(QBrush(QColor('gray')))
        painter.drawEllipse(QRectF(-pin_r, -pin_r, 2 * pin_r, 2 * pin_r))

        # Title
        painter.translate(-cx, -cy)
        painter.setPen(QPen(QColor('white')))
        painter.setFont(QFont('Arial', 9, QFont.Bold))
        painter.drawText(QRectF(0, 2, w, 20), Qt.AlignHCenter, "Heading Indicator")

        painter.end()


# ═══════════════════════════════════════════════════════════
#  Helper — static NED axis line for GLViewWidget
# ═══════════════════════════════════════════════════════════

def _gl_axis(tip_gl, color_rgba, width=2.5):
    """Returns a GLLinePlotItem from origin to tip_gl."""
    pts = np.array([[0, 0, 0], tip_gl], dtype=np.float32)
    return gl.GLLinePlotItem(pos=pts, color=color_rgba, width=width,
                              antialias=True, mode='lines')


# ═══════════════════════════════════════════════════════════
#  Matplotlib-style 3D axis box for the trajectory view
# ═══════════════════════════════════════════════════════════

def _nice_ticks(lo, hi, max_ticks=5):
    """Return a list of evenly-spaced, human-friendly tick values in [lo, hi]."""
    rng = hi - lo
    if rng < 1e-12:
        return [round(lo, 6)]
    step_raw = rng / max_ticks
    mag = 10 ** np.floor(np.log10(abs(step_raw)))
    step = min([1, 2, 5, 10], key=lambda c: abs(c * mag - step_raw)) * mag
    first = np.ceil(lo / step) * step
    ticks, v = [], first
    while v <= hi + step * 1e-6:
        ticks.append(float(round(v / step) * step))
        v += step
    return ticks


class GL3DAxesItem:
    """
    Draws a matplotlib-style 3D axis box inside a GLViewWidget.

    Three grid faces are drawn (bottom E-N, left wall N-Alt, back wall E-Alt),
    a bounding-box outline, axis labels, and numeric tick labels on the edges.

    Coordinate convention: GL x = East, GL y = North, GL z = Altitude (up).
    Call rebuild(x_range, y_range, z_range) whenever the data bounds change.
    """

    _GRID_COLOR = (0.22, 0.22, 0.28, 0.85)
    _BOX_COLOR  = (0.42, 0.42, 0.48, 1.00)

    def __init__(self, view):
        self._view  = view
        self._items = []

    # ── public ──────────────────────────────────────────────

    def remove_all(self):
        for item in self._items:
            try:
                self._view.removeItem(item)
            except Exception:
                pass
        self._items.clear()

    def rebuild(self, x_range, y_range, z_range):
        """Rebuild all GL items for new axis ranges."""
        self.remove_all()
        x0, x1 = x_range
        y0, y1 = y_range
        z0, z1 = z_range
        xt = _nice_ticks(x0, x1)
        yt = _nice_ticks(y0, y1)
        zt = _nice_ticks(z0, z1)
        self._draw_box(x0, x1, y0, y1, z0, z1)
        self._draw_grids(x0, x1, y0, y1, z0, z1, xt, yt, zt)
        self._draw_labels(x0, x1, y0, y1, z0, z1, xt, yt, zt)

    # ── private helpers ──────────────────────────────────────

    def _add_lines(self, pts_list, color, width=1):
        pts = np.array(pts_list, dtype=np.float32)
        item = gl.GLLinePlotItem(pos=pts, color=color, width=width,
                                  antialias=True, mode='lines')
        self._view.addItem(item)
        self._items.append(item)

    def _add_text(self, pos, text, color):
        try:
            item = gl.GLTextItem(pos=np.array(pos, dtype=float),
                                  text=text, color=color)
            self._view.addItem(item)
            self._items.append(item)
        except Exception:
            pass   # GLTextItem not available in this build

    def _draw_box(self, x0, x1, y0, y1, z0, z1):
        c = self._BOX_COLOR
        edges = [
            # bottom ring
            [x0,y0,z0],[x1,y0,z0],  [x1,y0,z0],[x1,y1,z0],
            [x1,y1,z0],[x0,y1,z0],  [x0,y1,z0],[x0,y0,z0],
            # top ring
            [x0,y0,z1],[x1,y0,z1],  [x1,y0,z1],[x1,y1,z1],
            [x1,y1,z1],[x0,y1,z1],  [x0,y1,z1],[x0,y0,z1],
            # verticals
            [x0,y0,z0],[x0,y0,z1],  [x1,y0,z0],[x1,y0,z1],
            [x0,y1,z0],[x0,y1,z1],  [x1,y1,z0],[x1,y1,z1],
        ]
        self._add_lines(edges, c, width=1)

    def _draw_grids(self, x0, x1, y0, y1, z0, z1, xt, yt, zt):
        c = self._GRID_COLOR

        # Floor  (z = z0) : East–North grid
        floor = []
        for x in xt:
            floor += [[x, y0, z0], [x, y1, z0]]
        for y in yt:
            floor += [[x0, y, z0], [x1, y, z0]]
        if floor:
            self._add_lines(floor, c)

        # Left wall  (x = x0) : North–Altitude grid
        left = []
        for y in yt:
            left += [[x0, y, z0], [x0, y, z1]]
        for z in zt:
            left += [[x0, y0, z], [x0, y1, z]]
        if left:
            self._add_lines(left, c)

        # Back wall  (y = y1) : East–Altitude grid
        back = []
        for x in xt:
            back += [[x, y1, z0], [x, y1, z1]]
        for z in zt:
            back += [[x0, y1, z], [x1, y1, z]]
        if back:
            self._add_lines(back, c)

    def _draw_labels(self, x0, x1, y0, y1, z0, z1, xt, yt, zt):
        ph = (x1 - x0 + y1 - y0) * 0.07   # horizontal padding
        pv = max((z1 - z0) * 0.07, ph * 0.3)

        # Tick values — East (front-bottom edge, y = y0)
        for x in xt:
            self._add_text([x, y0 - ph, z0 - pv],
                           f"{x:.3g}", (220, 110, 110, 255))

        # Tick values — North (left-bottom edge, x = x0)
        for y in yt:
            self._add_text([x0 - ph, y, z0 - pv],
                           f"{y:.3g}", (110, 220, 110, 255))

        # Tick values — Altitude (left-back vertical edge, x = x0, y = y1)
        for z in zt:
            self._add_text([x0 - ph, y1, z],
                           f"{z:.3g}", (110, 165, 255, 255))

        # Axis name labels
        self._add_text([(x0+x1)/2, y0 - ph*2.8, z0 - pv],
                       "East (m)",  (255,  80,  80, 255))
        self._add_text([x0 - ph*2.8, (y0+y1)/2, z0 - pv],
                       "North (m)", ( 80, 255,  80, 255))
        self._add_text([x0 - ph*2.8, y1, (z0+z1)/2],
                       "Alt (m)",   ( 80, 150, 255, 255))


# ═══════════════════════════════════════════════════════════
#  Aircraft surface-panel geometry (body frame)
#  Body frame: +x = nose, +y = right, +z = down
#  Each panel: 4 corners as (4,3) array, wound for 2 triangles
#    0=front-root  1=back-root  2=front-tip  3=back-tip
# ═══════════════════════════════════════════════════════════

_PANEL_FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

_AC_PANELS = {
    # Main wings — slight aft sweep, span to ±2.5
    'wing_L':   np.array([[ 0.55, -0.10,  0.0],
                           [-0.60, -0.10,  0.0],
                           [ 0.25, -2.50,  0.0],
                           [-0.85, -2.50,  0.0]], dtype=np.float32),
    'wing_R':   np.array([[ 0.55,  0.10,  0.0],
                           [-0.60,  0.10,  0.0],
                           [ 0.25,  2.50,  0.0],
                           [-0.85,  2.50,  0.0]], dtype=np.float32),
    # Horizontal stabilizer — at tail, span to ±1.1
    'hstab_L':  np.array([[-1.40,  0.0,  0.0],
                           [-2.00,  0.0,  0.0],
                           [-1.50, -1.10, 0.0],
                           [-2.00, -1.10, 0.0]], dtype=np.float32),
    'hstab_R':  np.array([[-1.40,  0.0,  0.0],
                           [-2.00,  0.0,  0.0],
                           [-1.50,  1.10, 0.0],
                           [-2.00,  1.10, 0.0]], dtype=np.float32),
    # Vertical stabilizer — stands upward (-z) from tail
    'vstab':    np.array([[-1.50,  0.0,  0.0],
                           [-2.00,  0.0,  0.0],
                           [-1.65,  0.0, -1.05],
                           [-2.00,  0.0, -1.05]], dtype=np.float32),
}

# RGBA face colors (two triangles per panel share the same color)
_AC_PANEL_COLORS = {
    'wing_L':  (0.72, 0.72, 0.82, 0.88),
    'wing_R':  (0.72, 0.72, 0.82, 0.88),
    'hstab_L': (0.60, 0.60, 0.72, 0.85),
    'hstab_R': (0.60, 0.60, 0.72, 0.85),
    'vstab':   (0.68, 0.62, 0.72, 0.85),
}


# ═══════════════════════════════════════════════════════════
#  Main viewer widget
# ═══════════════════════════════════════════════════════════

class FlightDataViewer(QWidget):

    scenario_requested = pyqtSignal()
    speed_changed      = pyqtSignal(float)

    def __init__(self, euler_in, v_body_in, history=None):
        super().__init__()

        self.euler  = np.asarray(euler_in,  dtype=float)
        self.v_body = np.asarray(v_body_in, dtype=float)
        self.history = history

        # Derived state
        self.v_ned, _     = transform_flight_data(self.euler, self.v_body)
        self.aero_angles  = compute_aero_angles(self.euler, self.v_body)
        self.C_b_n        = build_dcm(self.euler)

        # Extended AHRS state (defaults)
        self.quaternion  = np.array([1.0, 0.0, 0.0, 0.0])
        self.rot_angle   = 0.0
        self.euler_axis  = np.zeros(3)
        self.position    = np.zeros(2)
        self.altitude    = 0.0
        self.gimbal_lock = False
        self.time        = 0.0
        self.current_idx = 0
        self.controls    = np.zeros(5)   # [dA, dE, dR, dTH1, dTH2]

        self._setup_ui()
        self._update_3d_aircraft()

    # ─────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; background: #1a1a1a; }
            QTabBar::tab { background: #2d2d2d; color: white; padding: 6px 12px; }
            QTabBar::tab:selected { background: #444; }
        """)
        root.addWidget(self.tabs)

        # Tab order matches state vector: [uB,vB,wB] → [p,q,r] → [φ,θ,ψ] → [X,Y,h]
        self._build_instruments_tab()
        self._build_body_vel_tab()
        self._build_rates_tab()
        self._build_euler_tab()
        self._build_position_tab()
        self._build_trajectory_tab()

        # ── Control bar: speed selector + New Scenario button ──
        ctrl_bar = QWidget()
        ctrl_bar.setStyleSheet("background: #252525; border-top: 1px solid #444;")
        ctrl_bar.setFixedHeight(38)
        cl = QHBoxLayout(ctrl_bar)
        cl.setContentsMargins(10, 4, 10, 4)
        cl.setSpacing(10)

        spd_lbl = QLabel("Speed:")
        spd_lbl.setStyleSheet("color: #ccc; font-size: 12px;")
        cl.addWidget(spd_lbl)

        self.speed_combo = QComboBox()
        self.speed_combo.setStyleSheet("""
            QComboBox { background: #2d2d2d; color: white;
                        border: 1px solid #555; padding: 2px 8px;
                        min-width: 80px; font-size: 12px; }
            QComboBox::drop-down { border: 0; }
            QComboBox QAbstractItemView { background: #2d2d2d; color: white; }
        """)
        for label, value in [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0),
                              ("5x", 5.0), ("10x", 10.0), ("20x", 20.0),
                              ("50x", 50.0)]:
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(1)   # default 1x real-time
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        cl.addWidget(self.speed_combo)

        cl.addStretch()

        self.btn_new_scenario = QPushButton("New Scenario")
        self.btn_new_scenario.setStyleSheet("""
            QPushButton { background: #0078d4; color: white; border: none;
                          padding: 4px 18px; font-size: 12px; border-radius: 3px; }
            QPushButton:hover { background: #1a8ad4; }
        """)
        self.btn_new_scenario.clicked.connect(self.scenario_requested.emit)
        cl.addWidget(self.btn_new_scenario)

        root.addWidget(ctrl_bar)

    def _on_speed_changed(self, _index):
        self.speed_changed.emit(float(self.speed_combo.currentData()))

    # ── Tab 1: Instruments ──────────────────────────────

    def _build_instruments_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #1a1a1a;")
        self.tabs.addTab(tab, "  Instruments  ")

        outer = QHBoxLayout(tab)
        outer.setContentsMargins(4, 4, 4, 4)

        # ── Left: text readout ──
        self.data_label = QLabel()
        self.data_label.setFont(QFont('Courier New', 10))
        self.data_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.data_label.setStyleSheet(
            "color: #e0e0e0; background: #111; padding: 10px; border-right: 1px solid #333;")
        self.data_label.setMinimumWidth(265)
        self.data_label.setMaximumWidth(290)
        self._refresh_text()
        outer.addWidget(self.data_label)

        # ── Right: instruments (AI, HI, 3D) ──
        instr = QWidget()
        grid = QGridLayout(instr)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        self.ai_widget = AttitudeIndicatorWidget()
        self.ai_widget.update_angles(self.euler[0], self.euler[1])
        self.ai_widget.setStyleSheet("background: #1a1a1a;")
        grid.addWidget(self.ai_widget, 0, 0)

        self.hi_widget = HeadingIndicatorWidget()
        self.hi_widget.update_yaw(self.euler[2])
        self.hi_widget.setStyleSheet("background: #1a1a1a;")
        grid.addWidget(self.hi_widget, 1, 0)

        # 3D aircraft (GL widget, spans both rows)
        self.gl_aircraft = gl.GLViewWidget()
        self.gl_aircraft.setBackgroundColor('#1a1a1a')
        self.gl_aircraft.setCameraPosition(distance=12, elevation=25, azimuth=45)
        self._setup_aircraft_gl()
        grid.addWidget(self.gl_aircraft, 0, 1, 2, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        outer.addWidget(instr, stretch=1)

    def _setup_aircraft_gl(self):
        """Populate the 3D aircraft GLViewWidget with static axes and dynamic wireframe."""
        # NED reference axes mapped to GL: (GL x=E, GL y=N, GL z=-D)
        # — matches the trajectory view's convention so East shows on the right
        self.gl_aircraft.addItem(_gl_axis([2.5,  0,   0],  (0, 1, 0, 1)))   # E green
        self.gl_aircraft.addItem(_gl_axis([0,  2.5,   0],  (1, 0, 0, 1)))   # N red
        self.gl_aircraft.addItem(_gl_axis([0,    0, -2.5], (0, 0, 1, 1)))   # D blue

        # Axis labels
        for pos, text, color in [
            ((2.8,  0,   0),  'E', (80,  255, 80,  255)),
            ((0,  2.8,   0),  'N', (255, 80,  80,  255)),
            ((0,    0, -2.8), 'D', (80,  80,  255, 255)),
        ]:
            try:
                lbl = gl.GLTextItem(pos=np.array(pos, dtype=float),
                                    text=text, color=color)
                self.gl_aircraft.addItem(lbl)
            except Exception:
                pass   # GLTextItem unavailable in this build

        # Reference grid (XY = North-East ground plane)
        grid = gl.GLGridItem()
        grid.setSize(6, 6)
        grid.setSpacing(1, 1)
        self.gl_aircraft.addItem(grid)

        # Dynamic aircraft wireframe (fuselage, spine to vert-tail)
        zero2 = np.zeros((2, 3), dtype=np.float32)
        ac_color = (0.95, 0.95, 0.95, 1.0)
        self.ac_fuselage = gl.GLLinePlotItem(pos=zero2, color=ac_color, width=3,
                                              antialias=True, mode='lines')
        self.ac_tail     = gl.GLLinePlotItem(pos=zero2, color=ac_color, width=3,
                                              antialias=True, mode='lines')
        self.ac_velocity = gl.GLLinePlotItem(pos=zero2, color=(1, 0, 1, 1), width=2,
                                              antialias=True, mode='lines')
        for item in (self.ac_fuselage, self.ac_tail, self.ac_velocity):
            self.gl_aircraft.addItem(item)

        # Surface panels: wings, horizontal & vertical stabilizers
        self.ac_panels = {}
        for name, rgba in _AC_PANEL_COLORS.items():
            fc = np.array([rgba, rgba], dtype=np.float32)   # 2 faces, same color
            item = gl.GLMeshItem(
                vertexes=np.zeros((4, 3), dtype=np.float32),
                faces=_PANEL_FACES,
                faceColors=fc,
                smooth=False,
                drawEdges=True,
                edgeColor=(0.9, 0.9, 0.9, 0.6),
            )
            item._fc = fc   # keep a reference to avoid re-allocating
            self.gl_aircraft.addItem(item)
            self.ac_panels[name] = item

    # ── Tabs 2–4: Time-series ────────────────────────────

    def _build_euler_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #1a1a1a;")
        self.tabs.addTab(tab, "  Euler Angles  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        t_max = self.history['times'][-1] if self.history is not None else 60.0
        labels = ['φ — Roll (°)', 'θ — Pitch (°)', 'ψ — Yaw (°)']
        colors = ['#E53935', '#1E88E5', '#43A047']

        self.euler_plots  = []
        self.euler_curves = []
        for j, (lbl, col) in enumerate(zip(labels, colors)):
            pw = self._make_plot_widget(lbl, col, t_max,
                                         show_xaxis=(j == 2))
            self.euler_plots.append(pw)
            self.euler_curves.append(pw.plot(pen=pg.mkPen(color=col, width=1)))
            layout.addWidget(pw)

        self._link_x_axes(self.euler_plots)

    def _build_rates_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #1a1a1a;")
        self.tabs.addTab(tab, "  Euler Rates  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        t_max = self.history['times'][-1] if self.history is not None else 60.0
        labels = ['p — Roll rate (rad/s)', 'q — Pitch rate (rad/s)', 'r — Yaw rate (rad/s)']
        colors = ['#E53935', '#1E88E5', '#43A047']

        self.rates_plots  = []
        self.rates_curves = []
        for j, (lbl, col) in enumerate(zip(labels, colors)):
            pw = self._make_plot_widget(lbl, col, t_max,
                                         show_xaxis=(j == 2))
            self.rates_plots.append(pw)
            self.rates_curves.append(pw.plot(pen=pg.mkPen(color=col, width=1)))
            layout.addWidget(pw)

        self._link_x_axes(self.rates_plots)

    def _build_body_vel_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #1a1a1a;")
        self.tabs.addTab(tab, "  Body Velocity  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        t_max = self.history['times'][-1] if self.history is not None else 60.0
        labels = ['uB — Forward (m/s)', 'vB — Lateral (m/s)', 'wB — Down (m/s)']
        colors = ['#D32F2F', '#388E3C', '#1565C0']

        self.body_vel_plots  = []
        self.body_vel_curves = []
        for j, (lbl, col) in enumerate(zip(labels, colors)):
            pw = self._make_plot_widget(lbl, col, t_max,
                                         show_xaxis=(j == 2))
            self.body_vel_plots.append(pw)
            self.body_vel_curves.append(pw.plot(pen=pg.mkPen(color=col, width=1)))
            layout.addWidget(pw)

        self._link_x_axes(self.body_vel_plots)

    def _build_position_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #1a1a1a;")
        self.tabs.addTab(tab, "  Position  ")
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        t_max = self.history['times'][-1] if self.history is not None else 60.0
        labels = ['X — North (m)', 'Y — East (m)', 'h — Altitude (m)']
        colors = ['#E53935', '#43A047', '#1E88E5']

        self.pos_plots  = []
        self.pos_curves = []
        for j, (lbl, col) in enumerate(zip(labels, colors)):
            pw = self._make_plot_widget(lbl, col, t_max,
                                         show_xaxis=(j == 2))
            self.pos_plots.append(pw)
            self.pos_curves.append(pw.plot(pen=pg.mkPen(color=col, width=1)))
            layout.addWidget(pw)

        self._link_x_axes(self.pos_plots)

    # ── Tab 5: Trajectory ───────────────────────────────

    def _build_trajectory_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #1a1a1a;")
        self.tabs.addTab(tab, "  Trajectory  ")
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        # 2D: East vs Altitude
        pw2d = pg.PlotWidget()
        pw2d.setBackground('#1a1a1a')
        pw2d.setLabel('bottom', 'East (m)',  color='white')
        pw2d.setLabel('left',   'North (m)', color='white')
        pw2d.setTitle("2D Trajectory  (North vs East)", color='white', size='10pt')
        pw2d.showGrid(x=True, y=True, alpha=0.25)
        pw2d.enableAutoRange()
        self.traj_curve_2d   = pw2d.plot(pen=pg.mkPen(color='#2196F3', width=1))
        self.traj_start_2d   = pw2d.plot(symbol='s', symbolBrush='#4CAF50',
                                          symbolSize=10, pen=None)
        self.traj_current_2d = pw2d.plot(symbol='o', symbolBrush='#FF5722',
                                          symbolSize=8,  pen=None)

        # ── 3D trajectory (GL) ───────────────────────────────
        self.gl_traj = gl.GLViewWidget()
        self.gl_traj.setBackgroundColor('#1a1a1a')
        self.gl_traj.setCameraPosition(distance=5, elevation=30, azimuth=45)

        # Matplotlib-style axis box — rebuilt whenever data bounds change.
        # Coordinate mapping: GL x = East, GL y = North, GL z = Altitude (up)
        self._traj_axes       = GL3DAxesItem(self.gl_traj)
        self._traj_last_ticks = None   # cache: skip rebuild when ticks unchanged

        # Trajectory path line + start/current markers
        zero2 = np.zeros((2, 3), dtype=np.float32)
        self.traj_line_3d    = gl.GLLinePlotItem(pos=zero2,
                                                  color=(0.13, 0.59, 0.95, 1),
                                                  width=4, antialias=True,
                                                  mode='line_strip')
        self.traj_start_3d   = gl.GLScatterPlotItem(pos=np.zeros((1, 3)),
                                                     color=(0.3, 0.69, 0.31, 1),
                                                     size=14, pxMode=True)
        self.traj_current_3d = gl.GLScatterPlotItem(pos=np.zeros((1, 3)),
                                                     color=(1, 0.34, 0.13, 1),
                                                     size=12, pxMode=True)
        for item in (self.traj_line_3d, self.traj_start_3d, self.traj_current_3d):
            self.gl_traj.addItem(item)

        # QSplitter gives a reliable 50/50 split and a draggable handle
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #333; width: 4px; }")
        splitter.addWidget(pw2d)
        splitter.addWidget(self.gl_traj)
        splitter.setSizes([600, 600])
        layout.addWidget(splitter)

    # ─────────────────────────────────────────────────────
    # Plot-widget factory helpers
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _make_plot_widget(ylabel, color, t_max, show_xaxis=False):
        pw = pg.PlotWidget()
        pw.setBackground('#1a1a1a')
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setLabel('left', ylabel, color=color)
        pw.setXRange(0, t_max, padding=0)
        pw.enableAutoRange(axis=pg.ViewBox.YAxis)
        if show_xaxis:
            pw.setLabel('bottom', 'Time (s)', color='white')
        else:
            pw.getAxis('bottom').setStyle(showValues=False)
        return pw

    @staticmethod
    def _link_x_axes(plots):
        for pw in plots[1:]:
            pw.setXLink(plots[0])

    # ─────────────────────────────────────────────────────
    # Text readout
    # ─────────────────────────────────────────────────────

    def _refresh_text(self):
        gl_flag = "  ⚠ GIMBAL LOCK" if self.gimbal_lock else ""
        text = (
            f"t = {self.time:.2f} s{gl_flag}\n"
            f"{'─' * 32}\n"
            f"EULER ANGLES:\n"
            f"  Roll  (φ): {self.euler[0]:+8.2f}°\n"
            f"  Pitch (θ): {self.euler[1]:+8.2f}°\n"
            f"  Yaw   (ψ): {self.euler[2]:+8.2f}°\n\n"
            f"BODY VELOCITY:\n"
            f"  u: {self.v_body[0]:+8.3f} m/s\n"
            f"  v: {self.v_body[1]:+8.3f} m/s\n"
            f"  w: {self.v_body[2]:+8.3f} m/s\n\n"
            f"NED VELOCITY:\n"
            f"  V_N: {self.v_ned[0]:+8.3f} m/s\n"
            f"  V_E: {self.v_ned[1]:+8.3f} m/s\n"
            f"  V_D: {self.v_ned[2]:+8.3f} m/s\n\n"
            f"POSITION & ALTITUDE:\n"
            f"  P_N: {self.position[0]:+8.2f} m\n"
            f"  P_E: {self.position[1]:+8.2f} m\n"
            f"  h  : {self.altitude:+8.2f} m\n\n"
            f"CONTROLS:\n"
            f"  Aileron  (dA): {np.degrees(self.controls[0]):+7.2f}°\n"
            f"  Elevator (dE): {np.degrees(self.controls[1]):+7.2f}°\n"
            f"  Rudder   (dR): {np.degrees(self.controls[2]):+7.2f}°\n"
            f"  Throttle L  : {self.controls[3]:7.4f}\n"
            f"  Throttle R  : {self.controls[4]:7.4f}\n\n"
            f"AERO ANGLES:\n"
            f"  α: {self.aero_angles[0]:+7.2f}°\n"
            f"  β: {self.aero_angles[1]:+7.2f}°\n"
            f"  γ: {self.aero_angles[2]:+7.2f}°"
        )
        self.data_label.setText(text)

    # ─────────────────────────────────────────────────────
    # 3D aircraft update
    # ─────────────────────────────────────────────────────

    def _update_3d_aircraft(self):
        # ── Fuselage & spine wireframe ──────────────────────────
        # Body-frame key points: nose, tail, vert-tail-tip
        pts_body = np.array([
            [ 2.0,  0.0,  0.0],   # 0 Nose
            [-2.0,  0.0,  0.0],   # 1 Tail
            [-2.0,  0.0, -1.0],   # 2 Vert Tail tip (up in body = -Z)
        ]).T  # (3, 3)

        pts_ned = self.C_b_n @ pts_body
        # NED → GL mapping: (GL x=E, GL y=N, GL z=-D)
        def gl_pt(i):
            return [pts_ned[1, i], pts_ned[0, i], -pts_ned[2, i]]

        self.ac_fuselage.setData(pos=np.array([gl_pt(0), gl_pt(1)], dtype=np.float32))
        self.ac_tail.setData(    pos=np.array([gl_pt(1), gl_pt(2)], dtype=np.float32))

        # ── Surface panels ──────────────────────────────────────
        for name, corners_body in _AC_PANELS.items():
            # corners_body: (4, 3) in body frame
            pts = (self.C_b_n @ corners_body.T).T   # (4, 3) NED [N, E, D]
            verts = np.empty_like(pts)
            verts[:, 0] = pts[:, 1]   # E → GL x
            verts[:, 1] = pts[:, 0]   # N → GL y
            verts[:, 2] = -pts[:, 2]  # -D → GL z (up)
            item = self.ac_panels[name]
            item.setMeshData(vertexes=verts.astype(np.float32),
                             faces=_PANEL_FACES,
                             faceColors=item._fc)

        # ── Velocity vector ─────────────────────────────────────
        v_norm = np.linalg.norm(self.v_ned)
        if v_norm > 1e-3:
            v_draw = (self.v_ned / v_norm) * 2 if v_norm > 2 else self.v_ned
            vn, ve, vd = v_draw
            self.ac_velocity.setData(
                pos=np.array([[0, 0, 0], [ve, vn, -vd]], dtype=np.float32))

    # ─────────────────────────────────────────────────────
    # Time-series update
    # ─────────────────────────────────────────────────────

    def _update_timeseries(self):
        if self.history is None:
            return

        h = self.history
        i = self.current_idx + 1
        t = h['times'][:i]

        for j, curve in enumerate(self.euler_curves):
            curve.setData(x=t, y=h['euler'][:i, j])

        for j, curve in enumerate(self.rates_curves):
            curve.setData(x=t, y=h['gyro'][:i, j])

        for j, curve in enumerate(self.body_vel_curves):
            curve.setData(x=t, y=h['vel_body'][:i, j])

        if hasattr(self, 'pos_curves') and 'pos_3d' in h:
            for j, curve in enumerate(self.pos_curves):
                if j < 2:
                    curve.setData(x=t, y=h['pos_3d'][:i, j])
                else:
                    curve.setData(x=t, y=-h['pos_3d'][:i, 2])

        pos_e = h['position'][:i, 1]
        pos_n = h['position'][:i, 0]
        alt   = h['altitude'][:i]

        self.traj_curve_2d.setData(x=pos_e, y=pos_n)
        self.traj_start_2d.setData(  x=[pos_e[0]],  y=[pos_n[0]])
        self.traj_current_2d.setData(x=[pos_e[-1]], y=[pos_n[-1]])

        if len(pos_e) >= 2:
            # 3D GL: x=East, y=North, z=Altitude (up)
            pts3 = np.column_stack(
                [pos_e, pos_n, alt]).astype(np.float32)
            self.traj_line_3d.setData(   pos=pts3)
            self.traj_start_3d.setData(  pos=pts3[[0]])
            self.traj_current_3d.setData(pos=pts3[[-1]])

            # ── Rebuild axis box when tick values change ─────────
            def _prange(vals, frac=0.12, minspan=0.5):
                lo, hi = float(vals.min()), float(vals.max())
                pad = max(hi - lo, minspan) * frac
                return lo - pad, hi + pad

            xr = _prange(pos_e)
            yr = _prange(pos_n)
            zr = _prange(alt)
            new_ticks = (tuple(_nice_ticks(*xr)),
                         tuple(_nice_ticks(*yr)),
                         tuple(_nice_ticks(*zr)))
            if new_ticks != self._traj_last_ticks:
                self._traj_axes.rebuild(xr, yr, zr)
                self._traj_last_ticks = new_ticks

            # Auto-fit camera so the trajectory stays visible at any scale
            extent = max(float(np.ptp(pts3, axis=0).max()), 0.5)
            center = pts3.mean(axis=0)
            self.gl_traj.setCameraPosition(
                pos=pg.Vector(center[0], center[1], center[2]),
                distance=extent * 3.5,
            )

    # ─────────────────────────────────────────────────────
    # Load new scenario (called when user picks a new scenario)
    # ─────────────────────────────────────────────────────

    def load_history(self, history):
        """Replace the history dict and reset all plots for a new scenario."""
        self.history     = history
        self.current_idx = 0
        t_max = history['times'][-1]

        # Update time axis on all time-series plots
        for pw in (self.euler_plots + self.rates_plots
                   + self.body_vel_plots + self.pos_plots):
            pw.setXRange(0, t_max, padding=0)

        # Clear all time-series curves
        for curves in (self.euler_curves, self.rates_curves,
                       self.body_vel_curves, self.pos_curves):
            for c in curves:
                c.setData(x=[], y=[])

        # Clear trajectory
        self.traj_curve_2d.setData(x=[], y=[])
        self.traj_start_2d.setData(x=[], y=[])
        self.traj_current_2d.setData(x=[], y=[])
        zero2 = np.zeros((2, 3), dtype=np.float32)
        self.traj_line_3d.setData(pos=zero2)
        self.traj_start_3d.setData(pos=np.zeros((1, 3), dtype=np.float32))
        self.traj_current_3d.setData(pos=np.zeros((1, 3), dtype=np.float32))
        self._traj_last_ticks = None

    # ─────────────────────────────────────────────────────
    # Public update entry point (called every frame)
    # ─────────────────────────────────────────────────────

    def update_state(self, new_euler, new_v_body, v_ned=None,
                     aero_angles=None, quaternion=None, rot_angle=None,
                     euler_axis=None, position=None, altitude=None,
                     gimbal_lock=False, time=0.0, frame_idx=0,
                     controls=None):
        self.euler       = np.asarray(new_euler)
        self.v_body      = np.asarray(new_v_body)
        self.time        = time
        self.gimbal_lock = gimbal_lock
        self.current_idx = frame_idx

        if v_ned is not None:
            self.v_ned = np.asarray(v_ned)
        else:
            self.v_ned, _ = transform_flight_data(self.euler, self.v_body)

        if aero_angles is not None:
            self.aero_angles = np.asarray(aero_angles)
        else:
            self.aero_angles = compute_aero_angles(self.euler, self.v_body)

        if quaternion is not None: self.quaternion = np.asarray(quaternion)
        if rot_angle  is not None: self.rot_angle  = rot_angle
        if euler_axis is not None: self.euler_axis = np.asarray(euler_axis)
        if position   is not None: self.position   = np.asarray(position)
        if altitude   is not None: self.altitude   = altitude
        if controls   is not None: self.controls   = np.asarray(controls)

        self.C_b_n = build_dcm(self.euler)

        self._refresh_text()
        self.ai_widget.update_angles(self.euler[0], self.euler[1])
        self.hi_widget.update_yaw(self.euler[2])
        self._update_3d_aircraft()
        self._update_timeseries()
