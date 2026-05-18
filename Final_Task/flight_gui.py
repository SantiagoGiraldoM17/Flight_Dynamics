import os
import struct
import colorsys
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
#  Aircraft 3D model configuration
# ═══════════════════════════════════════════════════════════
#
# Point _AIRCRAFT_STL_PATH at a binary .stl OR a Wavefront .obj file
# next to flight_gui.py and it is auto-loaded on startup. The loader
# dispatches on the file extension. If the file is missing or fails
# to parse, the code falls back to the hand-coded wireframe.
#
# Color support:
#   .stl  - VisCAM/SolidView per-face extension (RGB555 in attribute
#           byte) or Materialise 'COLOR=' header extension.
#   .obj  - reads diffuse Kd from any .mtl file referenced by 'mtllib'.
# If no embedded colors are found, the mesh is rendered as flat
# _AIRCRAFT_COLOR.
#
# The model is centred, scaled to fit AIRCRAFT_TARGET_SIZE, then
# rotated by AIRCRAFT_ORIENT to put it in body-frame convention
# (+x=nose, +y=right wing, +z=down). Tweak AIRCRAFT_ORIENT if the
# model appears upside-down, sideways or mirrored.
#
# Per-frame cost is just a 4x4 matrix update — independent of how
# many triangles the mesh has — because the rotation is applied as
# a GPU model-matrix transform, not by re-uploading vertices.
#
_AIRCRAFT_STL_PATH    = os.path.join(os.path.dirname(__file__), '737.obj')
_AIRCRAFT_TARGET_SIZE = 4.0          # span in GL units (axes are ~2.5)
_AIRCRAFT_COLOR       = (0.78, 0.78, 0.85, 1.0)

# Fine-tune position offset (body frame: +X=nose, +Y=right, +Z=down).
# Auto-centering handles most models out of the box; nudge here if you
# want the belly resting on the grid (try (0, 0, -0.3)) or the model
# raised "in flight" above the grid (try (0, 0, -0.6)).
_AIRCRAFT_OFFSET      = (0.0, 0.0, 0.0)

# Synthesize a distinct color per material when the .mtl is uninformative
# (e.g. PBR/textured Blender exports where every Kd is the same placeholder
# gray and the real color lives in PNG textures we cannot render). Set to
# False to always use whatever Kd the .mtl gives, even if everything ends
# up gray.
_AIRCRAFT_AUTO_PALETTE = True

# Orientation matrix: maps STL model axes → body frame axes.
# Each ROW is a body-frame axis expressed in STL coordinates.
#   Row 0 -> body +X (nose forward)
#   Row 1 -> body +Y (right wing)
#   Row 2 -> body +Z (down, belly)
#
# Default assumes STL convention "+Z = fuselage length, +Y = up,
# +X = wingspan" (common from many CAD exports). If the model looks
# wrong on first launch, try one of the variants in _ORIENT_PRESETS
# below or just edit the matrix entries directly (use 0, 1, -1 only).
_AIRCRAFT_ORIENT = np.array([
[0,0,1],[1,0,0],[0,-1,0]  # body +Z (down)        = STL -Y  (i.e. STL +Y is up)
], dtype=np.float32)

# Common alternates to try if the default looks wrong.
# Copy one of these into _AIRCRAFT_ORIENT.
_ORIENT_PRESETS = {
    # STL already in body frame (+X=nose, +Y=right, +Z=down)
    'body_native':   [[1,0,0],[0,1,0],[0,0,1]],
    # STL +X=nose, +Y=right, +Z=up   (just flip Z)
    'x_nose_z_up':   [[1,0,0],[0,1,0],[0,0,-1]],
    # STL +Y=nose, +X=right, +Z=up
    'y_nose_z_up':   [[0,1,0],[1,0,0],[0,0,-1]],
    # STL +Z=nose, +Y=up, +X=right wing  (default chosen here)
    'z_nose_y_up':   [[0,0,1],[1,0,0],[0,-1,0]],
    # STL +Z=tail, +Y=up, +X=right wing  (nose-flipped variant)
    'neg_z_nose':    [[0,0,-1],[1,0,0],[0,-1,0]],
    # STL +Z=nose, +X=up, +Y=right wing
    'z_nose_x_up':   [[0,0,1],[0,1,0],[-1,0,0]],
}


def _load_stl_binary(path):
    """Read a binary STL file and return (verts, faces, n_tri, face_colors, vertex_colors).

    vertex_colors is always None for STL (the format has no per-vertex colour
    extension); included only so the tuple shape matches _load_obj's return.

    face_colors is an (M, 4) float32 RGBA array if the STL embeds colors via
    either the VisCAM/SolidView per-face extension (attribute byte: bit 15 =
    'valid' flag, bits 10..14 = R, 5..9 = G, 0..4 = B as RGB555) or the
    Materialise Magics global-color header extension ('COLOR=' + 4 RGBA
    bytes). If neither is present, face_colors is None and the caller
    should fall back to a flat color.

    Vertices are not deduplicated - pyqtgraph's GLMeshItem handles redundant
    vertices fine, and skipping the dedup keeps load time well under a
    second even for ~500k-triangle models.
    """
    with open(path, 'rb') as f:
        header = f.read(80)                          # 80-byte header
        n_tri  = struct.unpack('<I', f.read(4))[0]   # triangle count
        rec = np.dtype([
            ('normal', '<f4', 3),
            ('v0',     '<f4', 3),
            ('v1',     '<f4', 3),
            ('v2',     '<f4', 3),
            ('attr',   '<u2'),
        ])
        data = np.fromfile(f, dtype=rec, count=n_tri)

    verts = np.stack([data['v0'], data['v1'], data['v2']],
                      axis=1).reshape(-1, 3).astype(np.float32)
    faces = np.arange(3 * n_tri, dtype=np.uint32).reshape(-1, 3)

    # ── Per-face color extraction (VisCAM / SolidView extension) ────
    attr  = data['attr']
    valid = (attr & 0x8000) != 0
    face_colors = None
    if valid.any():
        r = ((attr >> 10) & 0x1F).astype(np.float32) / 31.0
        g = ((attr >>  5) & 0x1F).astype(np.float32) / 31.0
        b = ( attr        & 0x1F).astype(np.float32) / 31.0
        default_rgb = np.array(_AIRCRAFT_COLOR[:3], dtype=np.float32)
        rgb = np.where(valid[:, None],
                       np.stack([r, g, b], axis=1),
                       default_rgb)
        face_colors = np.concatenate(
            [rgb, np.ones((n_tri, 1), dtype=np.float32)], axis=1)
    else:
        # ── Materialise Magics global color in the header ───────────
        idx = header.find(b'COLOR=')
        if idx >= 0 and idx + 10 <= 80:
            rgba = (np.frombuffer(header[idx+6:idx+10], dtype=np.uint8)
                      .astype(np.float32) / 255.0)
            face_colors = np.tile(rgba, (n_tri, 1))

    return verts, faces, n_tri, face_colors, None


def _load_mtl(path):
    """Parse a Wavefront .mtl file. Returns {material_name: (r, g, b, a)}.

    Only the diffuse color (Kd) and opacity (d / Tr) are extracted - we
    do not have texture support in pyqtgraph's GL, so map_Kd etc. are
    ignored. Missing / unparseable file -> empty dict.
    """
    materials = {}
    cur_name = [None]
    cur_kd   = [(0.78, 0.78, 0.85)]
    cur_d    = [1.0]

    def _commit():
        if cur_name[0] is not None:
            r, g, b = cur_kd[0]
            materials[cur_name[0]] = (r, g, b, cur_d[0])

    try:
        with open(path, 'r', errors='replace') as f:
            for raw in f:
                parts = raw.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue
                kw = parts[0]
                if kw == 'newmtl':
                    _commit()
                    cur_name[0] = ' '.join(parts[1:])
                    cur_kd[0]   = (0.78, 0.78, 0.85)
                    cur_d[0]    = 1.0
                elif kw == 'Kd' and len(parts) >= 4:
                    cur_kd[0] = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif kw == 'd' and len(parts) >= 2:
                    cur_d[0] = float(parts[1])
                elif kw == 'Tr' and len(parts) >= 2:        # transparency
                    cur_d[0] = 1.0 - float(parts[1])
        _commit()
    except OSError:
        pass
    return materials


def _load_obj(path):
    """Read a Wavefront .obj file (+ any referenced .mtl).

    Returns (verts, faces, n_tri, face_colors, vertex_colors).
        - vertex_colors (N,4) is populated when the OBJ uses the inline
          'v x y z r g b' extension (produced by bake_obj_texture.py)
          and takes precedence over face_colors at render time.
        - face_colors (M,4) is built per-face from each face's
          material's Kd if 'usemtl' / 'mtllib' were used.
        - Either or both can be None if no colour info is present.

    Notes:
        - Triangulates n-gons via a fan from the first vertex of each face.
        - Handles 1-based and negative face indices.
        - Vertex normals (vn) and texture coords (vt) are parsed but
          discarded; pyqtgraph computes its own normals, and there is
          no live-texture support (use bake_obj_texture.py instead).
    """
    obj_dir      = os.path.dirname(path)
    raw_verts    = []                             # list of [x, y, z]
    raw_v_colors = []                             # parallel: [r,g,b] or None
    faces_idx    = []                             # list of (v0, v1, v2)
    face_mat     = []                             # parallel to faces_idx
    materials    = {}                             # name -> (r,g,b,a)
    cur_mat      = None

    with open(path, 'r', errors='replace') as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            kw = parts[0]
            if kw == 'v':
                # 'v x y z'  or  'v x y z r g b'  (inline vertex color
                # extension - produced by bake_obj_texture.py).
                raw_verts.append([float(parts[1]), float(parts[2]),
                                   float(parts[3])])
                if len(parts) >= 7:
                    raw_v_colors.append([float(parts[4]), float(parts[5]),
                                          float(parts[6])])
                else:
                    raw_v_colors.append(None)
            elif kw == 'f':
                # 'f v[/vt[/vn]] v[/vt[/vn]] ...'  - any polygon size
                idx = []
                n_v = len(raw_verts)
                for tok in parts[1:]:
                    v = int(tok.split('/')[0])
                    if v < 0:
                        v = n_v + v        # negative: count from end
                    else:
                        v = v - 1          # OBJ is 1-based
                    idx.append(v)
                # Fan triangulation
                for i in range(1, len(idx) - 1):
                    faces_idx.append((idx[0], idx[i], idx[i+1]))
                    face_mat.append(cur_mat)
            elif kw == 'mtllib':
                # Resolve the referenced .mtl with three fallbacks, because
                # downloaded OBJs are frequently renamed without updating
                # the embedded mtllib directive:
                #   1) the exact name in the OBJ                  (3d-model.mtl)
                #   2) <obj_basename>.mtl   next to the OBJ       (737.mtl)
                #   3) the single .mtl file in the same directory (if exactly one)
                referenced  = ' '.join(parts[1:])
                obj_base    = os.path.splitext(os.path.basename(path))[0]
                candidates  = [
                    os.path.join(obj_dir, referenced),
                    os.path.join(obj_dir, obj_base + '.mtl'),
                ]
                same_dir_mtls = [
                    f for f in os.listdir(obj_dir) if f.lower().endswith('.mtl')
                ] if os.path.isdir(obj_dir) else []
                if len(same_dir_mtls) == 1:
                    candidates.append(os.path.join(obj_dir, same_dir_mtls[0]))

                for mtl_path in candidates:
                    if os.path.isfile(mtl_path):
                        loaded = _load_mtl(mtl_path)
                        if loaded:
                            materials.update(loaded)
                            if os.path.basename(mtl_path) != referenced:
                                print(f"[flight_gui] OBJ referenced '{referenced}' "
                                      f"but used '{os.path.basename(mtl_path)}' instead.")
                            break
                else:
                    print(f"[flight_gui] OBJ referenced '{referenced}' but no "
                          f".mtl was found in {obj_dir!r}.")
            elif kw == 'usemtl':
                cur_mat = ' '.join(parts[1:])

    verts = np.asarray(raw_verts, dtype=np.float32)
    faces = np.asarray(faces_idx, dtype=np.uint32)
    n_tri = len(faces)

    # Build per-face colors if any usemtl was active AND the .mtl
    # actually defined that material.
    face_colors = None
    used_mats   = sorted({m for m in face_mat if m})
    if used_mats and materials:
        default = np.array(_AIRCRAFT_COLOR, dtype=np.float32)
        fc      = np.tile(default, (n_tri, 1))
        for i, m in enumerate(face_mat):
            if m and m in materials:
                fc[i] = materials[m]
        face_colors = fc

    # ── Auto-palette fallback ──────────────────────────────────
    # If the .mtl is uninformative (PBR/textured exports where every
    # Kd is the same placeholder gray), synthesize a distinct hue per
    # material so the user actually sees a multi-color aircraft.
    if (_AIRCRAFT_AUTO_PALETTE and len(used_mats) >= 3
            and face_colors is not None):
        n_unique = len(np.unique(np.round(face_colors[:, :3], 3), axis=0))
        if n_unique <= 2:
            palette = {}
            n = len(used_mats)
            for i, name in enumerate(used_mats):
                # HSV cycle, muted saturation/value so it looks plane-y
                hue = (i / n) * 0.85          # avoid wrap to red
                r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.85)
                palette[name] = (r, g, b, 1.0)
            fc = np.tile(np.array(_AIRCRAFT_COLOR, dtype=np.float32),
                         (n_tri, 1))
            for i, m in enumerate(face_mat):
                if m in palette:
                    fc[i] = palette[m]
            face_colors = fc
            print(f"[flight_gui] .mtl had only {n_unique} unique color(s) "
                  f"across {n} materials (textured / placeholder Kd); "
                  f"auto-coloring with distinct hues per material.")

    # ── Inline vertex colours ('v x y z r g b') ────────────────
    # Produced by bake_obj_texture.py. These win over face colours.
    vertex_colors = None
    if any(c is not None for c in raw_v_colors):
        default = list(_AIRCRAFT_COLOR[:3])
        rgb     = np.array(
            [c if c is not None else default for c in raw_v_colors],
            dtype=np.float32)
        vertex_colors = np.concatenate(
            [rgb, np.ones((len(rgb), 1), dtype=np.float32)], axis=1)

    return verts, faces, n_tri, face_colors, vertex_colors


def _load_model(path):
    """Dispatch to the right loader based on file extension.

    Returns (verts, faces, n_tri, face_colors)  -- the same 4-tuple for
    every format, so the caller does not need to care which one was used.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.stl':
        return _load_stl_binary(path)
    if ext == '.obj':
        return _load_obj(path)
    raise ValueError(f"Unsupported aircraft model format: {ext!r} (use .stl or .obj)")


def _normalize_aircraft_mesh(verts):
    """Centre on bbox midpoint, scale to AIRCRAFT_TARGET_SIZE, rotate by
    AIRCRAFT_ORIENT to align with body-frame convention, then re-centre
    vertically on the fuselage centerline (median Z) so a tall vertical
    stabilizer does not push the visible fuselage below the grid.

    Final axis convention after this call:
        +X = nose forward,  +Y = right wing,  +Z = down (belly)
    """
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    centre   = 0.5 * (bbox_min + bbox_max)
    verts    = verts - centre
    extent   = float((bbox_max - bbox_min).max())
    if extent > 0:
        verts = verts * (_AIRCRAFT_TARGET_SIZE / extent)
    verts = verts @ _AIRCRAFT_ORIENT.T               # to body frame

    # ── Re-centre vertical axis on the fuselage line ────────────────
    # For any normal aircraft most vertices live in the fuselage tube,
    # so the median Z lands close to the centerline. The bbox midpoint
    # would be biased upward by the tip of the vertical stabilizer.
    verts[:, 2] -= float(np.median(verts[:, 2]))

    # Add the user-tweakable fine-tune offset (default: zero).
    verts = verts + np.asarray(_AIRCRAFT_OFFSET, dtype=np.float32)

    return verts.astype(np.float32)


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
        """Populate the 3D aircraft GLViewWidget with static axes and the
        dynamic aircraft (STL model if available, wireframe otherwise)."""
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

        # Velocity vector (always drawn, both STL and wireframe paths)
        zero2 = np.zeros((2, 3), dtype=np.float32)
        self.ac_velocity = gl.GLLinePlotItem(pos=zero2, color=(1, 0, 1, 1),
                                              width=2, antialias=True, mode='lines')
        self.gl_aircraft.addItem(self.ac_velocity)

        # ── Try to load an STL aircraft model ────────────────────────
        self._stl_loaded = False
        if os.path.isfile(_AIRCRAFT_STL_PATH):
            try:
                verts, faces, n_tri, face_colors, vertex_colors = \
                    _load_model(_AIRCRAFT_STL_PATH)
                verts = _normalize_aircraft_mesh(verts)
                kwargs = dict(
                    vertexes=verts, faces=faces,
                    smooth=False,           # flat-shaded; fast load for big meshes
                    shader='shaded',
                    drawEdges=False,
                    glOptions='opaque',
                )
                if vertex_colors is not None:
                    # Baked texture output: per-vertex colors give the best
                    # appearance with smooth shading so the sampled pixels
                    # interpolate across each triangle.
                    kwargs['vertexColors'] = vertex_colors
                    kwargs['smooth']       = True
                    color_note = "with baked per-vertex texture colors"
                elif face_colors is not None:
                    kwargs['faceColors'] = face_colors
                    color_note = "with embedded per-face colors"
                else:
                    kwargs['color'] = _AIRCRAFT_COLOR
                    color_note = "no embedded colors (using default flat color)"
                self.ac_model = gl.GLMeshItem(**kwargs)
                self.gl_aircraft.addItem(self.ac_model)
                self._stl_loaded = True
                print(f"[flight_gui] Loaded model '{os.path.basename(_AIRCRAFT_STL_PATH)}': "
                      f"{n_tri} triangles, {len(verts)} vertices, {color_note}.")
            except Exception as e:
                print(f"[flight_gui] STL load failed ({e!r}); using wireframe.")

        # ── Fallback wireframe (only if STL did not load) ────────────
        if not self._stl_loaded:
            ac_color = (0.95, 0.95, 0.95, 1.0)
            self.ac_fuselage = gl.GLLinePlotItem(pos=zero2, color=ac_color, width=3,
                                                  antialias=True, mode='lines')
            self.ac_tail     = gl.GLLinePlotItem(pos=zero2, color=ac_color, width=3,
                                                  antialias=True, mode='lines')
            for item in (self.ac_fuselage, self.ac_tail):
                self.gl_aircraft.addItem(item)

            # Surface panels: wings, horizontal & vertical stabilizers
            self.ac_panels = {}
            for name, rgba in _AC_PANEL_COLORS.items():
                fc = np.array([rgba, rgba], dtype=np.float32)
                item = gl.GLMeshItem(
                    vertexes=np.zeros((4, 3), dtype=np.float32),
                    faces=_PANEL_FACES,
                    faceColors=fc,
                    smooth=False,
                    drawEdges=True,
                    edgeColor=(0.9, 0.9, 0.9, 0.6),
                )
                item._fc = fc
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

    # NED -> GL coordinate swap (constant): GL=[E, N, Up] = M_swap @ NED=[N, E, D]
    _M_SWAP_NED_GL = np.array([[0.0, 1.0,  0.0],
                                [1.0, 0.0,  0.0],
                                [0.0, 0.0, -1.0]])

    def _update_3d_aircraft(self):
        # ── Velocity vector (always drawn) ──────────────────────
        v_norm = np.linalg.norm(self.v_ned)
        if v_norm > 1e-3:
            v_draw = (self.v_ned / v_norm) * 2 if v_norm > 2 else self.v_ned
            vn, ve, vd = v_draw
            self.ac_velocity.setData(
                pos=np.array([[0, 0, 0], [ve, vn, -vd]], dtype=np.float32))

        # ── STL path: apply a single 4x4 transform on the GPU ───
        # This is O(1) per frame regardless of triangle count, because the
        # mesh stays uploaded and only the model matrix changes.
        if self._stl_loaded:
            R = self._M_SWAP_NED_GL @ self.C_b_n      # body -> GL (3x3)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            self.ac_model.setTransform(
                pg.Transform3D(*T.flatten().tolist()))
            return

        # ── Wireframe fallback: per-vertex re-upload (small mesh) ──
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
