import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as mtransforms
from matplotlib.patches import Circle

# Import your math engine!
from dcm import transform_flight_data, compute_aero_angles

class FlightDataViewer:
    def __init__(self, root, euler_in, v_body_in):
        self.root = root
        self.root.title("Flight Dynamics State Viewer")
        self.root.geometry("1000x600")

        # Store inputs
        self.euler = euler_in
        self.v_body = v_body_in

        # Process the math using your custom functions
        self.v_ned, _ = transform_flight_data(self.euler, self.v_body)
        self.alpha, self.beta, self.gamma = compute_aero_angles(self.v_body, self.v_ned)
        
        # Re-build the DCM for the 3D rotation (since transform_flight_data doesn't return it directly)
        phi, theta, psi = np.radians(self.euler)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        self.C_b_n = np.array([
            [c_th*c_psi, s_phi*s_th*c_psi - c_phi*s_psi, c_phi*s_th*c_psi + s_phi*s_psi],
            [c_th*s_psi, s_phi*s_th*s_psi + c_phi*c_psi, c_phi*s_th*s_psi - s_phi*c_psi],
            [-s_th,      s_phi*c_th,                     c_phi*c_th]
        ])

        self.setup_ui()
        self.update_plots()

    def setup_ui(self):
        # --- Left Panel: Data Readout ---
        data_frame = ttk.Frame(self.root, width=300, padding="15")
        data_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(data_frame, text="Current Flight State", font=('Arial', 14, 'bold')).pack(pady=10)
        
        text = (f"INPUTS (Body Frame):\n"
                f"  Roll  : {self.euler[0]:.1f}°\n"
                f"  Pitch : {self.euler[1]:.1f}°\n"
                f"  Yaw   : {self.euler[2]:.1f}°\n"
                f"  u (Fwd): {self.v_body[0]:.1f} m/s\n"
                f"  v (Rgt): {self.v_body[1]:.1f} m/s\n"
                f"  w (Dwn): {self.v_body[2]:.1f} m/s\n\n"
                f"OUTPUTS (NED Frame):\n"
                f"  V_North: {self.v_ned[0]:.1f} m/s\n"
                f"  V_East : {self.v_ned[1]:.1f} m/s\n"
                f"  V_Down : {self.v_ned[2]:.1f} m/s\n\n"
                f"AERODYNAMICS:\n"
                f"  Alpha (AoA) : {self.alpha:.1f}°\n"
                f"  Beta (Slip) : {self.beta:.1f}°\n"
                f"  Gamma (Path): {self.gamma:.1f}°\n")
        
        self.data_label = ttk.Label(data_frame, text=text, font=('Courier', 11), justify=tk.LEFT)
        self.data_label.pack(anchor=tk.W)
        
        # --- Right Panel: Plots (Updated Layout) ---
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(10, 6))
        
        # Use GridSpec to stack the AI and Compass on the left, and the 3D plot on the right
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.5]) 
        
        self.ax_ai = self.fig.add_subplot(gs[0, 0])
        self.ax_ai.set_aspect('equal')
        
        self.ax_hi = self.fig.add_subplot(gs[1, 0])  # New Heading Indicator axis
        self.ax_hi.set_aspect('equal')
        
        self.ax_3d = self.fig.add_subplot(gs[:, 1], projection='3d') # Spans both rows

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_state(self, new_euler, new_v_body):
        """Receives new data, recalculates the math, and triggers a UI refresh."""
        self.euler = new_euler
        self.v_body = new_v_body
        
        # 1. Recalculate using your math engine
        self.v_ned, _ = transform_flight_data(self.euler, self.v_body)
        self.alpha, self.beta, self.gamma = compute_aero_angles(self.v_body, self.v_ned)
        
        # 2. Re-build the DCM for the 3D model rotation
        phi, theta, psi = np.radians(self.euler)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        self.C_b_n = np.array([
            [c_th*c_psi, s_phi*s_th*c_psi - c_phi*s_psi, c_phi*s_th*c_psi + s_phi*s_psi],
            [c_th*s_psi, s_phi*s_th*s_psi + c_phi*c_psi, c_phi*s_th*s_psi - s_phi*c_psi],
            [-s_th,      s_phi*c_th,                     c_phi*c_th]
        ])
        
        # 3. Update the text panel
        text = (f"INPUTS (Body Frame):\n"
                f"  Roll  : {self.euler[0]:.1f}°\n"
                f"  Pitch : {self.euler[1]:.1f}°\n"
                f"  Yaw   : {self.euler[2]:.1f}°\n"
                f"  u (Fwd): {self.v_body[0]:.1f} m/s\n"
                f"  v (Rgt): {self.v_body[1]:.1f} m/s\n"
                f"  w (Dwn): {self.v_body[2]:.1f} m/s\n\n"
                f"OUTPUTS (NED Frame):\n"
                f"  V_North: {self.v_ned[0]:.1f} m/s\n"
                f"  V_East : {self.v_ned[1]:.1f} m/s\n"
                f"  V_Down : {self.v_ned[2]:.1f} m/s\n\n"
                f"AERODYNAMICS:\n"
                f"  Alpha (AoA) : {self.alpha:.1f}°\n"
                f"  Beta (Slip) : {self.beta:.1f}°\n"
                f"  Gamma (Path): {self.gamma:.1f}°\n")
        
        self.data_label.config(text=text)
        
        # 4. Redraw the instruments and 3D plane!
        self.update_plots()

    def update_plots(self):
        # ==========================================
        # 1. Attitude Indicator (Round with Bank Scale)
        # ==========================================
        self.ax_ai.clear()
        self.ax_ai.set_title("Attitude Indicator")
        self.ax_ai.set_xlim(-1.2, 1.2)
        self.ax_ai.set_ylim(-1.2, 1.2)
        self.ax_ai.axis('off') # Hides the square grid

        # Create the circular clipping mask
        clip_circle = Circle((0, 0), radius=1.0, transform=self.ax_ai.transData)

        pitch_deg = self.euler[1]
        roll_deg = self.euler[0]

        # Scaling: 1.0 graphical unit = 45 degrees of pitch
        horizon_y = -pitch_deg / 45.0

        # Draw Sky and Ground
        sky = plt.Rectangle((-2, horizon_y), 4, 4, color='#1f77b4')
        ground = plt.Rectangle((-2, horizon_y - 4), 4, 4, color='#8c564b')
        
        # Apply the circular mask
        sky.set_clip_path(clip_circle)
        ground.set_clip_path(clip_circle)
        
        # Create the rotation transform for Roll
        transform = mtransforms.Affine2D().rotate_deg_around(0, 0, roll_deg) + self.ax_ai.transData
        
        # Apply rotation to sky and ground
        sky.set_transform(transform)
        ground.set_transform(transform)
        
        self.ax_ai.add_patch(sky)
        self.ax_ai.add_patch(ground)

        # --- Draw the Pitch Ladder (No numbers, limited to +/- 30) ---
        for p in range(-30, 31, 10):
            if p == 0: continue # Skip horizon line
            
            y_pos = (p - pitch_deg) / 45.0
            
            # Draw if inside the visible dial
            if -1.1 < y_pos < 1.1:
                # Make 10 and 30 smaller, 20 wider
                width = 0.4 if p % 20 == 0 else 0.2
                
                line, = self.ax_ai.plot([-width/2, width/2], [y_pos, y_pos], color='white', lw=1.5)
                line.set_clip_path(clip_circle)
                line.set_transform(transform)

        # --- Draw the Bank Angle Scale (Roll Ticks) ---
        # Standard aviation marks at 0, 10, 20, 30, 45, and 60 degrees left/right
        bank_angles = [0, 10, 20, 30, 45, 60, -10, -20, -30, -45, -60]
        for angle in bank_angles:
            rad = np.radians(angle)
            # Coordinates for the outer edge of the circle (0 degrees is straight up)
            x_out, y_out = np.sin(rad), np.cos(rad)
            
            # Make the 0, 30, and 60 degree marks slightly longer
            inner_r = 0.85 if abs(angle) in [0, 30, 60] else 0.92
            x_in, y_in = inner_r * np.sin(rad), inner_r * np.cos(rad)
            
            # Plot the tick and apply the roll transform so it rotates with the sky
            tick, = self.ax_ai.plot([x_in, x_out], [y_in, y_out], color='white', lw=2)
            tick.set_transform(transform)
            tick.set_clip_path(clip_circle)

        # --- Draw the Static Aircraft Reference (Orange Crosshair) ---
        self.ax_ai.plot([-0.4, -0.1], [0, 0], color='orange', lw=3)
        self.ax_ai.plot([0.1, 0.4], [0, 0], color='orange', lw=3)
        self.ax_ai.plot(0, 0, marker='o', color='orange', markersize=6)
        
        # --- Draw the Stationary Roll Pointer (Top Triangle) ---
        # This is drawn outside the transform, so it stays perfectly still at 12 o'clock
        self.ax_ai.fill([-0.06, 0.06, 0], [1.08, 1.08, 0.92], color='orange', zorder=10)

        # --- Draw the physical instrument Bezel ---
        bezel = Circle((0, 0), radius=1.0, edgecolor='dimgray', facecolor='none', lw=6, zorder=11)
        self.ax_ai.add_patch(bezel)

        # ==========================================
        # 2. Heading Indicator (Compass)
        # ==========================================
        self.ax_hi.clear()
        self.ax_hi.set_title("Heading Indicator")
        self.ax_hi.set_xlim(-1.2, 1.2)
        self.ax_hi.set_ylim(-1.2, 1.2)
        self.ax_hi.axis('off')

        # Draw the black instrument background and bezel
        hi_bg = Circle((0, 0), radius=1.0, facecolor='#111111', edgecolor='dimgray', lw=6)
        self.ax_hi.add_patch(hi_bg)

        # Draw the Compass Ticks and N/E/S/W Labels
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            
            # Aviation math: 0 is North (Up), 90 is East (Right)
            # This maps perfectly to x=sin(rad), y=cos(rad)
            x_out, y_out = np.sin(rad), np.cos(rad)
            
            if angle % 90 == 0:
                # Cardinal directions get letters instead of lines
                label = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}[angle]
                self.ax_hi.text(x_out * 0.75, y_out * 0.75, label, color='white', 
                                fontsize=14, ha='center', va='center', weight='bold')
                
                # Shorter tick mark for cardinal directions
                x_in, y_in = 0.85 * x_out, 0.85 * y_out
            else:
                # Standard tick mark every 30 degrees
                x_in, y_in = 0.9 * x_out, 0.9 * y_out
                
            self.ax_hi.plot([x_in, x_out], [y_in, y_out], color='white', lw=2)

        # Get the current Yaw (Heading) angle
        yaw_rad = np.radians(self.euler[2])
        
        # Calculate the tip of the arrow
        arrow_x = np.sin(yaw_rad) * 0.6
        arrow_y = np.cos(yaw_rad) * 0.6
        
        # Draw the rotating Heading Arrow
        self.ax_hi.arrow(0, 0, arrow_x, arrow_y, head_width=0.15, head_length=0.2, 
                         fc='orange', ec='orange', lw=4, zorder=5)
        
        # Draw a center pin to make it look mechanical
        self.ax_hi.plot(0, 0, marker='o', color='gray', markersize=8, zorder=6)

        # ==========================================
        # 3. 3D Wireframe Aircraft
        # =========================================

        self.ax_3d.clear()
        self.ax_3d.set_title("3D Aircraft Orientation")
        self.ax_3d.set_xlim([-3, 3])
        self.ax_3d.set_ylim([-3, 3])
        self.ax_3d.set_zlim([-3, 3])
        self.ax_3d.invert_zaxis() # Down is positive
        
        self.ax_3d.set_xlabel("North")
        self.ax_3d.set_ylabel("East")
        self.ax_3d.set_zlabel("Down")

        # Define basic aircraft points in Body Frame (X-fwd, Y-right, Z-down)
        # 0:Nose, 1:Tail, 2:LeftWing, 3:RightWing, 4:VertStabTop
        pts_body = np.array([
            [ 2.0,  0.0,  0.0],  # Nose
            [-2.0,  0.0,  0.0],  # Tail
            [ 0.0, -2.5,  0.0],  # Left Wing
            [ 0.0,  2.5,  0.0],  # Right Wing
            [-2.0,  0.0, -1.0]   # Vert Tail (Negative Z is UP relative to aircraft)
        ]).T

        # Rotate points to NED Frame using the DCM
        pts_ned = self.C_b_n @ pts_body

        # Extract transformed coordinates
        x, y, z = pts_ned[0], pts_ned[1], pts_ned[2]

        # Draw the wireframe lines
        self.ax_3d.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color='black', lw=3) # Fuselage
        self.ax_3d.plot([x[2], x[3]], [y[2], y[3]], [z[2], z[3]], color='blue', lw=3)  # Wings
        self.ax_3d.plot([x[1], x[4]], [y[1], y[4]], [z[1], z[4]], color='red', lw=3)   # Tail

        # --- Draw Fixed NED Axes for Reference ---
        # Draw thick colored arrows for North, East, and Down
        # The axes stretch out 2.5 units from the Center of Gravity
        
        # North (X-axis) -> RED
        self.ax_3d.quiver(0, 0, 0, 2.5, 0, 0, color='red', lw=2.5, arrow_length_ratio=0.1)
        self.ax_3d.text(2.7, 0, 0, 'N', color='red', fontsize=12, weight='bold')

        # East (Y-axis) -> GREEN
        self.ax_3d.quiver(0, 0, 0, 0, 2.5, 0, color='green', lw=2.5, arrow_length_ratio=0.1)
        self.ax_3d.text(0, 2.7, 0, 'E', color='green', fontsize=12, weight='bold')

        # Down (Z-axis) -> BLUE
        self.ax_3d.quiver(0, 0, 0, 0, 0, 2.5, color='blue', lw=2.5, arrow_length_ratio=0.1)
        self.ax_3d.text(0, 0, 2.7, 'D', color='blue', fontsize=12, weight='bold')

        # Draw Velocity Vector
        v_norm = np.linalg.norm(self.v_ned)
        if v_norm > 0:
            v_scale = (self.v_ned / v_norm) * 2  # Scaled for visibility
            self.ax_3d.quiver(0, 0, 0, v_scale[0], v_scale[1], v_scale[2], color='magenta', lw=2, label='Velocity')

        self.ax_3d.legend(loc='upper left', fontsize='small')
        self.canvas.draw()