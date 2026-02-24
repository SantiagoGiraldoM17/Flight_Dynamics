# Task 1: Coordinate Frames and Transformation Matrix

## Objective
 The goal of this assignment is to implement a basic software tool (combining code, a graphical interface, and terminal outputs) to transform and visualize a vector expressed in different coordinate frames. Additionally, it computes key aerodynamic and geometrical parameters ([α, β, γ, ψ, θ, φ]) across various flight conditions.

## Features & Implementation

### 1. Coordinate Transformations
* **Inputs:** Euler angles and a velocity vector in the body frame.
* **Outputs:** The vector expressed in the NED (North-East-Down) frame using transformation matrices.

### 2. Flight Simulation Cases
The tool simulates and visualizes three distinct logical flight conditions:
* **Case A:** Straight-and level flight.
* **Case B:** Climb or descent.
* **Case C:** Aircraft Turn.

### 3. Data Output
* **Terminal:** Prints the velocity vector described in the NED frame, along with Angle of Attack (α), Sideslip (β), and Climb angle (γ).
* **Structured Data:** Returns all aircraft state values (angles, body velocities, NED velocities, and attitude) in a structured dictionary format.

### 4. Graphical User Interface (GUI)
Our custom interface fulfills all visual requirements by displaying:
* Vector orientation in the Body frame.
* Visual representation of the NED and Body Frames.
* Live numerical values.
* A 3D rendered aircraft.

