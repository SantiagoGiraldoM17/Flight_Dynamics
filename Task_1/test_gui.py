import tkinter as tk
from flight_gui import FlightDataViewer
from flight_math import transform_flight_data, compute_aero_angles, aircraft_state

cases = [
    {"name": "Case A: Straight-and-level",  "euler": [0.0, 0.0, 45.0],  "v_body": [100.0, 0.0, 0.0]},
    {"name": "Case B: Climb",               "euler": [0.0, 15.0, 0.0],  "v_body": [100.0, 0.0, 0.0]},
    {"name": "Case C: Climbing Right Turn", "euler": [30.0, 5.0, 110.0], "v_body": [100.0, 5.0, 20.0]}
]

current_case_idx = 0


def show_next_case(event=None):
    global current_case_idx
    
    # Grab the data for the current case
    current_state = cases[current_case_idx]
    print(f"\n{'='*40}")
    print(f">>> Switched to {current_state['name']} <<<")
    print(f"{'='*40}")
    
    v_body = current_state["v_body"]
    euler = current_state["euler"]
    angular_rates = [0.0, 0.0, 0.0] # Test cases don't provide p, q, r
    
    # 1. Update the GUI
    app.update_state(euler, v_body)
    
    # 2. Do the math here in the test script
    v_ned, _ = transform_flight_data(euler, v_body)
    aero_angles = compute_aero_angles(euler, v_body)
    
    # 3. Package it into the dictionary format 
    state_dict = aircraft_state(v_body, v_ned, euler, angular_rates, aero_angles)
    
    # 4. Print the terminal output
    print(f"NED Velocity [V_N, V_E, V_D] : {state_dict['velocities_ned'].round(2)}")
    print(f"Alpha (α): {state_dict['angles'][0]:.2f} deg")
    print(f"Beta (β) : {state_dict['angles'][1]:.2f} deg")
    print(f"Gamma (γ): {state_dict['angles'][2]:.2f} deg\n")
    
    # Move the index to the next case (loops back to 0 at the end)
    current_case_idx = (current_case_idx + 1) % len(cases)

# --- Start up the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    
    print("\nInitializing Flight Dynamics Presentation Mode...")
    print(">>> PRESS THE SPACEBAR TO CYCLE THROUGH FLIGHT CASES <<<\n")
    
    # Initialize the GUI with Case A's starting values
    app = FlightDataViewer(root, euler_in=cases[0]["euler"], v_body_in=cases[0]["v_body"])
    
    # Bind the spacebar key to trigger the function
    root.bind('<space>', show_next_case)
    
    # Run it once immediately to print the first case to the terminal
    show_next_case()
    
    root.mainloop()
