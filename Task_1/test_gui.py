import tkinter as tk
from flight_gui import FlightDataViewer


# Your exact 3 test cases
cases = [
    {"name": "Case A: Straight-and-level",  "euler": [0.0, 0.0, 45.0],  "v_body": [100.0, 0.0, 0.0]},
    {"name": "Case B: Climb",               "euler": [0.0, 15.0, 0.0],  "v_body": [100.0, 0.0, 0.0]},
    {"name": "Case C: Climbing Right Turn", "euler": [30.0, 5.0, 90.0], "v_body": [100.0, 5.0, 8.0]}
]

current_case_idx = 0


def show_next_case(event=None):
    global current_case_idx
    
    # Grab the exact data for the current case
    current_state = cases[current_case_idx]
    print(f"\n{'='*40}")
    print(f">>> Switched to {current_state['name']} <<<")
    print(f"{'='*40}")
    
    # Push the exact arrays into your GUI
    app.update_state(current_state["euler"], current_state["v_body"])

    # Move the index to the next case (loops back to 0 at the end)
    current_case_idx = (current_case_idx + 1) % len(cases)

# --- Start up the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    
    print("\nInitializing Flight Dynamics Presentation Mode...")
    print(">>> PRESS THE SPACEBAR TO CYCLE THROUGH FLIGHT CASES <<<\n")
    
    # Initialize the GUI with Case A's starting values
    app = FlightDataViewer(root, euler_in=cases[0]["euler"], v_body_in=cases[0]["v_body"])
    
    # Bind the spacebar key to trigger our function
    root.bind('<space>', show_next_case)
    
    # Run it once immediately to print the first case to the terminal
    show_next_case()
    
    root.mainloop()