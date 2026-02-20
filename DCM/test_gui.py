import tkinter as tk
from flight_gui import FlightDataViewer

# Your exact 3 test cases
cases = [
    {"name": "Case A: Straight-and-level",  "euler": [0.0, 0.0, 45.0],  "v_body": [100.0, 0.0, 0.0]},
    {"name": "Case B: Climb",               "euler": [0.0, 15.0, 30.0],  "v_body": [100.0, 0.0, 0.0]},
    {"name": "Case C: Climbing Right Turn", "euler": [30.0, 5.0, 90.0], "v_body": [100.0, 5.0, 8.0]}
]

current_case_idx = 0

def show_next_case():
    global current_case_idx
    
    # Grab the exact data for the current case
    current_state = cases[current_case_idx]
    print(f"Switching to {current_state['name']}...")
    
    # Push the exact arrays into your GUI
    app.update_state(current_state["euler"], current_state["v_body"])
    
    # Move the index to the next case (the % loops it back to 0 at the end)
    current_case_idx = (current_case_idx + 1) % len(cases)
    
    # Wait exactly 10,000 milliseconds (10 seconds), then run this function again
    root.after(10000, show_next_case)

# --- Start up the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    
    print(f"Starting with {cases[0]['name']}...")
    
    # Initialize the GUI with Case A
    app = FlightDataViewer(root, euler_in=cases[0]["euler"], v_body_in=cases[0]["v_body"])
    
    # Increment the index so the next one to show is Case B
    current_case_idx = 1
    
    # Queue the first switch to happen in 10 seconds
    root.after(10000, show_next_case)
    
    root.mainloop()