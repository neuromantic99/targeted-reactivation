import csv
from pathlib import Path
from typing import Dict, Any, List
from main import main
from main import process_mouse

def load_mouse_mappings(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load mouse data from CSV"""
    mice = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mice[row['mouse_id']] = {
                'data_folder': Path(row['lfp_path']).parent.parent,
                'lfp_path': Path(row['lfp_path']),
                'ap': float(row['AP']),
                'ml': float(row['ML']),
                'az': int(row['AZ']),
                'elevation': int(row['elevation']),
                'depth': int(row['depth']),
                'n_channels': int(row['n_channels'])
            }
    return mice



def run_phase(mice: dict, phase: int):
    """Run a specific phase for all mice to allow us to run swr and power plots separately"""
    all_sessions = []
    
    for mouse_id, params in mice.items():
        if phase == 1:
            process_mouse(phase=1, **params)
        elif phase == 2:
            print(f"\nMouse: {mouse_id}")
            ca1 = (int(input("CA1 start: ")), int(input("CA1 end: ")))
            rsc = (int(input("RSC start: ")), int(input("RSC end: ")))
            
            sessions_list = process_mouse(
                phase=2,
                ca1_range=ca1,
                rsc_range=rsc,
                **params
            )
            all_sessions.extend(sessions_list)
    
    return all_sessions

if __name__ == "__main__":
    mice = load_mouse_mappings(Path("Z:/Alex/Reactivations/mouse_lfp_mapping.csv"))
    
    # PHASE 1: Generate SWR plots
    print("=== GENERATING SWR PLOTS ===")
    run_phase(mice, phase=1)
    
    # PHASE 2: Power plots with manual input
    print("\n=== POWER PLOTS ===")
    all_sessions = run_phase(mice, phase=2)
    
    # Optional: Combined processing across all mice
    # plot_ripple_power_combined_processing(all_sessions)