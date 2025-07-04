import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import json
from pathlib import Path

def input_channel_ranges(mouse_id: str, session_plots: list) -> dict:
    """
    Simple GUI to input CA1 and RSC channels after viewing plots
    Args:
        mouse_id: Mouse identifier
        session_plots: List of paths to SWR plot images
    Returns:
        Dictionary with {'ca1_range': (start, end), 'rsc_range': (start, end)}
    """
    # First show all plots
    for plot_path in session_plots:
        img = plt.imread(plot_path)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(img)
        ax.axis('off')
        plt.title(f"{mouse_id} - {plot_path.name}")
        plt.show()
    
    # Now create input dialog
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.4)
    ax.axis('off')
    ax.set_title(f"Enter channel ranges for {mouse_id}")
    
    # Add input boxes
    ax_ca1_start = plt.axes([0.3, 0.25, 0.1, 0.05])
    ax_ca1_end = plt.axes([0.45, 0.25, 0.1, 0.05])
    ax_rsc_start = plt.axes([0.3, 0.15, 0.1, 0.05])
    ax_rsc_end = plt.axes([0.45, 0.15, 0.1, 0.05])
    ax_save = plt.axes([0.3, 0.05, 0.3, 0.05])
    
    ca1_start = TextBox(ax_ca1_start, 'CA1 start:', initial='')
    ca1_end = TextBox(ax_ca1_end, 'CA1 end:', initial='')
    rsc_start = TextBox(ax_rsc_start, 'RSC start:', initial='')
    rsc_end = TextBox(ax_rsc_end, 'RSC end:', initial='')
    btn_save = Button(ax_save, 'Save Selections', color='lightgreen')
    
    # Store results
    result = {}
    
    def save(event):
        try:
            result['ca1_range'] = (int(ca1_start.text), int(ca1_end.text))
            result['rsc_range'] = (int(rsc_start.text), int(rsc_end.text))
            plt.close()
        except ValueError:
            print("Please enter valid integers for all fields")
    
    btn_save.on_clicked(save)
    plt.show()
    
    return result

def save_channel_ranges(mouse_dir: Path, ranges: dict):
    """Save ranges to JSON file in mouse directory"""
    with open(mouse_dir / "channel_ranges.json", 'w') as f:
        json.dump(ranges, f, indent=2)