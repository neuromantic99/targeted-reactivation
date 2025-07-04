import matplotlib.pyplot as plt
import numpy as np

def plot_sync_vs_camera_times(aligner, max_points=10000):
    sync_times = np.array(aligner.sync_times)
    cam_times = np.array(aligner.camera_times)

    # Basic checks
    print(f"Total sync pulses: {len(sync_times)}")
    print(f"Total camera triggers: {len(cam_times)}")
    print(f"Sync time range: {sync_times[0]:.3f} to {sync_times[-1]:.3f} seconds")
    print(f"Camera time range: {cam_times[0]:.3f} to {cam_times[-1]:.3f} seconds")

    # Optional truncation for plotting performance
    sync_plot = sync_times[:max_points]
    cam_plot = cam_times[:max_points]

    plt.figure(figsize=(12, 6))
    plt.plot(sync_plot, np.ones_like(sync_plot), 'k.', label='Sync Pulses', alpha=0.6)
    plt.plot(cam_plot, np.ones_like(cam_plot)*1.1, 'r.', label='Camera Triggers', alpha=0.6)

    plt.xlabel('Time (s)')
    plt.yticks([])
    plt.legend()
    plt.title('Sync Pulse Times vs Camera Trigger Times')
    plt.tight_layout()
    plt.show()
