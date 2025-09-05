from pathlib import Path


LOCAL_SSD = Path("/mnt/local-ssd/james/targeted-reactivation")
DETECTION_METHOD = "median"  # options 'median' or 'sd'

RIPPLE_BAND = [120, 250]
SUPRA_RIPPLE_BAND = [250, 500]

KILOSORT_UMBRELLA = Path("/mnt/MarcBusche/Alex/Reactivations")
SERVER_CACHE_PATH = Path(
    "/mnt/MarcBusche/James/caches/spontaneous_reactivation_strength"
)
