"""
Download model files from Google Drive if not already present.
Called automatically at app startup.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "sepsis_bilstm_full.keras": "10y6GDENSPUfUO_LJ0OQGE74ZiDIGtFAq",
    "scaler.pkl":               "1ndZ9T7gcxBjnVB1txBibEFIW2G9GoXWP",
    "global_medians.pkl":       "1IeK8udLT4C_9W4Hq1aay2nl4WAbahxMR",
    "eval_meta.pkl":            "1uKy0dirVP4E-i_u2pUrhtluUr8Yn-PDI",
}

def download_models():
    all_present = all(os.path.exists(os.path.join(HERE, f)) for f in FILES)
    if all_present:
        print("  ✓ All model files already present", flush=True)
        return True

    try:
        import gdown
    except ImportError:
        print("  ⚠  gdown not installed — skipping model download", flush=True)
        return False

    print("Downloading model files from Google Drive …", flush=True)
    success = True
    for filename, file_id in FILES.items():
        dest = os.path.join(HERE, filename)
        if os.path.exists(dest):
            print(f"  ✓ {filename} already exists", flush=True)
            continue
        try:
            print(f"  ↓ Downloading {filename} …", flush=True)
            gdown.download(id=file_id, output=dest, quiet=False, fuzzy=True)
            print(f"  ✓ {filename} downloaded", flush=True)
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}", flush=True)
            success = False

    return success

if __name__ == "__main__":
    download_models()
