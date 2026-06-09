
from bhume import load
from pathlib import Path

def check_village(village_name, village_dir):
    print(f"\n{'=' * 60}")
    print(f"Checking: {village_name}")
    print(f"{'=' * 60}")
    try:
        village = load(village_dir)
        print("[OK] Successfully loaded village data")

        print(f"\nInput Plots:")
        print(f"   Number of plots in input: {len(village.plots)}")

        print(f"\nExample Truths:")
        if village.example_truths is not None:
            print(f"   Number of example truths: {len(village.example_truths)}")
            print(f"   Plot numbers in truths: {list(village.example_truths['plot_number'])}")
        else:
            print("   No example truths found")
    except Exception as e:
        print(f"[FAIL] Failed to load village: {e}")

print("=== BhuMe Village Check ===")

check_village(
    "34855_vadnerbhairav_chandavad_nashik",
    Path("data/34855_vadnerbhairav_chandavad_nashik")
)

check_village(
    "Malatavari",
    Path("data/Malatavari")
)
