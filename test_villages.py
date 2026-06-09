
from bhume import load
from pathlib import Path

print("Testing village loading...")
print()

# Test Vadnerbhairav
vadner_dir = Path("data/34855_vadnerbhairav_chandavad_nashik")
if vadner_dir.exists():
    try:
        village = load(vadner_dir)
        print("[OK] Successfully loaded Vadnerbhairav village")
        print(f"   Number of plots: {len(village.plots)}")
        if village.example_truths is not None:
            print(f"   Number of example truths: {len(village.example_truths)}")
        print()
    except Exception as e:
        print(f"[FAIL] Failed to load Vadnerbhairav village: {e}")
        print()

# Test Malatavari
malatavari_dir = Path("data/Malatavari")
if malatavari_dir.exists():
    try:
        village = load(malatavari_dir)
        print("[OK] Successfully loaded Malatavari village")
        print(f"   Number of plots: {len(village.plots)}")
        if village.example_truths is not None:
            print(f"   Number of example truths: {len(village.example_truths)}")
        print()
    except Exception as e:
        print(f"[FAIL] Failed to load Malatavari village: {e}")
        print()
