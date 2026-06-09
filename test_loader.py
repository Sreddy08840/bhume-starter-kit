"""
Test script for VillageDataLoader.
"""

from pathlib import Path
from solution import VillageDataLoader


def main():
    # Replace with your village directory path
    village_dir = Path("data") / "34855_vadnerbhairav_chandavad_nashik"

    if not village_dir.exists():
        print(f"Village directory not found: {village_dir}")
        print("Please download a village bundle and place it in data/")
        return

    print(f"Loading village: {village_dir.name}")

    # Test loading
    loader = VillageDataLoader(village_dir)
    village_data = loader.load()

    print(f"Loaded {len(village_data.plots)} plots")
    print(f"Imagery path: {village_data.imagery_path}")
    print(f"Boundaries path: {village_data.boundaries_path}")

    # Test plot lookup
    plot_num = village_data.plots.index[0]
    print(f"\nTesting plot lookup for: {plot_num}")
    geom = village_data.get_plot(plot_num)
    print(f"Plot geometry type: {geom.geom_type}")
    print(f"Plot centroid: {geom.centroid}")

    # Test area calculation
    area = village_data.calculate_area(plot_num)
    print(f"\nPlot area: {area:.2f} m²")

    # Test neighbor lookup
    neighbors = village_data.find_neighbors(plot_num, distance_m=5.0, max_neighbors=5)
    print(f"\nNeighbors ({len(neighbors)}): {neighbors}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
