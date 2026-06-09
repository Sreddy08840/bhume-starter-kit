"""
Test script for area validation.
"""

from pathlib import Path
from solution import VillageDataLoader, AreaValidator


def main():
    # Replace with your village directory path
    village_dir = Path("data") / "34855_vadnerbhairav_chandavad_nashik"

    if not village_dir.exists():
        print(f"Village directory not found: {village_dir}")
        print("Please download a village bundle and place it in data/")
        return

    print(f"Loading village: {village_dir.name}")

    # Load village data
    loader = VillageDataLoader(village_dir)
    village_data = loader.load()

    # Get a few plots to test
    plot_nums = village_data.plots.index[:5]  # Test first 5 plots
    print(f"Testing {len(plot_nums)} plots")

    # Initialize area validator
    validator = AreaValidator(min_ratio=0.5, max_ratio=2.0)

    print("\n--- Area Validation Results ---")
    for plot_num in plot_nums:
        # Calculate recorded area - check if there's an area column, otherwise use our own calculation
        # For this test, we'll use the calculated area from the plot as "recorded"
        plot_geom = village_data.get_plot(plot_num)
        recorded_area = village_data.calculate_area(plot_num)
        # Let's test with some modified areas
        for i, scale in enumerate([0.5, 0.8, 1.0, 1.2, 2.0, 2.5]):
            polygon_area = recorded_area * scale
            result = validator.validate(recorded_area, polygon_area)
            print(f"\nPlot {plot_num} (scale {scale:.1f}x):")
            print(f"  Recorded area: {recorded_area:.2f} m²")
            print(f"  Polygon area: {polygon_area:.2f} m²")
            print(f"  Area ratio: {result.area_ratio:.4f}")
            print(f"  Area score: {result.area_score:.4f}")
            print(f"  Flag reason: {result.flag_reason.value}")
            print(f"  Is suspicious: {result.is_suspicious}")


if __name__ == "__main__":
    main()
