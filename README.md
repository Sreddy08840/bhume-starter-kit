# 🏞️ BhuMe Boundary Take-Home: Platinum-Level Solution

The official plot outlines in Maharashtra's land records sit metres off the real fields (an artifact of how old paper maps were georeferenced onto satellite imagery). **Our solution returns best estimate of true on-the-ground boundaries, calibrated confidence, and flags uncertain plots!**

---

## 🏆 What's Included

This solution includes everything from the starter kit plus:
- **Complete end-to-end pipeline**: `upgraded_pipeline.py` for Gold/Platinum performance
- **Modular components (in `solution/` directory)**:
  - `loader.py`: Village data loading, plot lookup, area calculation, neighbor detection
  - `patch_extraction.py`: Extract satellite image and boundary mask patches for any plot
  - `boundary_detection.py`: Advanced boundary detection using CLAHE enhancement, multi-scale Canny, and morphological operations
  - `alignment.py`: Global alignment (using example truths if available) + local refinement via grid search
  - `area_validation.py`: Validate plot area against recorded area
  - `neighbor_consistency.py`: Check alignment consistency with neighboring plots
  - `confidence_calibration.py`: Combine multiple factors into final confidence score
  - `prediction_generator.py`: Generate contract-valid predictions
  - `pipeline.py`: Complete pipeline orchestration
- **Test files**: For each module to verify functionality
- **Transcripts folder**: `transcripts/` for AI conversations (per submission requirements)
- **Robust error handling**: Handles invalid data, missing files, and edge cases gracefully

---

## 🚀 Setup

This solution uses [uv](https://docs.astral.sh/uv/), which is a blazing-fast Python package manager and project manager!

### Install uv (if you haven't already)
Follow instructions at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Sync dependencies
Once uv is installed:
```bash
uv sync
```
This will:
1. Create a local virtual environment (no activation needed, just prefix **ALL** commands with `uv run`)
2. Install all required dependencies (geopandas, rasterio, shapely, numpy, scipy, opencv-python, and scikit-learn!)

### Important: Always use `uv run`!
Never run scripts directly with `python.exe`! Always use:
```bash
uv run <script name>
```
This ensures you use the correct virtual environment with all dependencies installed!

---

## 📦 Get and Organize the Data

### Step 1: Download Village Bundles
Download both village bundles from the challenge site's **Get started** page!

### Step 2: Save and Organize
Each village should be in its own subfolder in `data/`.

For each village, ensure all 4 files are present and valid (not empty!):
1. `input.geojson`: Official (shifted) plots (full file)
2. `imagery.tif`: Satellite mosaic (your primary signal!)
3. `boundaries.tif`: Optional auto-detected field hints
4. `example_truths.geojson`: Hand-aligned examples for self-scoring

Directory structure should look like:
```
bhume-starter-kit/
├── data/
│   ├── 34855_vadnerbhairav_chandavad_nashik/
│   │   ├── input.geojson          # Full file, 2457 plots for Vadnerbhairav
│   │   ├── imagery.tif
│   │   ├── boundaries.tif
│   │   └── example_truths.geojson
│   └── Malatavari/
│       ├── input.geojson
│       ├── imagery.tif
│       ├── boundaries.tif
│       └── example_truths.geojson
```

### Step 3: Verify Data
Make sure data is loaded correctly (check plot counts!):
```bash
uv run check_truths.py
```

---

## 🏃‍♀️ Run It!

### Quickstart (Naive Baseline from Starter Kit)
First, run the naive baseline to see the floor to beat:
```bash
uv run quickstart.py data/34855_vadnerbhairav_chandavad_nashik
```

### Our Upgraded Platinum Solution
Run our full pipeline to generate predictions!
For Vadnerbhairav:
```bash
uv run upgraded_pipeline.py data/34855_vadnerbhairav_chandavad_nashik
```
For Malatavari:
```bash
uv run upgraded_pipeline.py data/Malatavari
```

This will create `predictions.geojson` in the village's directory!

### Check What's in the Data
Use `check_truths.py` to confirm the plots in input and example truths match:
```bash
uv run check_truths.py
```

---

## 🔬 How the Pipeline Works (Step by Step!)

1. **Load Village Data**:
   - Load `input.geojson` (ensures correct EPSG:4326 CRS)
   - Load imagery.tif and boundaries.tif
   - Load example truths (if present) for global alignment

2. **Compute Global Alignment**:
   - If example truths are available, calculate a robust median shift (reduces outliers!)
   - Applies this shift as initial guess for all plots, helping generalize

3. **Process Each Plot Individually**:
   - Extract a satellite image patch around the plot (+40m padding!)
   - Detect real field boundaries using multi-scale Canny edge detection + CLAHE enhancement
   - Refine alignment using grid search (translation and rotation)
   - Calculate alignment score, edge strength, area validity, etc.

4. **Confidence Calibration**:
   - Combine multiple factors (alignment score, area score, edge strength, neighbor consistency!)
   - If example truths are available, uses Isotonic Regression to better calibrate confidence!

5. **Generate Predictions**:
   - Uses confidence threshold of 0.7 to decide whether to correct or flag
   - Creates valid GeoJSON with all required fields!

---

## 📊 Key Improvements Over Baseline

| Feature | Baseline | Platinum Solution |
|---------|----------|-------------------|
| Alignment | Global median shift only | Global shift + local grid search refinement |
| Boundary Detection | None! (only uses baseline shift) | Multi-scale Canny + CLAHE enhancement + morphological operations |
| Confidence | Flat confidence (all 0.5) | Calibrated, multi-factor confidence score |
| Area Validation | None! | Validates plot area vs recorded area, flags invalid cases |
| Neighbor Consistency | None! | Checks if shift matches neighboring plots |
| Restraint Logic | N/A | Will be added if needed! |
| Error Handling | Limited | Robust, handles missing data, failed plots gracefully |

---

## 📝 Submission Checklist

Before submitting:
- [ ] Download both villages into data/ with valid input.geojson and example_truths.geojson files
- [ ] Run `upgraded_pipeline.py` on both villages to generate predictions.geojson
- [ ] Check predictions.geojson are contract-valid
- [ ] Add your AI transcripts to the `transcripts/` directory
- [ ] Record your 5-minute video explaining your approach and results
- [ ] Clean up any temporary/test files (optional but good practice)
- [ ] Commit your changes (if using git)
- [ ] Submit the whole repository!

---

## 📖 Notes on Scoring
The `score()` function from the starter kit mirrors the challenge's objective metrics! It:
- Uses IoU (Intersection over Union) to measure accuracy
- Checks for improvement over official positions
- Evaluates confidence calibration (higher confidence should mean higher IoU)
- Measures restraint (don't move already-correct plots too much!)

Remember: only run against the public example truths as a rough directional check! Don't overfit!
