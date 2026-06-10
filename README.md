# 🏞️ BhuMe Boundary Take-Home: Platinum-Level Solution

The official plot outlines in Maharashtra's land records sit metres off the real fields (an artifact of how old paper maps were georeferenced onto satellite imagery). **Our solution returns best estimate of true on-the-ground boundaries, calibrated confidence, and flags uncertain plots!**

---

## 📋 Table of Contents

- [What's Included](#-whats-included)
- [Requirements](#-requirements)
- [Setup](#-setup)
- [Get and Organize the Data](#-get-and-organize-the-data)
- [Run It!](#️-run-it)
- [How the Pipeline Works](#-how-the-pipeline-works-step-by-step)
- [Key Improvements Over Baseline](#-key-improvements-over-baseline)
- [Submission Checklist](#-submission-checklist)
- [Notes on Scoring](#-notes-on-scoring)
- [Contributing](#-contributing)
- [Contributors](#-contributors)

---

## 🏆 What's Included

This solution includes everything from the starter kit plus:

- **Complete end-to-end pipeline**: `upgraded_pipeline.py` for Gold/Platinum performance
- **Modular components (in `solution/` directory)**:
  - `loader.py` — Village data loading, plot lookup, area calculation, neighbor detection
  - `patch_extraction.py` — Extract satellite image and boundary mask patches for any plot
  - `boundary_detection.py` — Advanced boundary detection using CLAHE enhancement, multi-scale Canny, and morphological operations
  - `alignment.py` — Global alignment (using example truths if available) + local refinement via grid search
  - `area_validation.py` — Validate plot area against recorded area
  - `neighbor_consistency.py` — Check alignment consistency with neighboring plots
  - `confidence_calibration.py` — Combine multiple factors into final confidence score
  - `prediction_generator.py` — Generate contract-valid predictions
  - `pipeline.py` — Complete pipeline orchestration
- **Test files** — For each module to verify functionality
- **Transcripts folder** — `transcripts/` for AI conversations (per submission requirements)
- **Robust error handling** — Handles invalid data, missing files, and edge cases gracefully

---

## ⚙️ Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 / macOS 14 / Ubuntu 22.04 |
| Python | 3.10 | 3.12+ |
| RAM | 4 GB | 8 GB+ |
| Disk Space | 2 GB free | 5 GB+ free (for village data) |
| Internet | Required (for `uv` and data download) | Stable broadband |

### Python Version

This project uses **Python 3.12** (see `.python-version`). The project is managed with [uv](https://docs.astral.sh/uv/), which handles the virtual environment and all dependencies automatically.

### Python Dependencies

All dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Key packages:

| Package | Purpose |
|---|---|
| `geopandas` | Geospatial data loading and manipulation |
| `rasterio` | Reading satellite imagery `.tif` files |
| `shapely` | Geometric operations on plot boundaries |
| `numpy` | Numerical array operations |
| `scipy` | Scientific computing (e.g., grid search, statistics) |
| `opencv-python` | Image processing (CLAHE, Canny edge detection) |
| `scikit-learn` | Isotonic regression for confidence calibration |

> **Note:** You do **not** need to install these manually. Running `uv sync` (see [Setup](#-setup)) installs everything automatically.

### Special Requirements

- **Village Data** — Two village bundles must be downloaded separately from the BhuMe challenge site's *Get Started* page. These are **not** included in the repository due to size:
  - `34855_vadnerbhairav_chandavad_nashik/`
  - `Malatavari/`

  Each bundle must contain all 4 files:
  1. `input.geojson` — Official (shifted) plots
  2. `imagery.tif` — Satellite mosaic
  3. `boundaries.tif` — Optional auto-detected field hints
  4. `example_truths.geojson` — Hand-aligned examples for self-scoring

- **AI Transcripts** — Per submission requirements, any AI conversation transcripts used during development must be placed in the `transcripts/` folder before submission.

- **Video Recording** — A 5-minute video explaining your approach and results is required at submission time (not stored in this repo).

---

## 🚀 Setup

This solution uses [uv](https://docs.astral.sh/uv/), a blazing-fast Python package manager and project manager.

### Install uv (if you haven't already)

Follow the instructions at <https://docs.astral.sh/uv/getting-started/installation/>

### Sync dependencies

Once uv is installed, from the repository root run:

```bash
uv sync
```

This will:

1. Create a local virtual environment (no activation needed — just prefix **all** commands with `uv run`)
2. Install all required dependencies

### Important: Always use `uv run`!

Never run scripts directly with `python` or `python.exe`. Always use:

```bash
uv run <script_name>
```

This ensures you use the correct virtual environment with all dependencies installed.

---

## 📦 Get and Organize the Data

### Step 1: Download Village Bundles

Download both village bundles from the challenge site's **Get Started** page.

### Step 2: Save and Organize

Each village should be in its own subfolder inside `data/`. Ensure all 4 required files are present and non-empty for each village.

Expected directory structure:

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

Confirm data is loaded correctly (check plot counts):

```bash
uv run check_truths.py
```

---

## 🏃‍♀️ Run It!

### Quickstart (Naive Baseline from Starter Kit)

Run the naive baseline first to see the floor to beat:

```bash
uv run quickstart.py data/34855_vadnerbhairav_chandavad_nashik
```

### Our Upgraded Platinum Solution

Run the full pipeline to generate predictions.

For Vadnerbhairav:

```bash
uv run upgraded_pipeline.py data/34855_vadnerbhairav_chandavad_nashik
```

For Malatavari:

```bash
uv run upgraded_pipeline.py data/Malatavari
```

This will create `predictions.geojson` in the village's directory.

### Check What's in the Data

```bash
uv run check_truths.py
```

---

## 🔬 How the Pipeline Works (Step by Step!)

1. **Load Village Data**
   - Load `input.geojson` (ensures correct EPSG:4326 CRS)
   - Load `imagery.tif` and `boundaries.tif`
   - Load example truths (if present) for global alignment

2. **Compute Global Alignment**
   - If example truths are available, calculate a robust median shift (reduces outliers)
   - Applies this shift as an initial guess for all plots

3. **Process Each Plot Individually**
   - Extract a satellite image patch around the plot (+40 m padding)
   - Detect real field boundaries using multi-scale Canny edge detection + CLAHE enhancement
   - Refine alignment using grid search (translation and rotation)
   - Calculate alignment score, edge strength, area validity, etc.

4. **Confidence Calibration**
   - Combine multiple factors (alignment score, area score, edge strength, neighbor consistency)
   - If example truths are available, uses Isotonic Regression for better calibration

5. **Generate Predictions**
   - Uses a confidence threshold of 0.7 to decide whether to correct or flag
   - Creates valid GeoJSON with all required fields

---

## 📊 Key Improvements Over Baseline

| Feature | Baseline | Platinum Solution |
|---|---|---|
| Alignment | Global median shift only | Global shift + local grid search refinement |
| Boundary Detection | None | Multi-scale Canny + CLAHE + morphological operations |
| Confidence | Flat (all 0.5) | Calibrated, multi-factor confidence score |
| Area Validation | None | Validates plot area vs recorded area, flags invalid cases |
| Neighbor Consistency | None | Checks if shift matches neighboring plots |
| Error Handling | Limited | Robust — handles missing data and failed plots gracefully |

---

## 📝 Submission Checklist

- [ ] Download both villages into `data/` with valid `input.geojson` and `example_truths.geojson`
- [ ] Run `upgraded_pipeline.py` on both villages to generate `predictions.geojson`
- [ ] Verify `predictions.geojson` files are contract-valid
- [ ] Add AI transcripts to the `transcripts/` directory
- [ ] Record a 5-minute video explaining your approach and results
- [ ] Clean up any temporary/test files (optional but good practice)
- [ ] Commit your changes
- [ ] Submit the whole repository

---

## 📖 Notes on Scoring

The `score()` function from the starter kit mirrors the challenge's objective metrics:

- Uses **IoU** (Intersection over Union) to measure accuracy
- Checks for improvement over official positions
- Evaluates confidence calibration (higher confidence should mean higher IoU)
- Measures restraint (don't move already-correct plots too much!)

> ⚠️ Only run against the public example truths as a rough directional check. Don't overfit!

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/bhume-starter-kit.git
   cd bhume-starter-kit
   ```
3. **Install dependencies**:
   ```bash
   uv sync
   ```
4. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and add tests where applicable (see `test_*.py` files for examples)
6. **Run the tests** to make sure nothing is broken:
   ```bash
   uv run pytest
   ```
7. **Commit** your changes with a clear message:
   ```bash
   git commit -m "Add: brief description of your change"
   ```
8. **Push** to your fork and open a **Pull Request** against the `main` branch

### Contribution Guidelines

- Follow existing code style (PEP 8)
- Add or update docstrings for any new functions
- Keep PRs focused — one feature or fix per PR
- Update this README if your change affects setup or usage
- Place any AI-assisted development transcripts in the `transcripts/` folder

### Reporting Issues

Found a bug or have a suggestion? Please [open an issue](https://github.com/Sreddy08840/bhume-starter-kit/issues) with:
- A clear title and description
- Steps to reproduce (if it's a bug)
- Your OS, Python version, and any error messages

---

## 👥 Contributors

Thanks to everyone who has contributed to this project!

| Name | GitHub | Role |
|---|---|---|
| Sreddy08840 | [@Sreddy08840](https://github.com/Sreddy08840) | Project Owner / Lead Developer |

> Want to see your name here? Check out the [Contributing](#-contributing) section above and submit a pull request!

---

*Built for the BhuMe Boundary Challenge — helping bring Maharashtra's land records into the real world, one plot at a time.*