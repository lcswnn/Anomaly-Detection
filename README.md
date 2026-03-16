# Space Image Anomaly Detection

An unsupervised anomaly detection pipeline that uses a **Convolutional Autoencoder** to identify unusual objects in astronomical images from the [Sloan Digital Sky Survey (SDSS)](https://www.sdss.org/). The system detects anomalies such as saturated stars, satellite trails, galaxy mergers, and other scientifically interesting phenomena — then exports them for citizen science classification on [Zooniverse](https://www.zooniverse.org/).

---

## How It Works

### The Idea

Most astronomical images contain ordinary stars and galaxies. An autoencoder trained on these "normal" images learns to reconstruct them well — but struggles to reconstruct unusual objects it hasn't seen before. By measuring the **reconstruction error** (how different the output is from the input), we can flag tiles with high error as anomalies.

### Pipeline Overview

```
SDSS Database ──► Download FITS ──► Preprocess & Tile ──► Train Autoencoder
                                                                │
                Zooniverse ◄── Export PNGs ◄── Detect Anomalies ◄┘
```

1. **Data Acquisition** — Query and download ~250 FITS image files from the SDSS DR17 SkyServer API (r-band, ~1.4 GB)
2. **Preprocessing** — Decompress FITS files, apply asinh stretch normalization, crop edges, and slice into 128x128 pixel tiles. Blank sky tiles are filtered out. Result: ~70,000 tiles stored in a single HDF5 file
3. **Training** — A convolutional autoencoder (encoder + decoder with ~2.9M parameters) is trained using MSE loss to reconstruct the tiles. Early stopping and learning rate scheduling prevent overfitting
4. **Anomaly Detection** — Reconstruction error is computed for every tile. Tiles above the 97th percentile error threshold are flagged as anomalies (~2,100 tiles)
5. **Quality Checks** — Anomalies are sub-classified (saturated star, satellite trail, extended source, unknown), the threshold is validated via bootstrap analysis, and false negative candidates are surfaced
6. **Export & Upload** — Anomalous tiles are exported as PNG images with a manifest CSV, then uploaded to Zooniverse for citizen science classification

### Data Source

All image data comes from the **Sloan Digital Sky Survey (SDSS) Data Release 17**. The download script queries the SDSS SkyServer SQL API (`skyserver.sdss.org`) — a public API that requires no authentication — for verified (run, camcol, field) image combinations in the r-band.

### Model Architecture

```
Input: 1x128x128 grayscale tile

Encoder:
  Conv2d(1→32)   + BatchNorm + ReLU    →  32x64x64
  Conv2d(32→64)  + BatchNorm + ReLU    →  64x32x32
  Conv2d(64→128) + BatchNorm + ReLU    → 128x16x16
  Conv2d(128→256)+ BatchNorm + ReLU    → 256x8x8
  Conv2d(256→512)+ BatchNorm + ReLU    → 512x4x4  (bottleneck)

Decoder:
  ConvT(512→256) + BatchNorm + ReLU    → 256x8x8
  ConvT(256→128) + BatchNorm + ReLU    → 128x16x16
  ConvT(128→64)  + BatchNorm + ReLU    →  64x32x32
  ConvT(64→32)   + BatchNorm + ReLU    →  32x64x64
  ConvT(32→1)    + Sigmoid             →   1x128x128

Output: 1x128x128 reconstructed tile
```

The model supports **Apple Silicon (MPS)**, **NVIDIA CUDA**, and **CPU** backends automatically.

---

## Project Structure

```
Anomaly-Detection/
├── src/
│   ├── model.py              # Autoencoder architecture (Encoder + Decoder)
│   ├── dataset.py            # PyTorch Dataset wrapper for HDF5 tiles
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Anomaly detection & quality checks
│   ├── quality_checks.py     # Sub-classification, bootstrap, false negatives
│   ├── tile_processing.py    # FITS → HDF5 preprocessing pipeline
│   ├── download-images.py    # Async SDSS FITS file downloader
│   ├── export_images.py      # Export anomaly tiles as individual PNGs
│   ├── view_anomlaies.py     # Visualize top anomalies as a mosaic
│   ├── auto-manifest.py      # Generate manifest CSV for Zooniverse
│   └── example-data-pull.py  # Example FITS file exploration
├── data/
│   ├── raw/                  # Downloaded FITS files (not tracked by git)
│   └── tiles/                # Processed HDF5 tiles (not tracked by git)
├── models/                   # Trained weights, visualizations, reports
├── zoon_imgs/                # Exported anomaly PNGs (not tracked by git)
├── manifest.csv              # Metadata CSV for Zooniverse upload
├── upload_to_zooniverse.py   # Upload anomalies to Zooniverse
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Git**
- ~4 GB of free disk space (for raw data, tiles, and model outputs)

### Clone the Repository

```bash
git clone https://github.com/lucaswaunn/Anomaly-Detection.git
cd Anomaly-Detection
```

### Set Up the Environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note (Windows):** If you encounter issues installing PyTorch, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for platform-specific install commands. For CPU-only on Windows:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```
> Then install the remaining dependencies with `pip install -r requirements.txt`.

### Run the Pipeline

All scripts are run from the `src/` directory:

```bash
cd src
```

#### Step 1 — Download SDSS Images

```bash
python download-images.py
```

Downloads ~250 FITS files (~1.4 GB) from the SDSS SkyServer into `data/raw/`. Uses async requests with 10 concurrent connections.

#### Step 2 — Preprocess into Tiles

```bash
python tile_processing.py
```

Decompresses FITS files, normalizes the images (asinh stretch + min-max scaling), slices them into 128x128 tiles, filters out blank sky, and saves everything to `data/tiles/tiles.h5`.

#### Step 3 — Train the Autoencoder

```bash
python train.py
```

Trains for up to 30 epochs with early stopping (patience of 5). Outputs:
- `models/best_model.pth` — best weights (lowest validation loss)
- `models/final_model.pth` — weights after the final epoch
- `models/loss_curve.png` — training/validation loss plot

#### Step 4 — Evaluate & Detect Anomalies

```bash
python evaluate.py
```

Computes reconstruction errors for all tiles, sets the anomaly threshold (97th percentile), and runs quality checks including sub-classification, bootstrap threshold validation, and false negative mining.

To skip quality checks:

```bash
python evaluate.py --skip-quality-checks
```

Outputs in `models/`:
- `error_distribution.png` — histogram of reconstruction errors with threshold line
- `anomalies.png` — top 16 anomalous tiles
- `reconstructions.png` — input vs. reconstruction comparison
- `subclass_mosaic.png` — anomalies colored by sub-class
- `threshold_robustness.png` — bootstrap confidence interval
- `false_negatives.png` — candidate missed anomalies
- `quality_report.txt` — text summary of all checks

#### Step 5 — Export & Upload (Optional)

```bash
python export_images.py        # Export anomaly tiles as PNGs to zoon_imgs/
python auto-manifest.py        # Generate manifest.csv
python ../upload_to_zooniverse.py  # Upload to Zooniverse (requires credentials)
```

---

## Outputs & Visualizations

After running the full pipeline, the `models/` directory contains:

| File | Description |
|------|-------------|
| `best_model.pth` | Trained model weights (lowest validation loss) |
| `loss_curve.png` | Training vs. validation loss over epochs |
| `error_distribution.png` | Histogram of per-tile reconstruction errors |
| `anomalies.png` | Top 16 most anomalous tiles |
| `anomaly_mosaic.png` | Top 64 anomalies ranked by reconstruction error |
| `reconstructions.png` | Side-by-side original vs. reconstructed tiles |
| `subclass_mosaic.png` | Anomalies colored by sub-classification type |
| `threshold_robustness.png` | Bootstrap analysis of threshold stability |
| `false_negatives.png` | 64 candidate false negatives for manual review |
| `quality_report.txt` | Full text report of all quality checks |

---

## Configuration

Key hyperparameters can be adjusted at the top of each script:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `TILE_SIZE` | `tile_processing.py` | 128 | Tile dimensions in pixels |
| `MIN_STD` | `tile_processing.py` | 0.01 | Minimum std to keep a tile (filters blank sky) |
| `BATCH_SIZE` | `train.py` | 64 | Training batch size |
| `EPOCHS` | `train.py` | 30 | Maximum training epochs |
| `LEARNING_RATE` | `train.py` | 1e-3 | Adam optimizer learning rate |
| `PATIENCE` | `train.py` | 5 | Early stopping patience |
| `ANOMALY_PERCENTILE` | `evaluate.py` | 97 | Percentile threshold for anomaly detection |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
