# Lightning Prediction Pipeline (Karin Pitlik)

Overview

This project focuses on building a machine learning pipeline to predict lightning occurrence over Israel and the Eastern Mediterranean using atmospheric model outputs.

The workflow integrates:
	•	Numerical weather prediction model data (ensemble-based)
	•	Observational lightning data (ENTLN / ILDN)
	•	Spatial-temporal aggregation into structured grids
	•	Deep learning (UNet) for probabilistic lightning prediction

The goal is to generate spatial probability maps of lightning occurrence and evaluate them using meteorological verification metrics such as CSI, FAR, POD, and Brier Score.

⸻

🧠 Problem Definition

Given atmospheric variables (e.g. CAPE, KI, LPI, etc.):

➡️ Predict whether lightning occurs in each grid cell
➡️ Output probabilistic maps
➡️ Compare predictions to real lightning observations

⸻

📦 Data Sources

1. Model Data (Ensemble)

Atmospheric variables from numerical weather prediction models:
	•	LPI (Lightning Potential Index)
	•	KI (K Index)
	•	DS (Dynamic Scheme: lneg + lpos + lneu)
	•	CAPE2D
	•	PREC_RATE
	•	FLUX_UP
	•	WMAX_LAYER

Each variable is stored as NetCDF files per timestamp and ensemble member.

⸻

2. Lightning Observations

ENTLN (Earth Networks)
	•	Pulse-based lightning detections
	•	Processed into spatial grids using histogram binning

ILDN (Israeli Lightning Detection Network)
	•	Ground-based lightning observations
	•	Also mapped to model grid

⸻

🗺️ Spatial & Temporal Processing

Spatial Grid
	•	Base grid: ~4 km resolution (from model coordinates)
	•	Downscaled to:
	•	4 km
	•	12 km
	•	24 km
	•	40 km

Aggregation methods:
	•	Mean (for most variables)
	•	Sum (for precipitation / DS)

⸻

Temporal Aggregation
	•	Data grouped into fixed intervals (e.g. 1-hour)
	•	Within each interval:
	•	Model variables aggregated over time
	•	Lightning counts aggregated per grid cell

⸻

⚙️ Pipeline Structure

1. Preprocessing (MATLAB)

Main responsibilities:
	•	Read raw NetCDF model files
	•	Separate ensemble members
	•	Aggregate spatially and temporally
	•	Map lightning observations to grid
	•	Export processed NetCDF files per interval
