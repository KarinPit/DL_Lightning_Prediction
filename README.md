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

Key function:
process_model_files(...)

Output:
	•	One NetCDF file per:
	•	variable
	•	ensemble member
	•	time interval

⸻

2. Dataset Construction (Python / PyTorch)
	•	Load processed NetCDF data
	•	Stack variables into tensors:
X shape: (N, C, H, W)
y shape: (N, 1, H, W)

Where:
	•	N = number of time intervals
	•	C = number of atmospheric variables

⸻

3. Target Definition

Lightning is converted to binary classification:

y = (lightning_count > 0)
Meaning:
	•	1 → lightning occurred
	•	0 → no lightning

⸻

4. Normalization

Per-channel standardization:
x' = (x - mean) / std

Computed on training set only, then applied to:
	•	training
	•	validation

Important:
	•	NaNs are removed after normalization
	•	Small std values are stabilized

⸻

5. Model: UNet

Architecture:
	•	Encoder–decoder convolutional network
	•	Skip connections for spatial detail preservation
	•	Output: logits per pixel

Input:  (C, H, W)
Output: (1, H, W)

⸻

6. Training

Loss:
nn.BCEWithLogitsLoss

Pipeline:
	1.	Forward pass → logits
	2.	Apply sigmoid for probabilities
	3.	Threshold → binary predictions
	4.	Compute metrics

⸻

📊 Evaluation Metrics

Binary Metrics (threshold-based)
	•	CSI (Critical Success Index)
	•	FAR (False Alarm Ratio)
	•	POD (Probability of Detection)

Probabilistic Metrics
	•	Brier Score (BS)
	•	Brier Skill Score (BSS)

⸻

⚠️ Key Challenges

1. Extreme Class Imbalance
	•	Very few lightning pixels per map
	•	Majority of pixels are zeros

2. Spatial Uncertainty
	•	Lightning is localized and noisy
	•	Model tends to predict broader areas

3. Data Distribution Issues
	•	Some variables are sparse (e.g. LPI)
	•	Others have large dynamic ranges (e.g. CAPE)

4. False Positives
	•	Model often predicts correct regions
	•	But overestimates spatial extent

⸻

🔬 Current Status

✔ Pipeline working end-to-end
✔ Data preprocessing stable
✔ Normalization issues resolved
✔ Model learns spatial lightning regions

⚠ Remaining issues:
	•	High False Alarm Rate
	•	Calibration of prediction threshold
	•	Handling sparse signals
	•	Need for improved spatial representation

⸻

🚀 Next Steps
	1.	Improve target representation
	•	Gaussian smoothing of lightning
	•	Multi-scale labeling
	2.	Optimize threshold selection
	•	ROC / PR analysis
	•	Dynamic thresholding
	3.	Handle imbalance
	•	Loss weighting
	•	Focal loss (optional)
	4.	Feature analysis
	•	Understand model dependence on variables
	•	Interpret learned relationships
	5.	Experiment with resolutions
	•	Compare 4km vs 24km vs 40km grids
	6.	Advanced modeling
	•	UNet improvements
	•	Spatial attention mechanisms

⸻

🧩 Tech Stack
	•	MATLAB → preprocessing & NetCDF generation
	•	Python → data loading
	•	PyTorch → model training
	•	NumPy / SciPy → metrics & processing

⸻

🧠 Key Insight So Far

The model is not failing — it successfully learns:
	•	where lightning is likely to occur

But struggles with:
	•	precise localization
	•	false positives due to spatial uncertainty

⸻

👩‍💻 Author

Karin Pitlik
MSc Data Science & Machine Learning student
Lightning Prediction Research
