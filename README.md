# ✈️ Predictive Maintenance of Turbofan Engines

**Tools:** Python · pandas · NumPy · scikit-learn · matplotlib · seaborn  
**Dataset:** NASA CMAPSS FD001  
**Type:** Machine Learning  Regression + Clustering

---

## Project Overview

Built a complete end-to-end machine learning pipeline to predict 
the Remaining Useful Life (RUL) of turbofan jet engines using 
NASA's CMAPSS FD001 dataset.

The project covers data loading, cleaning, exploratory data 
analysis, feature engineering, regression modeling, 
and unsupervised clustering.

---

## Results

Evaluated on the 100 FD001 test engines (last available cycle per engine), with true RUL capped at 125 cycles.

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression (baseline) | 20.92 cycles | 16.39 cycles |
| **Random Forest (best)** | **17.97 cycles** | **12.74 cycles** |

- **14.1% RMSE improvement** over Linear Regression, driven mainly by rolling-mean features rather than model choice

### In context

Reported FD001 results under a similar piecewise-linear (capped) RUL target, for reference:

| Method | FD001 RMSE | Source |
|---|---|---|
| CNN | 18.45 | Babu et al., 2016 |
| **This work: Random Forest + rolling features** | **17.97** | |
| Deep LSTM | 16.14 | Zheng et al., 2017 |
| Deep CNN (time-window input) | 12.61 | Li et al., 2018 |

This Random Forest is a **competitive classical baseline**, in line with published tree-based and early deep-learning results, but it does not reach state-of-the-art sequence models. Published numbers use slightly different RUL caps (125–130) and preprocessing, so the comparison is indicative, not exact.

---

## Dataset

**NASA CMAPSS FD001**  Commercial Modular Aero-Propulsion System Simulation

| File | Content |
|---|---|
| `train_FD001.txt` | 20,631 rows  100 engines run to failure |
| `test_FD001.txt` | 13,096 rows  engines cut off mid-life |
| `RUL_FD001.txt` | 100 true RUL values (ground truth) |

26 columns per row: engine_id, cycle, 3 operational settings, 21 sensor readings.

---

## Project Pipeline

### 1. Data Loading & Cleaning
- Loaded space-separated files with manual column naming
- Checked for missing values  zero found
- Dropped 7 constant sensors (std < 0.001)
- Dropped sensor_6 (cycle correlation r = 0.11  no degradation signal)
- **Result: 14 useful sensors retained**

### 2. Exploratory Data Analysis
- Engine lifetime distribution: 128 to 362 cycles (mean 206)
- Sensor trend plots: sensors 4, 11, 15 rise with age; sensors 7, 12, 20, 21 fall
- Correlation heatmap: sensor_11 highest cycle correlation (r = 0.63)
- Z-score outlier analysis: max 2.5% outlier rate  kept all (genuine operational extremes)

### 3. Feature Engineering
- **RUL column:** max_cycle − current_cycle, capped at 125
- **Rolling averages:** 5-cycle rolling mean on 7 key sensors per engine
- **Z-score normalization:** StandardScaler fitted on training data only (data leakage prevention)

### 4. Regression Modeling
- Linear Regression as baseline
- Random Forest (100 trees, random_state=42)
- Evaluated with RMSE and MAE on test set
- Feature importance analysis

### 5. K-Means Clustering
- Degradation profiles built from last 20 cycles per engine
- Elbow method determined k=3
- PCA reduced 14D to 2D for visualization
- 3 clusters: Fast Degraders (avg 194.5 cycles), Medium (215.2), Slow (217.9)

---

## Key Findings

**Biggest Finding:**
> Raw sensor_4 importance = 0.01 (ranked 19th).  
> sensor_4_roll5 importance = 0.63 (ranked 1st).  
> Same sensor, one rolling-average transformation: the smoothed version carries 63× the importance of the raw signal.

**Feature Engineering matters more than Algorithm Choice here:**
Smoothing noisy sensor channels did more for accuracy than switching
from a linear to a non-linear model. The natural next step is a
sequence model (1D-CNN / LSTM) on time windows, which is where the
published state of the art on FD001 sits.

**Safety Critical Finding:**
Linear Regression over-predicts RUL near failure  it tells 
operators an engine is fine when it is actually close to failing. 
Random Forest is significantly safer in the low-RUL zone.

**Clustering Surprise:**
Clusters 1 and 2 have nearly identical lifetimes (215 vs 218 cycles) 
but completely different sensor fingerprints  multiple degradation 
pathways can lead to the same total engine lifespan.

---

## Plots

| Plot | Description |
|---|---|
| ![Lifetime Distribution](plot1_engine_lifetimes.png) | Engine lifetime histogram and sorted bar |
| ![Sensor Trends](plot2_sensor_trends.png) | Sensor degradation trends over cycle |
| ![Correlation Heatmap](plot3_correlation_heatmap.png) | Sensor correlation matrix |
| ![Boxplots](plot4_boxplots.png) | Outlier detection across all sensors |
| ![RUL Distribution](plot5_RUL_distribution.png) | RUL histogram and countdown |
| ![Scatter Plots](plot6_predictions_vs_actual.png) | Predicted vs Actual RUL  both models |
| ![Feature Importance](plot7_feature_importance.png) | Random Forest feature importance |
| ![Elbow Plot](plot8_elbow.png) | K-Means elbow method |
| ![PCA Scatter](plot9_clusters_pca.png) | Engine degradation clusters (2D) |
| ![Lifetime Boxplots](plot10_cluster_lifetimes.png) | Lifetime by cluster |

---

## Limitations and Next Steps

**Limitations**
- Single sub-dataset (FD001: one operating condition, one fault mode); results may not transfer to FD002–FD004.
- Each test engine is represented by its last cycle only; no temporal model of the degradation trajectory.
- Single train/test run; no cross-validation over engines or uncertainty on the reported RMSE.
- C-MAPSS is simulated data, so conclusions are methodological rather than operational.

**Next steps**
1. Validate on FD002–FD004 (multiple operating conditions and fault modes), with operating-condition normalisation
2. Sequence models (1D-CNN / LSTM) on sliding windows, for a like-for-like comparison with the literature
3. Report the asymmetric PHM08 score alongside RMSE, since late predictions are costlier than early ones
4. Add predictive uncertainty (e.g. quantile regression forests) so a maintenance threshold can be set with a known risk

---

## References

- Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* PHM 2008.
- Babu, G. S., Zhao, P., Li, X.-L. (2016). *Deep convolutional neural network based regression approach for estimation of remaining useful life.* DASFAA 2016.
- Zheng, S., Ristovski, K., Farahat, A., Gupta, C. (2017). *Long short-term memory network for remaining useful life estimation.* IEEE ICPHM 2017.
- Li, X., Ding, Q., Sun, J.-Q. (2018). *Remaining useful life estimation in prognostics using deep convolution neural networks.* Reliability Engineering & System Safety, 172, 1–11.

---

## Files in This Repository

| File | Description |
|---|---|
| `CMAPSS_Analysis.ipynb` | Complete Python notebook |
| `plot1_engine_lifetimes.png` | Engine lifetime charts |
| `plot2_sensor_trends.png` | Sensor degradation trends |
| `plot3_correlation_heatmap.png` | Correlation matrix |
| `plot4_boxplots.png` | Boxplot outlier analysis |
| `plot5_RUL_distribution.png` | RUL distribution |
| `plot6_predictions_vs_actual.png` | Model scatter plots |
| `plot7_feature_importance.png` | Feature importance chart |
| `plot8_elbow.png` | Elbow method plot |
| `plot9_clusters_pca.png` | PCA cluster scatter |
| `plot10_cluster_lifetimes.png` | Cluster lifetime boxplots |

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
```
