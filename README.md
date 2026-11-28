# **Bayesian Forecasting of Inter-Event Intervals among Historical Global Biological Events**

This repository contains the self-contained code required to reproduce the computational analysis and figures for the manuscript: **"Bayesian forecasting of inter-event intervals among historical global biological events"**.

## **📁 Repository Structure**

```
├── main.py            # Self-contained pipeline (dataset N=38 embedded, no external data needed)
├── data/raw/data.csv  # Raw dataset of historical global biological events (N=38)
├── results/           # Output directory for generated figures and logs
├── requirements.txt   # Pinned dependencies for exact reproducibility
├── License.txt        # MIT License
└── README.md          # This file
```

## **🔬 Methods Overview**

* **Data:** Validated dataset of N=38 historical global biological events (1492–2020). A **Global Anthropogenic Index** (population, urbanization, trade openness) is extracted via PCA to remove multicollinearity.
* **Main model:** Shallow LSTM (16 units) with dual uncertainty quantification — **aleatoric** (Gaussian negative log-likelihood loss, predicting μ and log σ² for interval, severity, and duration) and **epistemic** (Monte Carlo Dropout).
* **Baselines:** Ridge regression, Random Forest, non-homogeneous Poisson process (NHPP, exponential acceleration fit), and Weibull survival model.
* **Validation:** Expanded walk-forward cross-validation over the last 15 events (MAE/RMSE for all models).
* **Forecast:** 3000 MC simulations with **causality-constrained rejection sampling** (predicted onset year ≥ current year) and percentile-based 95% CI.

## **⚙️ System Requirements**

* **Operating System:** Cross-platform (Linux, macOS, Windows).
* **Hardware:** Standard consumer-grade computer (4+ CPU cores, 8GB+ RAM). GPU is optional but not required due to the shallow network architecture.
* **Software:** Python 3.10.x

## **🚀 Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/Kurban123/pandemic-occurrence-prediction
   cd pandemic-occurrence-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the strict dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Expected installation time: < 2 minutes (TensorFlow download may take longer on slow connections).*

## **🏃 Execution**

To run the entire pipeline (PCA index extraction, walk-forward validation, final training, 3000 MC simulations, and figure generation):

```bash
python main.py
```

*Expected runtime: ~10–30 minutes on a standard modern CPU (the walk-forward loop retrains the LSTM 15 times).*

### **Expected Output**

1. **Console Logs:** Walk-forward validation results (MAE for LSTM, Ridge, Random Forest, NHPP, Weibull; RMSE for LSTM) and the final scientific summary: predicted onset-year window with 95% CI, severity index, and expected duration.
2. **Figures:** High-resolution vector figure saved as `Figure_2_HighRes_Updated.pdf` in the working directory (panels A–D: historical acceleration, anthropogenic index vs severity, correlation matrix, index vs duration/severity; panels E–G: forecast distribution, threat-profile radar chart, log–log cluster analysis).

## **📄 License**

This software is licensed under the MIT License. See the License.txt file for details.
