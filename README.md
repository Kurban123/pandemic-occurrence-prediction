# **Pandemic Occurrence and Severity Prediction**

This repository contains the code, data, and models required to reproduce the computational analysis and figures for the manuscript: **"Anthropogenic drivers accelerate the recurrence of global biological threats"**.  
The codebase adheres to the strict reproducibility and computational tool reporting guidelines of Nature Portfolio journals.

## **📁 Repository Structure**

├── data/  
│   └── raw/  
│       └── pandemics.csv       \# Raw dataset of historical pandemics (N=38)  
├── results/  
│   ├── figures/                \# Output directory for generated high-res plots  
│   └── logs/                   \# Execution logs  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── data.py                 \# Data loading, preprocessing, and sequencing  
│   ├── models.py               \# LSTM architecture and baseline models  
│   ├── utils.py                \# Reproducibility settings and logging  
│   └── visualization.py        \# Code for generating publication-ready figures  
├── main.py                     \# Main execution script  
├── requirements.txt            \# Pinned dependencies for exact reproducibility  
├── LICENSE                     \# Open Source Initiative (OSI) approved license  
└── README.md                   \# This file

## **⚙️ System Requirements**

* **Operating System:** Cross-platform (Linux, macOS, Windows). Tested on Ubuntu 22.04 LTS.  
* **Hardware:** Standard consumer-grade computer (e.g., 4+ CPU cores, 8GB+ RAM). GPU is optional but not required due to the shallow network architecture.  
* **Software:** Python 3.10.x

## **🚀 Installation**

1. Clone the repository:  
   git clone https://github.com/Kurban123/pandemic-occurrence-prediction  
   cd pandemic-occurrence-prediction

2. Create a virtual environment and activate it:  
   python3 \-m venv venv  
   source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

3. Install the strict dependencies:  
   pip install \-r requirements.txt

   *Expected installation time: \< 2 minutes.*

## **🏃 Execution**

To run the entire pipeline (data preprocessing, cross-validation, final training, Monte Carlo simulations, and figure generation):  
python main.py

*Expected runtime: \< 1 minute on a standard modern CPU.*

### **Expected Output**

1. **Console Logs:** Detailed output of walk-forward validation (MAE and RMSE for LSTM, Ridge, and Random Forest) and the final scientific summary with 95% CIs.  
2. **Figures:** High-resolution vector (.pdf) and raster (.tiff with LZW compression) files saved automatically in the results/figures/ directory.

## **📄 License**

This software is licensed under the MIT License. See the LICENSE file for details.