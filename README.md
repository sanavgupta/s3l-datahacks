# 🌊 Marine Eco-Forecast: Interactive Trophic Survey

An ML-powered diagnostic and prognostic dashboard visualizing 70 years of **CalCOFI** biological observations and projecting future habitat suitability for the California Current ecosystem.

---

## 🚀 Quick Start

Because the full database containing 2024 projections is large, this repository provides a reproducible pipeline to generate the forecast locally.

### 1. Prerequisites

Ensure you have **Python 3.10+** installed. This project utilizes modern libraries for high-performance mapping.

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/sanavgupta/s3l-datahacks.git
cd s3l-datahacks
pip3 install -r requirements.txt
```

### 3. Generate the 2024 Forecast

Run the predictive engine to train the HistGradientBoosting Classifier and inject the 12-month 2024 projections into the SQLite database:

```bash
python3 predictive_model.py
```

> **What this does:** Processes historical data, engineers lagged environmental features, trains the ML model, and saves a cleaned 2024 forecast to `marine_observations.db`.

### 4. Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

> **Note:** Use the **🔄 Hard Reload** button in the sidebar once the app launches to ensure the new 2024 data is visible on the timeline.

---

## 🔬 Methodology & Technical Architecture

### The ML Pipeline

Instead of standard linear regressions, we utilize a **HistGradientBoosting Classifier** to handle the non-linear relationships between ocean physics and biology.

- **Feature Engineering:** The model uses Current Temperature, Salinity, Month, and Lagged Environmental Variables ($T_{lag1}$, $S_{lag1}$) to capture the "memory" of the ecosystem.
- **Spatial Filtering:** To ensure visual clarity and biological realism, the model implements a **0.75 probability threshold** and spatial thinning (1:5 sampling), visualizing only high-confidence habitat core ranges.

### Performance Optimization

To handle 450,000+ historical records smoothly on a local machine, the dashboard utilizes:

- **Dictionary-based Caching:** Pre-split data structures allow for O(1) lookup times for slider updates.
- **Streamlit Fragments:** Isolating the map render-loop from the rest of the UI to prevent full-page refreshes during animation.
- **WebGL Rendering:** GPU acceleration via Plotly's Mapbox engine for high-density scatter plots.

---

## 📂 Project Structure

```
marine-eco-forecast/
├── app.py                   # High-performance Streamlit dashboard
├── predictive_model.py      # ML training and 1-year projection engine
├── marine_observations.db   # SQLite database with 70 years of CalCOFI data
└── requirements.txt         # Python dependencies
```

---

## This opens the project without future predictions as a feature, come in person to view projections. 

## 🌟 Inspiration & Impact

The health of the California Current is determined by its smallest residents. Larval fish and zooplankton are the biological engine of our coast. We built this tool to turn CalCOFI's "Big Data" into a "Big Picture" for ocean conservation — moving marine science from a diagnostic archive to a prognostic engine.

*Developed for DataHacks 2026.*
