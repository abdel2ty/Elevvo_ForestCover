# ForestIQ — Forest Cover Type Classification

A machine learning web application that classifies forest cover types in the Roosevelt National Forest using cartographic and topographic features. Built with Streamlit and scikit-learn, featuring multi-class probability output, sensitivity analysis, and a comprehensive reference guide.

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-8B5CF6?style=flat-square)](https://elevvo-forest-cover.streamlit.app/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle_Notebook-abdel2ty-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/code/abdel2ty/elevvo-forest-cover)

---

## Overview

ForestIQ uses a **Random Forest Classifier** trained on the UCI Covertype dataset to classify land areas into one of seven forest cover types based on 54 cartographic variables — including elevation, slope, aspect, distances to hydrological features and roads, hillshade indices, wilderness area designation, and soil type. The application makes this multi-class problem accessible through an intuitive input panel, full probability distribution charts, feature sensitivity sweeps, and a detailed forest type reference.

---

## Application Pages

**Classify** — Input 11 key terrain parameters to receive a predicted forest cover type with confidence percentage. A full probability bar chart displays the model's confidence across all seven classes simultaneously.

**Explore** — An elevation sweep chart shows how changing altitude shifts class probabilities across the full 1,800–3,900m range. Four additional sensitivity curves cover Slope, Horizontal Distance to Hydrology, and Horizontal Distance to Roads, each with a live marker at the current input value.

**Reference** — Detailed cards for all seven cover types with ecological descriptions. Includes a full feature reference table covering all 10 feature groups, their value ranges, data types, and ecological significance. Dataset overview statistics: 7 classes, 54 features, 4 wilderness areas, 40 soil types.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Library | scikit-learn |
| Output Classes | 7 (multi-class) |
| Feature Groups | Continuous topographic + Binary (wilderness area, soil type) |
| Model Persistence | joblib |

---

## Forest Cover Types

| Type | Name |
|---|---|
| 0 | Spruce / Fir |
| 1 | Lodgepole Pine |
| 2 | Ponderosa Pine |
| 3 | Cottonwood / Willow |
| 4 | Aspen |
| 5 | Douglas-Fir |
| 6 | Krummholz |

---

## Input Features

| Feature | Range | Type | Description |
|---|---|---|---|
| Elevation | 1800 – 3900m | Numeric | Primary predictor; determines temperature and moisture regime |
| Aspect | 0 – 360° | Numeric | Compass direction the slope faces; affects sun exposure |
| Slope | 0 – 52° | Numeric | Steepness of terrain; affects drainage and erosion |
| H-Dist to Hydrology | 0 – 1400m | Numeric | Horizontal distance to the nearest water feature |
| V-Dist to Hydrology | -150 – 600m | Numeric | Vertical distance to the nearest water feature |
| H-Dist to Roads | 0 – 7000m | Numeric | Distance from road network; proxy for human influence |
| H-Dist to Fire Points | 0 – 7000m | Numeric | Distance to historical fire ignition points |
| Hillshade (9am / Noon / 3pm) | 0 – 254 | Numeric | Shadow index at three times of day |
| Wilderness Area | 4 types | Binary | One-hot encoded wilderness area designation |
| Soil Type | 40 types | Binary | One-hot encoded USFS soil classification |

---

## Tech Stack

| Component | Technology |
|---|---|
| App Framework | Streamlit |
| ML Model | scikit-learn — RandomForestClassifier, StandardScaler |
| Visualizations | Plotly |
| Model Persistence | joblib, JSON |

---

## Local Setup

```bash
git clone <your-repo-url>
cd forestiq

pip install streamlit numpy plotly scikit-learn joblib

streamlit run app3.py
```

---

## Links

| | |
|---|---|
| Live Application | https://elevvo-forest-cover.streamlit.app/ |
| Kaggle Notebook | https://www.kaggle.com/code/abdel2ty/elevvo-forest-cover |

---

*Built by @abdel2ty as part of the Elevvo ML project series.*
