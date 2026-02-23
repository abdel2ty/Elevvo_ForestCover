# ✦ ForestIQ — Forest Cover Type Classification

> **Identify forest cover from cartographic data alone.** ForestIQ is an interactive ML-powered web application that classifies forest cover types in the Roosevelt National Forest using topographic and geographic features — powered by a Random Forest model and presented through a stunning glassmorphism interface.

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-8B5CF6?style=for-the-badge)](https://elevvo-forest-cover.streamlit.app/)
[![Kaggle Notebook](https://img.shields.io/badge/📓_Kaggle_Notebook-abdel2ty-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/abdel2ty/elevvo-forest-cover)

---

## 📌 Overview

ForestIQ leverages a **Random Forest Classifier** trained on the classic UCI Covertype dataset to classify land areas into one of 7 forest cover types based on 54 cartographic features — including elevation, slope, aspect, distance to water bodies and roads, hillshade indices, wilderness area, and soil type. The app makes this complex multi-class problem approachable with an intuitive input panel, probability bar charts, sensitivity curves, and a comprehensive forest type reference guide.

---

## ✨ Features

### 🌲 Classify Page
- Input 11 key terrain parameters to predict the forest cover type
- Displays top prediction with confidence percentage
- Full probability distribution bar chart across all 7 cover types
- Color-coded result cards matching each forest type's unique palette

### 🔬 Explore / Analytics Page
- **Elevation Sweep** — visualize how elevation changes shift class probabilities
- **4 Sensitivity Curves** — Elevation, Slope, H-Distance to Hydrology, H-Distance to Roads
- Live pink crosshair marking your current input values

### 📖 Reference Page
- Detailed cards for all 7 forest cover types with icons, descriptions, and accent colors
- Complete feature reference table: 10 feature groups, ranges, data types, and ecological descriptions
- Dataset overview KPIs: 7 classes, 54 features, 4 wilderness areas, 40 soil types

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Library | scikit-learn |
| Output Classes | 7 (multi-class) |
| Feature Groups | Topographic (continuous) + Wilderness Area (binary) + Soil Type (binary) |
| Persistence | joblib |

---

## 🌳 Forest Cover Types

| Type | Name | Accent Color |
|---|---|---|
| 0 | Spruce/Fir | 🟢 Green |
| 1 | Lodgepole Pine | 🔵 Blue |
| 2 | Ponderosa Pine | 🟡 Amber |
| 3 | Cottonwood/Willow | 🟠 Orange |
| 4 | Aspen | 🟣 Violet |
| 5 | Douglas-Fir | 🔴 Red |
| 6 | Krummholz | 💜 Fuchsia |

---

## 📥 Input Features

| Feature | Range | Type | Description |
|---|---|---|---|
| Elevation | 1800 – 3900m | Numeric | Primary predictor; determines temperature & moisture |
| Aspect | 0 – 360° | Numeric | Compass direction the slope faces |
| Slope | 0 – 52° | Numeric | Steepness of terrain |
| H-Dist to Hydrology | 0 – 1400m | Numeric | Horizontal distance to nearest water feature |
| V-Dist to Hydrology | -150 – 600m | Numeric | Vertical distance to nearest water feature |
| H-Dist to Roads | 0 – 7000m | Numeric | Distance from road network |
| H-Dist to Fire Points | 0 – 7000m | Numeric | Distance to historical fire ignition points |
| Hillshade (9am / Noon / 3pm) | 0 – 254 | Numeric | Shadow index at three times of day |
| Wilderness Area | 4 types | Binary | One-hot encoded wilderness designation |
| Soil Type | 40 types | Binary | One-hot encoded USFS soil classification |

---

## 🛠️ Tech Stack

- **Frontend / App Framework** — [Streamlit](https://streamlit.io/)
- **ML Model** — scikit-learn (RandomForestClassifier, StandardScaler)
- **Visualizations** — Plotly (bar, scatter, line with fill, sensitivity sweep)
- **Persistence** — joblib + JSON
- **UI Design** — Custom glassmorphism CSS (Manrope, Sora, JetBrains Mono fonts)

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd forestiq

# 2. Install dependencies
pip install streamlit numpy plotly scikit-learn joblib

# 3. Launch the app
streamlit run app3.py
```

---

## 🎨 UI Design

ForestIQ uses the Elevvo glassmorphism design system with 7 unique cover-type accent colors:
- Each forest type has its own color, dim background, and border for visual identity
- Sticky glassmorphism navigation bar
- Sensitivity sweep charts with spline smoothing and area fill
- Dark ambient background with violet/pink radial gradients

---

## 🔗 Links

| Resource | URL |
|---|---|
| 🚀 Live Streamlit App | [elevvo-forest-cover.streamlit.app](https://elevvo-forest-cover.streamlit.app/) |
| 📓 Kaggle Notebook | [kaggle.com/code/abdel2ty/elevvo-forest-cover](https://www.kaggle.com/code/abdel2ty/elevvo-forest-cover) |

---

## 👤 Author

**@abdel2ty** — Built as part of the Elevvo ML project series.
