# 🏠 House Price Prediction — Machine Learning Project

> Predict house prices using real-world features like size, location, and condition.  
> Built with Python · Pandas · Scikit-learn · Matplotlib · Seaborn

---

## What Does This Project Do?

Ever wondered how a real estate website instantly suggests a price for a house? This project does exactly that — it trains a machine learning model on historical house data and learns to predict the price of any house based on its features.

Given inputs like square footage, number of bedrooms, neighborhood, and house condition, the model outputs an estimated price in dollars. It's a classic **regression problem** — we're not classifying things into categories, we're predicting a continuous number.

---

## The Dataset

The dataset (`house_prices.csv`) contains **1,000 house records**, each with the following information:

| Feature | What It Means |
|---|---|
| `Sqft` | Total area of the house in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `House_Age` | Age of the house in years |
| `Garage_Cars` | How many cars the garage fits |
| `Has_Pool` | Whether the house has a pool (1 = Yes, 0 = No) |
| `Floors` | Number of floors |
| `Neighborhood` | Area type: Uptown, Downtown, Midtown, Suburbs, or Rural |
| `Condition` | Overall condition: Excellent, Good, Fair, or Poor |
| `Price` | 🎯 **Target** — what we want to predict |

---

## Project Structure

```
house_price_prediction/
│
├── house_price_prediction.py   ← Main script (all 8 phases)
├── house_prices.csv            ← Dataset
│
├── eda_1_target_corr.png       ← Price distribution & correlation heatmap
├── eda_2_boxplots.png          ← Price by Neighborhood & Condition
├── eda_3_histograms.png        ← Distribution of each feature
├── eda_4_missing.png           ← Missing values chart
├── eval_actual_vs_predicted.png← How well the best model predicted
├── feature_importance.png      ← Which features matter most
└── comparison_chart.png        ← All 5 models compared side by side
```

---

## How to Run It

### Step 1 — Make sure you have Python installed
Python 3.8 or higher is recommended.

### Step 2 — Install the required libraries
Open your terminal and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3 — Run the script
Make sure `house_prices.csv` and `house_price_prediction.py` are in the same folder, then:

```bash
python house_price_prediction.py
```

That's it! The script will print results phase by phase and save all charts as PNG files in the same folder.

---

## What Happens Inside — The 8 Phases

The script is organized into 8 clean phases. Here's what each one does in plain English:

### Phase 1 — Understanding the Data
Before touching any code, we get to know the dataset. How many rows? Any duplicates? Which columns have missing values? This is like reading the manual before driving a new car.

### Phase 2 — Exploratory Data Analysis (EDA)
We create 4 charts to visually understand the data:
- **Price Distribution** — Most houses fall between $200K–$500K; the distribution is right-skewed (a few very expensive homes pull the average up)
- **Price vs Square Footage** — Clear upward trend: bigger house → higher price
- **Price by Neighborhood** — Uptown and Downtown command the highest prices; Rural is the most affordable
- **Price by Condition** — Excellent condition homes are noticeably pricier than Poor ones
- **Missing Values** — About 8% of data is missing in Sqft, Bedrooms, Bathrooms, and Garage_Cars

### Phase 3 — Data Preprocessing (Cleaning & Transforming)
Raw data is messy. This phase fixes it:
- **Removes duplicates** — 10 duplicate rows were found and dropped
- **Fills missing values** — Missing numbers are filled with the median (e.g., if most houses have 3 bedrooms, missing values get 3)
- **Encodes categories** — Machine learning models only understand numbers, so we convert text to numbers:
  - `Condition`: Poor=0, Fair=1, Good=2, Excellent=3 (ordered encoding)
  - `Neighborhood`: Split into separate yes/no columns (one-hot encoding)

### Phase 4 — Splitting & Scaling
- **80/20 split** — 800 rows used for training, 200 held back for testing. The model never sees the test set during training — this is how we honestly measure performance.
- **StandardScaler** — Normalizes all features to a similar range. This is especially important for models like SVR and KNN that are sensitive to feature scale.

### Phase 5 — Training 5 Models
Five different algorithms are trained on the same data so we can compare them fairly:

| Model | What It Does (Simply Put) |
|---|---|
| **Linear Regression** | Draws a straight line through the data |
| **Decision Tree** | Asks a series of yes/no questions to arrive at a price |
| **Random Forest** | Combines hundreds of decision trees and averages their answers |
| **Support Vector (SVR)** | Finds the best-fit curve within a margin of error |
| **K-Nearest Neighbors** | Looks at the K most similar houses and averages their prices |

### Phase 6 — Evaluating the Models
Each model is tested on the 200 unseen rows. Three metrics are used:

- **MAE (Mean Absolute Error)** — On average, how many dollars off is the prediction?
- **RMSE (Root Mean Squared Error)** — Similar to MAE but penalizes large errors more
- **R² Score** — What percentage of the price variation does the model explain? (1.0 = perfect, 0 = no better than guessing the average)

We also plot **Actual vs Predicted** prices for the best model — if predictions were perfect, all dots would lie on the diagonal red line.

### Phase 7 — Comparing All Models
A side-by-side bar chart shows R² scores and RMSE for all 5 models.

**Results:**

| Rank | Model | R² Score | RMSE |
|---|---|---|---|
| 🥇 1 | **Random Forest** | **0.917** | ~$54,000 |
| 🥈 2 | Linear Regression | 0.913 | ~$57,000 |
| 🥉 3 | Decision Tree | 0.770 | ~$93,000 |
| 4 | K-Nearest Neighbors | 0.754 | ~$95,000 |
| 5 | Support Vector (SVR) | 0.001 | ~$195,000 |

**Random Forest wins** — it explains 91.7% of price variation with an average prediction error of around $54,000.

### Phase 8 — Feature Importance
Which features actually matter? The Random Forest model can tell us exactly how much each input contributed to its predictions:

- 🟢 **Square Footage (Sqft)** — By far the most important (~68% weight). Bigger house = higher price, and the model learned this strongly.
- 🟢 **Neighborhood_Rural & Neighborhood_Suburbs** — Location is the second biggest factor. Rural and suburban areas have noticeably lower prices.
- **House_Age** and **Condition** — Meaningful but secondary factors.
- **Has_Pool** — Surprisingly low impact on its own.

---

## Key Takeaways

- **Square footage is king.** It alone drives nearly 70% of the model's predictions.
- **Location matters a lot.** Neighborhood explains more of the price difference than condition or age.
- **Random Forest outperforms all others** — because house prices involve complex, non-linear interactions between features that a single decision tree or straight line can't fully capture.
- **SVR performed poorly** — it's very sensitive to hyperparameter tuning and doesn't scale well without careful optimization.

---

## Ideas for Making It Even Better

This project is beginner-friendly, but here are ways to push it further:

1. **Tune hyperparameters** — Use `GridSearchCV` to find the optimal settings for Random Forest
2. **Try XGBoost or LightGBM** — Gradient boosting models often outperform Random Forest
3. **Cross-validation** — Use 5-fold CV instead of a single train-test split for more reliable metrics
4. **Feature engineering** — Create new features like `price_per_sqft` or `sqft × neighborhood` interaction terms
5. **SHAP values** — Explain why the model predicted a specific price for a specific house
6. **Streamlit web app** — Build a simple UI where users type in house details and get an instant price estimate
7. **FastAPI deployment** — Turn the model into a REST API
8. **MLflow tracking** — Log experiments and compare runs systematically

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and chart generation |
| `seaborn` | Beautiful statistical visualizations |
| `scikit-learn` | All ML models, preprocessing, and metrics |

---

## About This Project

This is an end-to-end beginner-to-intermediate level machine learning project designed to be **portfolio-ready**. It covers the full data science workflow — from raw data to a trained, evaluated, and interpreted model — with clean code, meaningful comments, and professional-quality charts.

---

## Author

**Sujay A**  
GitHub: [@sujayanandhan2007](https://github.com/sujayanandhan2007)

---

*Built with Python 3 · scikit-learn · Matplotlib · Seaborn*
