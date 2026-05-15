# 🛒 E-Commerce Sales Data Analysis
![Dashboard](dashboard_summary.png)

---

## About This Project

A complete data analysis project on an e-commerce sales dataset. The goal was to clean messy real-world data, explore patterns through statistics, and communicate findings through clear visualizations.

This project was built as part of my data analytics learning journey and is suitable for internship submissions and portfolio showcasing.

---

## What I Did

**Data Cleaning**
- Removed 80+ duplicate records
- Fixed inconsistent text formatting across category columns
- Handled missing values using median and mode imputation
- Corrected invalid negative quantity entries
- Detected and capped outliers using the IQR method
- Renamed columns to snake_case and created new time-based features

**Exploratory Data Analysis**
- Analyzed revenue trends across months, weekdays, and quarters
- Compared performance across product categories and countries
- Identified the highest-spending customer age group
- Detected cancellation and refund patterns affecting revenue

**Visualizations (10 Charts)**
- Revenue by category — horizontal bar chart
- Monthly revenue trend — line chart with rolling average
- Revenue distribution — histogram with KDE overlay
- Price vs revenue — scatter plot by category
- Revenue spread — box plot per category
- Feature relationships — correlation heatmap
- Payment method share — donut and bar chart
- Revenue by age group and order status — grouped bar
- Weekday × category revenue — heatmap
- Customer rating distribution — violin plot

---

## Key Findings

- Electronics and Home & Garden drive the highest revenue
- Customers aged 26–35 have the highest average spend
- Credit Card is the most used payment method at 30%+
- A 12% cancellation rate is causing significant revenue leakage
- 65% of orders are rated 4 or 5 stars
- Revenue consistently peaks on Wednesday and Thursday

---

## Project Files

| File | Description |
|------|-------------|
| `ecommerce_data_project.ipynb` | Main notebook with full analysis |
| `ecommerce_raw.csv` | Original unprocessed dataset |
| `ecommerce_cleaned.csv` | Final cleaned dataset |
| `dashboard_summary.png` | 6-panel summary dashboard |
| `plot_01` to `plot_10` | Individual chart exports |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook ecommerce_data_project.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com) — File → Upload Notebook → Runtime → Run All

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook

---

## Author

**Sujay A**
📧 sujayanandhan2007@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/sujay-anandhan-b338bb381) · [GitHub](https://github.com/sujayanandhan2007)
