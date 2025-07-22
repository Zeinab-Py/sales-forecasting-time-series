# 📊 sales-forecasting-time-series
This project is an interactive web application built with Streamlit and Prophet for time series forecasting. It allows users to upload historical sales data and forecast future sales for the upcoming quarter.
# 🚀 Features
* Upload your own sales CSV file for forecasting
* Filter by Store and Item or forecast for all
* Add optional external regressors (e.g., holidays, marketing)
* Adjustable changepoint prior scale and seasonality settings
* View forecast results as interactive charts and tables
* Built-in support for U.S. holidays
* Custom image and clean interface for user-friendly experience
### 📁 Project Structure

<details>
<summary>Click to expand</summary>

```text
sales-forecasting-time-series/
├── app/
│   ├── NextQ_Forecast.py         # Streamlit app
│   └── assets/
│       └── Designer.jpeg         # App cover image
├── data/
│   ├── sales_example.csv         # Example sales data
│   └── US Bank holidays.csv      # Holiday data used for regressors
├── notebooks/
│   └── Data_analysing.ipynb      # ARIMA, SARIMA, Prophet analysis
├── .gitignore
├── requirements.txt
└── README.md
```
</details>

### ⚙️ Installation
To run the app locally:
```
git clone https://github.com/yourusername/sales-forecasting-time-series.git
cd sales-forecasting-time-series
pip install -r requirements.txt
streamlit run app/NextQ_Forecast.py
```
⚠️ Note: Ensure the image and CSV file paths are correct relative to the NextQ_Forecast.py script.



