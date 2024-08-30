import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 線形回帰で1階差を予測
def linear_regression_forecast(df, X_train, X_test, y_train, y_test, future_steps):
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # 予測値を累積して元のスケールに戻す
    y_pred_diff = model_lr.predict(X_test)
    y_pred = np.cumsum(y_pred_diff) + df['y'].iloc[len(X_train)-1]
    
    # 将来の予測
    future_preds = []
    last_value = df['y'].iloc[-1]
    
    for i in range(future_steps):
        next_day = pd.DataFrame([[len(df) + i]], columns=['day'])
        next_diff = model_lr.predict(next_day)
        last_value += next_diff[0]
        future_preds.append(last_value)
    
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
    
    return y_pred[-len(y_test):], future_dates, future_preds

# ARIMAでの予測
def arima_forecast(df, y_test, future_steps):
    model_arima = ARIMA(df['y'], order=(5,1,0))
    arima_result = model_arima.fit()
    
    forecast_arima = arima_result.forecast(steps=len(y_test))
    future_forecast_arima = arima_result.forecast(steps=future_steps)
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
    
    return forecast_arima, future_dates, future_forecast_arima

# SARIMAでの予測
def sarima_forecast(df, y_test, future_steps):
    sarima_model = SARIMAX(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_result = sarima_model.fit()
    
    forecast_sarima = sarima_result.get_forecast(steps=len(y_test)).predicted_mean
    future_forecast_sarima = sarima_result.get_forecast(steps=future_steps).predicted_mean
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
    
    return forecast_sarima, future_dates, future_forecast_sarima

# Prophetでの予測
def prophet_forecast(df, y_test, future_steps):
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=future_steps)
    forecast = model.predict(future)
    
    forecast_prophet = forecast['yhat'].iloc[-len(y_test):]
    future_forecast_prophet = forecast['yhat'].iloc[-future_steps:]
    
    return forecast_prophet, forecast['ds'].iloc[-future_steps:], future_forecast_prophet

# 誤差の計算
def calculate_errors(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse
