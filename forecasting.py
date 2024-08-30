import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 抽象クラスの定義
class ForecastingModel(ABC):
    def __init__(self, df, test_size=0.25, future_steps=30):
        self.df = df
        self.test_size = test_size
        self.future_steps = future_steps
        self._prepare_data()

    def _prepare_data(self):
        self.df['diff'] = self.df['y'].diff().fillna(0)
        self.df['day'] = np.arange(len(self.df))
        
        X = self.df[['day']]
        y = self.df['diff']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)

    @abstractmethod
    def forecast(self):
        pass

    def calculate_errors(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return mae, rmse

# 線形回帰モデル
class LinearRegressionModel(ForecastingModel):
    def forecast(self):
        from sklearn.linear_model import LinearRegression

        model_lr = LinearRegression()
        model_lr.fit(self.X_train, self.y_train)
        
        # 予測値を累積して元のスケールに戻す
        y_pred_diff = model_lr.predict(self.X_test)
        y_pred = np.cumsum(y_pred_diff) + self.df['y'].iloc[len(self.X_train)-1]
        
        # 将来の予測
        future_preds = []
        last_value = self.df['y'].iloc[-1]
        
        for i in range(self.future_steps):
            next_day = pd.DataFrame([[len(self.df) + i]], columns=['day'])
            next_diff = model_lr.predict(next_day)
            last_value += next_diff[0]
            future_preds.append(last_value)
        
        future_dates = pd.date_range(start=self.df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=self.future_steps)
        
        return y_pred[-len(self.y_test):], future_dates, future_preds

# ARIMAモデル
class ARIMAModel(ForecastingModel):
    def forecast(self):
        from statsmodels.tsa.arima.model import ARIMA

        model_arima = ARIMA(self.df['y'], order=(5,1,0))
        arima_result = model_arima.fit()
        
        forecast_arima = arima_result.forecast(steps=len(self.y_test))
        future_forecast_arima = arima_result.forecast(steps=self.future_steps)
        future_dates = pd.date_range(start=self.df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=self.future_steps)
        
        return forecast_arima, future_dates, future_forecast_arima

# SARIMAモデル
class SARIMAModel(ForecastingModel):
    def forecast(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sarima_model = SARIMAX(self.df['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
        sarima_result = sarima_model.fit()
        
        forecast_sarima = sarima_result.get_forecast(steps=len(self.y_test)).predicted_mean
        future_forecast_sarima = sarima_result.get_forecast(steps=self.future_steps).predicted_mean
        future_dates = pd.date_range(start=self.df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=self.future_steps)
        
        return forecast_sarima, future_dates, future_forecast_sarima

# Prophetモデル
class ProphetModel(ForecastingModel):
    def forecast(self):
        from prophet import Prophet

        model = Prophet()
        model.fit(self.df)
        
        future = model.make_future_dataframe(periods=self.future_steps)
        forecast = model.predict(future)
        
        forecast_prophet = forecast['yhat'].iloc[-len(self.y_test):]
        future_forecast_prophet = forecast['yhat'].iloc[-self.future_steps:]
        
        return forecast_prophet, forecast['ds'].iloc[-self.future_steps:], future_forecast_prophet
