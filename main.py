import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import joblib
import plotly.graph_objects as go

model = joblib.load("CatBoost_timerows.pkl")

st.title("Прогноз продаж на будущие даты")

data_train = pd.read_csv("data_train.csv", parse_dates=["Order Date"], index_col="Order Date")
data_test  = pd.read_csv("data_test.csv", parse_dates=["Order Date"], index_col="Order Date")

start_date = st.date_input("Начальная дата прогноза", pd.to_datetime("2018-01-01"))
end_date = st.date_input("Конечная дата прогноза", pd.to_datetime("2018-12-31"))

if start_date > end_date:
    st.error("Начальная дата не может быть больше конечной")
else:
    if st.button("Сделать прогноз"):
        # Предсказания на исторических данных
        y_pred_train = model.predict(data_train[["year","month","week","dayofweek"]])
        y_pred_test  = model.predict(data_test[["year","month","week","dayofweek"]])

        # Генерация будущих недель
        future_dates = pd.date_range(start=start_date, end=end_date, freq='W')
        future_df = pd.DataFrame(index=future_dates)
        future_df["year"] = future_df.index.year
        future_df["month"] = future_df.index.month
        future_df["week"] = future_df.index.isocalendar().week.astype(int)
        future_df["dayofweek"] = future_df.index.dayofweek
        future_df["y_pred"] = model.predict(future_df[["year","month","week","dayofweek"]])

        # Интерактивный график с Plotly
        fig = go.Figure()

        # Train
        fig.add_trace(go.Scatter(
            x=data_train.index, y=data_train['y'],
            mode='lines', name='Данные до 2017', line=dict(color='#0099CC')
        ))
        fig.add_trace(go.Scatter(
            x=data_train.index, y=y_pred_train,
            mode='lines', name='Прогноз до 2017', line=dict(color='red', dash='dash')
        ))

        # Test
        fig.add_trace(go.Scatter(
            x=data_test.index, y=data_test['y'],
            mode='lines', name='Данные до 2018', line=dict(color='#339999')
        ))
        fig.add_trace(go.Scatter(
            x=data_test.index, y=y_pred_test,
            mode='lines', name='Прогноз до 2018', line=dict(color='red', dash='dash')
        ))

        # Future
        fig.add_trace(go.Scatter(
            x=future_df.index, y=future_df["y_pred"],
            mode='lines', name='Прогноз', line=dict(color='purple', dash='dot')
        ))

        fig.update_layout(
            title="Прогноз продаж с историческими данными",
            xaxis_title="Дата",
            yaxis_title="Продажи",
            hovermode="x unified",
            width=1200,   # увеличиваем ширину
            height=700,   # увеличиваем высоту
            xaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Скачать CSV будущего прогноза
        csv_bytes = future_df[["y_pred"]].reset_index().rename(columns={"index":"ds"}).to_csv(index=False).encode()
        st.download_button(
            label="Скачать прогноз в CSV",
            data=csv_bytes,
            file_name="future_sales_forecast.csv",
            mime="text/csv"
        )