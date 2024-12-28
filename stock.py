import yfinance as yf
import pandas as pd
from datetime import date
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    stock_data = stock_data[['Close']]  
    stock_data['Prediction'] = stock_data['Close'].shift(-1)
    stock_data = stock_data[:-1]

    X = stock_data[['Close']].values
    y = stock_data['Prediction'].values
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae

def predict_stock_price(model, stock_data):
    last_close = stock_data['Close'].iloc[-1]
    predicted_price = model.predict([[last_close]])  
    return predicted_price[0]

company_to_ticker = {
    'Reliance Industries': 'RELIANCE.NS', 'Tata Consultancy Services': 'TCS.NS', 'Infosys': 'INFY.NS',
    'HDFC Bank': 'HDFCBANK.NS', 'ICICI Bank': 'ICICIBANK.NS', 'State Bank of India': 'SBIN.NS',
    'Bajaj Finance': 'BAJFINANCE.NS', 'HDFC': 'HDFC.NS', 'Larsen & Toubro': 'LARSEN.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS', 'Hindustan Unilever': 'HINDUNILVR.NS', 'Maruti Suzuki': 'MARUTI.NS',
    'ITC': 'ITC.NS', 'Mahindra & Mahindra': 'M&M.NS', 'Axis Bank': 'AXISBANK.NS', 'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'HCL Technologies': 'HCLTECH.NS', 'Sun Pharmaceutical': 'SUNPHARMA.NS', 'Wipro': 'WIPRO.NS',
    'Bharti Airtel': 'BHARTIARTL.NS', 'Tech Mahindra': 'TECHM.NS', 'UltraTech Cement': 'ULTRACEMCO.NS',
    'NTPC': 'NTPC.NS', 'Oil and Natural Gas Corporation': 'ONGC.NS', 'Adani Green Energy': 'ADANIGREEN.NS',
    'Adani Ports': 'ADANIPORTS.NS', 'Cipla': 'CIPLA.NS', 'Titan': 'TITAN.NS', 'Tata Steel': 'TATASTEEL.NS',
    'Apollo Hospitals': 'APOLLOHOSP.NS', 'Muthoot Finance': 'MUTHOOTFIN.NS', 'Divi’s Laboratories': 'DIVISLAB.NS',
    'Bharat Petroleum': 'BPCL.NS', 'Indian Oil Corporation': 'IOC.NS', 'Power Grid': 'POWERGRID.NS',
    'GAIL': 'GAIL.NS', 'IndusInd Bank': 'INDUSINDBK.NS', 'Shree Cement': 'SHREECEM.NS', 'Indigo': 'INDIGO.NS',
    'Marico': 'MARICO.NS', 'Reliance Infra': 'RELIANCEINFRA.NS', 'Eicher Motors': 'EICHERMOT.NS', 
    'Adani Gas': 'ADANIGAS.NS', 'HDFC Life': 'HDFCLIFE.NS', 'ICICI Lombard': 'ICICIGI.NS', 'Godrej Consumer': 'GODREJCP.NS',
}

EXCHANGE_RATE = 1

def calculate_rsi(stock_data, window=14):
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(stock_data, short_window=12, long_window=26, signal_window=9):
    ema_short = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = stock_data['Close'].ewm(span=long_window, adjust=False).mean()
    
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    
    return macd, signal

def technical_indicators(stock_data):
    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data['MACD'], stock_data['MACD_Signal'] = calculate_macd(stock_data)
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    return stock_data

def main():
    st.title("Stock Price Prediction")

    menu = st.sidebar.selectbox("Select Section", ["Information", "Graphs", "Data", "Technical Indicators", "Forecast", "Help"])
    
    company_name = st.sidebar.selectbox("Select Company", list(company_to_ticker.keys()))
    ticker = company_to_ticker[company_name]

    start_date = st.sidebar.date_input("Start Date:", pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End Date:", pd.to_datetime( date.today()))

    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data['Close_INR'] = stock_data['Close'] * EXCHANGE_RATE

    if menu == "Information":
        st.subheader(f"Information about {company_name} ({ticker})")
        st.write(f"Company: {company_name}")
        T = yf.Ticker(ticker)

        business_summary = T.info.get('longBusinessSummary')
        if business_summary:
                st.subheader("Business Summary")
                st.caption(business_summary)
                st.write("---")
        st.write(f"Ticker: {ticker}")
        st.write(f"Start Date: {start_date}")
        st.write(f"End Date: {end_date}")

    elif menu == "Graphs":
        st.subheader(f"Stock Price Visualization for {company_name} ({ticker})")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index, stock_data['Close_INR'], label='Actual Closing Price (INR)', color='blue')
        ax.set_title(f"{company_name} Stock Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        st.pyplot(fig)

    elif menu == "Data":
        st.subheader(f"Stock Data for {company_name} ({ticker})")
        st.write(stock_data.tail())

    elif menu == "Technical Indicators":
        st.subheader(f"Technical Indicators for {company_name} ({ticker})")
        
        stock_data = technical_indicators(stock_data)
        
        st.write("Technical Indicators (RSI, MACD, SMA 20, SMA 50):")
        st.write(stock_data[['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50']].tail())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index, stock_data['Close_INR'], label='Close Price', color='blue')
        ax.plot(stock_data.index, stock_data['SMA_20'], label='SMA 20', color='red', linestyle='--')
        ax.plot(stock_data.index, stock_data['SMA_50'], label='SMA 50', color='green', linestyle='--')
        ax.set_title(f"{company_name} Stock Price and Technical Indicators")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        st.pyplot(fig)

        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
        ax_rsi.plot(stock_data.index, stock_data['RSI'], label='RSI', color='purple')
        ax_rsi.axhline(70, color='red', linestyle='--')
        ax_rsi.axhline(30, color='green', linestyle='--')
        ax_rsi.set_title(f"RSI for {company_name} ({ticker})")
        ax_rsi.set_xlabel("Date")
        ax_rsi.set_ylabel("RSI")
        ax_rsi.legend()
        st.pyplot(fig_rsi)
        
        fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
        ax_macd.plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue')
        ax_macd.plot(stock_data.index, stock_data['MACD_Signal'], label='Signal Line', color='orange')
        ax_macd.set_title(f"MACD for {company_name} ({ticker})")
        ax_macd.set_xlabel("Date")
        ax_macd.set_ylabel("MACD")
        ax_macd.legend()
        st.pyplot(fig_macd)

    elif menu == "Forecast":
        st.subheader(f"Stock Price Forecast for {company_name} ({ticker})")
        if st.button("Predict"):
            try:
                X, y = preprocess_data(stock_data)
                model, mae = train_model(X, y)
                predicted_price = predict_stock_price(model, stock_data) * EXCHANGE_RATE

                st.subheader(f"Predicted Closing Price for {company_name} ({ticker}): ₹{predicted_price:.2f}")
                st.write(f"Mean Absolute Error: ₹{mae * EXCHANGE_RATE:.2f}")

                st.subheader(f"Stock Price Visualization for {company_name} ({ticker})")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(stock_data.index, stock_data['Close_INR'], label='Actual Closing Price (INR)', color='blue')
                ax.scatter(stock_data.index[-1], predicted_price, color='red', label=f'Predicted Price (₹{predicted_price:.2f})', zorder=5)
                ax.set_title(f"{company_name} Stock Price and Predicted Value")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (INR)")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif menu == "Help":
        st.subheader("Help Section")
        st.write("""
            ### How to Use:
            - **Information**: View general information about the selected company.
            - **Graphs**: View the historical stock price chart.
            - **Data**: View the latest stock data for the selected company.
            - **Technical Indicators**: View common technical indicators like SMA, RSI, MACD.
            - **Forecast**: View the stock price forecast using machine learning models.
            - **Help**: Get help on using this application.
        """)

if __name__ == "__main__":
    main()
