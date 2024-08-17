import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import plotly.express as px
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import requests
from bs4 import BeautifulSoup
import base64
import warnings
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
warnings.filterwarnings('ignore')

# List of companies (display name, ticker symbol)
COMPANIES = [
        ("3M", "MMM"), ("Abbott Laboratories", "ABT"), ("AbbVie", "ABBV"), ("Accenture", "ACN"), ("Adobe", "ADBE"),
        ("Advanced Micro Devices", "AMD"), ("Aflac", "AFL"), ("Agilent Technologies", "A"), ("Air Products and Chemicals", "APD"),
        ("Alcoa", "AA"), ("Allstate", "ALL"), ("Alphabet (Google)", "GOOGL"), ("Altria Group", "MO"),
        ("Amazon", "AMZN"), ("American Express", "AXP"), ("American International Group", "AIG"), ("Amgen", "AMGN"),
        ("Analog Devices", "ADI"), ("Apple", "AAPL"), ("Applied Materials", "AMAT"), ("Archer-Daniels-Midland", "ADM"),
        ("AT&T", "T"), ("Autodesk", "ADSK"), ("AutoZone", "AZO"), ("Bank of America", "BAC"), ("Bank of New York Mellon", "BK"),
        ("Baxter International", "BAX"), ("BB&T Corporation", "BBT"), ("Best Buy", "BBY"), ("Biogen", "BIIB"), ("BlackRock", "BLK"),
        ("Boeing", "BA"), ("Bristol-Myers Squibb", "BMY"), ("Broadcom", "AVGO"), ("Capital One Financial", "COF"),
        ("Caterpillar", "CAT"), ("Celgene", "CELG"), ("CenterPoint Energy", "CNP"), ("Chevron", "CVX"),
        ("Chipotle Mexican Grill", "CMG"), ("Chubb", "CB"), ("Cigna", "CI"), ("Cisco Systems", "CSCO"),
        ("Citigroup", "C"), ("Coca-Cola", "KO"), ("Colgate-Palmolive", "CL"), ("Comcast", "CMCSA"), ("ConocoPhillips", "COP"),
        ("Costco Wholesale", "COST"), ("CVS Health", "CVS"), ("Danaher", "DHR"), ("Deere & Company", "DE"),
        ("Dell Technologies", "DELL"), ("Delta Air Lines", "DAL"), ("Disney", "DIS"), ("Dollar General", "DG"),
        ("Dominion Energy", "D"), ("Dow", "DOW"), ("Duke Energy", "DUK"), ("DuPont", "DD"), ("Eaton", "ETN"),
        ("eBay", "EBAY"), ("Eli Lilly", "LLY"), ("Emerson Electric", "EMR"), ("EOG Resources", "EOG"), ("Equinix", "EQIX"),
        ("Exelon", "EXC"), ("Exxon Mobil", "XOM"), ("FedEx", "FDX"), ("Fidelity National Information Services", "FIS"),
        ("Ford Motor", "F"), ("General Dynamics", "GD"), ("General Electric", "GE"), ("General Mills", "GIS"),
        ("General Motors", "GM"), ("Gilead Sciences", "GILD"), ("Goldman Sachs", "GS"), ("Halliburton", "HAL"),
        ("Hewlett Packard Enterprise", "HPE"), ("Home Depot", "HD"), ("Honeywell International", "HON"), ("HP", "HPQ"),
        ("Humana", "HUM"), ("IBM", "IBM"), ("Illinois Tool Works", "ITW"), ("Intel", "INTC"), ("International Paper", "IP"),
        ("Intuit", "INTU"), ("Johnson & Johnson", "JNJ"), ("JPMorgan Chase", "JPM"), ("Kellogg", "K"), ("Kimberly-Clark", "KMB"),
        ("Kohl's", "KSS"), ("Kraft Heinz", "KHC"), ("Kroger", "KR"), ("Lockheed Martin", "LMT"), ("Lowe's", "LOW"),
        ("Macy's", "M"), ("Marathon Oil", "MRO"), ("Marriott International", "MAR"), ("Mastercard", "MA"), ("McDonald's", "MCD"),
        ("Medtronic", "MDT"), ("Merck", "MRK"), ("Meta Platforms (Facebook)", "META"), ("MetLife", "MET"), ("Microsoft", "MSFT"),
        ("Mondelez International", "MDLZ"), ("Morgan Stanley", "MS"), ("Motorola Solutions", "MSI"), ("Netflix", "NFLX"),
        ("Nike", "NKE"), ("Northrop Grumman", "NOC"), ("NVIDIA", "NVDA"), ("Oracle", "ORCL"), ("PepsiCo", "PEP"),
        ("Pfizer", "PFE"), ("Philip Morris International", "PM"), ("Phillips 66", "PSX"), ("PNC Financial Services", "PNC"),
        ("PPG Industries", "PPG"), ("Procter & Gamble", "PG"), ("Qualcomm", "QCOM"), ("Raytheon Technologies", "RTX"),
        ("Rockwell Automation", "ROK"), ("Ross Stores", "ROST"), ("Schlumberger", "SLB"), ("Starbucks", "SBUX"),
        ("State Street", "STT"), ("Sysco", "SYY"), ("Target", "TGT"), ("Texas Instruments", "TXN"), ("Thermo Fisher Scientific", "TMO"),
        ("TJX Companies", "TJX"), ("T-Mobile US", "TMUS"), ("Union Pacific", "UNP"), ("United Airlines Holdings", "UAL"),
        ("United Parcel Service", "UPS"), ("United Technologies", "UTX"), ("UnitedHealth Group", "UNH"), ("US Bancorp", "USB"),
        ("Valero Energy", "VLO"), ("Verizon Communications", "VZ"), ("Visa", "V"), ("Walgreens Boots Alliance", "WBA"), ("Walmart", "WMT"), ("Walt Disney", "DIS"),
        ("Wells Fargo", "WFC"), ("Western Digital", "WDC"), ("Weyerhaeuser", "WY"), ("Whirlpool", "WHR"), ("Williams Companies", "WMB"), ("Xcel Energy", "XEL"),
        ("Xilinx", "XLNX"), ("Yum! Brands", "YUM"), ("Zimmer Biomet", "ZBH") 
            ]

class StockPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None

    def preprocess_data(self):
        self.data = self.data.reset_index()
        self.data = self.data.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Add technical indicators
        self.data['SMA_20'] = SMAIndicator(close=self.data['y'], window=20).sma_indicator()
        self.data['EMA_20'] = EMAIndicator(close=self.data['y'], window=20).ema_indicator()
        self.data['RSI'] = RSIIndicator(close=self.data['y'], window=14).rsi()
        bb = BollingerBands(close=self.data['y'], window=20, window_dev=2)
        self.data['BB_high'] = bb.bollinger_hband()
        self.data['BB_low'] = bb.bollinger_lband()
        
        # Add lagged features
        self.data['lag_1'] = self.data['y'].shift(1)
        self.data['lag_7'] = self.data['y'].shift(7)
        
        # Add rolling statistics
        self.data['rolling_mean_7'] = self.data['y'].rolling(window=7).mean()
        self.data['rolling_std_7'] = self.data['y'].rolling(window=7).std()
        
        # Handle NaN values
        self.data = self.data.dropna()

    def train_model(self):
        try:
            self.model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Add additional regressors
            for column in ['SMA_20', 'EMA_20', 'RSI', 'BB_high', 'BB_low', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']:
                self.model.add_regressor(column)
            
            self.model.fit(self.data)
            return True
        except Exception as e:
            print(f"Error training Prophet model: {str(e)}")
            return False

    def predict(self, days=30):
        try:
            future = self.model.make_future_dataframe(periods=days)
            
            # Add regressor values for future dates
            for column in ['SMA_20', 'EMA_20', 'RSI', 'BB_high', 'BB_low', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']:
                future[column] = self.data[column].iloc[-1]  # Use last known value
            
            forecast = self.model.predict(future)
            
            # Calculate components
            forecast['trend'] = forecast['trend']
            forecast['yearly'] = forecast['yearly'] if 'yearly' in forecast.columns else 0
            forecast['weekly'] = forecast['weekly'] if 'weekly' in forecast.columns else 0
            forecast['daily'] = forecast['daily'] if 'daily' in forecast.columns else 0
            
            return forecast
        except Exception as e:
            print(f"Error predicting with Prophet model: {str(e)}")
            return None

    def evaluate_model(self, test_data):
        predictions = self.predict(days=len(test_data))
        
        if predictions is None:
            return None, None, None

        actual = test_data['Close'].values
        predicted = predictions['yhat'].values[-len(test_data):]

        mse = mean_squared_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)
        rmse = np.sqrt(mse)

        return mse, mape, rmse

    def cross_validate_model(self, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []

        for train_index, test_index in tscv.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            # Train the model
            model = Prophet()
            model.fit(train_data)

            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)

            # Calculate metrics
            y_true = test_data['y'].values
            y_pred = forecast['yhat'].tail(len(test_data)).values

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            cv_results.append({
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            })

        return pd.DataFrame(cv_results)

def fetch_stock_data(ticker):
    try:
        end_date = datetime.now()
        start_date = datetime(2000, 1, 1) 
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def create_test_plot(train_data, test_data, predicted_data, company_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_data['ds'],
        y=train_data['y'],
        mode='lines',
        name='Training Data',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=test_data['y'],
        mode='lines',
        name='Actual (Test) Data',
        line=dict(color='green')
    ))

    if predicted_data is not None:
        fig.add_trace(go.Scatter(
            x=test_data['ds'],  # Align predicted data with test data
            y=predicted_data['yhat'][-len(test_data):],
            mode='lines',
            name='Predicted Data',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title=f'{company_name} Stock Price Prediction (Test Model)',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark',
        hovermode='x unified',
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_prediction_plot(data, predicted_data, company_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Historical Data',
        line=dict(color='cyan')
    ))

    if predicted_data is not None:
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(predicted_data))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_data['yhat'],
            mode='lines',
            name='Predicted Data',
            line=dict(color='yellow')
        ))
        
        # Add prediction intervals
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_data['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_data['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 255, 0, 0.3)',
            fill='tonexty',
            name='Prediction Interval'
        ))

    fig.update_layout(
        title=f'{company_name} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark',
        hovermode='x unified',
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_candlestick_plot(data, company_name):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(
        title=f'{company_name} Stock Price Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=True
    )
    return fig

def fetch_news(company_name):
    try:
        url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features='xml')
        news_items = soup.findAll('item')
        
        news = []
        for item in news_items[:5]:
            news.append({
                'title': item.title.text,
                'link': item.link.text,
                'pubDate': item.pubDate.text
            })
        
        return news
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def get_table_download_link(df, filename, text):
    """Generates a link to download the given dataframe as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Advanced Stock Price Predictor using Prophet")

    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Test Model", "Predict Stock Prices", "Explore Data"])

    if app_mode == "Test Model":
        test_model()
    elif app_mode == "Predict Stock Prices":
        predict_stock_prices()
    elif app_mode == "Explore Data":
        explore_data()

def test_model():
    st.header("Test Enhanced Prophet Model")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Select Company", [company for company, _ in COMPANIES])
        test_split = st.slider("Test Data Split", 0.1, 0.5, 0.2, 0.05)

    if st.button("Train and Test Model"):
        with st.spinner("Fetching data and training model..."):
            company_name, ticker = next((name, symbol) for name, symbol in COMPANIES if name == company)

            data = fetch_stock_data(ticker)

            if data is not None:
                st.subheader("Stock Data Information")
                st.write(data.info())
                st.write(data.describe())
                
                # Display interactive dataframe
                st.subheader("Stock Data Preview")
                st.dataframe(data.head(100), use_container_width=True)
                
                # Provide download link for full dataset
                st.markdown(get_table_download_link(data, f"{ticker}_stock_data.csv", "Download full stock data CSV"), unsafe_allow_html=True)

                split_index = int(len(data) * (1 - test_split))
                train_data = data.iloc[:split_index]
                test_data = data.iloc[split_index:]

                predictor = StockPredictor(train_data)
                predictor.preprocess_data()
                if predictor.train_model():
                    test_pred = predictor.predict(days=len(test_data))

                    if test_pred is not None:
                        mse, mape, rmse = predictor.evaluate_model(test_data)
                        
                        if mse is not None and mape is not None and rmse is not None:
                            accuracy = 100 - mape * 100

                            st.subheader("Model Performance")
                            st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                            st.metric("Root Mean Squared Error", f"{rmse:.4f}")

                            plot = create_test_plot(predictor.data, test_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}), test_pred, company_name)
                            st.plotly_chart(plot, use_container_width=True)

                            # Cross-validation results
                            st.subheader("Cross-Validation Results")
                            cv_results = predictor.cross_validate_model()
                            
                            # Display interactive dataframe
                            st.dataframe(cv_results, use_container_width=True)
                            
                            # Provide download link for cross-validation results
                            st.markdown(get_table_download_link(cv_results, f"{ticker}_cv_results.csv", "Download cross-validation results CSV"), unsafe_allow_html=True)

                            # Calculate and display average metrics
                            avg_mse = cv_results['mse'].mean()
                            avg_rmse = cv_results['rmse'].mean()
                            avg_mape = cv_results['mape'].mean()
                            
                            st.write(f"Average MSE: {avg_mse:.4f}")
                            st.write(f"Average RMSE: {avg_rmse:.4f}")
                            st.write(f"Average MAPE: {avg_mape:.4f}")

                            # Display predictions
                            st.subheader("Predictions")
                            predictions_df = pd.DataFrame({
                                'Date': test_pred['ds'],
                                'Predicted': test_pred['yhat'],
                                'Lower Bound': test_pred['yhat_lower'],
                                'Upper Bound': test_pred['yhat_upper']
                            })
                            st.dataframe(predictions_df, use_container_width=True)
                            
                            # Provide download link for predictions
                            st.markdown(get_table_download_link(predictions_df, f"{ticker}_predictions.csv", "Download predictions CSV"), unsafe_allow_html=True)

                        else:
                            st.error("Failed to evaluate the model. The evaluation metrics are None.")
                    else:
                        st.error("Failed to generate predictions. The predicted data is None.")
                else:
                    st.error("Failed to train the Prophet model. Please try a different dataset.")
            else:
                st.error("Failed to fetch stock data. Please try again.")

def predict_stock_prices():
    st.header("Predict Stock Prices with Enhanced Model")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Select Company", [company for company, _ in COMPANIES])
        days_to_predict = st.slider("Days to Predict", 1, 365, 30)

    if st.button("Predict Stock Prices"):
        with st.spinner("Fetching data and making predictions..."):
            company_name, ticker = next((name, symbol) for name, symbol in COMPANIES if name == company)

            data = fetch_stock_data(ticker)

            if data is not None:
                st.subheader("Stock Data Information")
                st.write(data.info())
                st.write(data.describe())
                st.dataframe(data.head())

                predictor = StockPredictor(data)
                predictor.preprocess_data()
                if predictor.train_model():
                    predictions = predictor.predict(days=days_to_predict)

                    if predictions is not None:
                        # Create prediction plot
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='cyan')
                        ))

                        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions['yhat'].tail(days_to_predict),
                            mode='lines',
                            name='Predicted Data',
                            line=dict(color='yellow')
                        ))

                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions['yhat_upper'].tail(days_to_predict),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions['yhat_lower'].tail(days_to_predict),
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(255, 255, 0, 0.3)',
                            fill='tonexty',
                            name='Prediction Interval'
                        ))

                        fig.update_layout(
                            title=f'{company_name} Stock Price Prediction',
                            xaxis_title='Date',
                            yaxis_title='Close Price',
                            template='plotly_dark',
                            hovermode='x unified',
                            xaxis_rangeslider_visible=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Create forecast components plot using Plotly
                        # Instead of using predict_components, we'll extract the components from the predictions DataFrame
                        components = predictions[['trend', 'yearly', 'weekly', 'daily']]
                        n_components = len(components.columns)
                        
                        fig_components = make_subplots(rows=n_components, cols=1, 
                                                       subplot_titles=components.columns)
                        
                        for i, component in enumerate(components.columns, start=1):
                            fig_components.add_trace(
                                go.Scatter(x=predictions['ds'], y=components[component], 
                                           mode='lines', name=component),
                                row=i, col=1
                            )
                        
                        fig_components.update_layout(height=300*n_components, 
                                                     title_text="Forecast Components",
                                                     showlegend=False)
                        
                        st.plotly_chart(fig_components, use_container_width=True)

                        st.subheader("Predicted Prices")
                        pred_df = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict)
                        pred_df.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                        st.dataframe(pred_df)

                        # Provide download link for predictions
                        st.markdown(get_table_download_link(pred_df, f"{ticker}_predictions.csv", "Download predictions CSV"), unsafe_allow_html=True)

                        news = fetch_news(company_name)
                        st.subheader("Latest News")
                        for item in news:
                            st.markdown(f"[{item['title']}]({item['link']}) ({item['pubDate']})")
                    else:
                        st.error("Failed to generate predictions. The predicted data is None.")
                else:
                    st.error("Failed to train the Prophet model. Please try a different dataset.")
            else:
                st.error("Failed to fetch stock data. Please try again.")

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def explore_data():
    st.header("Explore Stock Data")

    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Select Company", [company for company, _ in COMPANIES])
    
    with col2:
        period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

    company_name, ticker = next((name, symbol) for name, symbol in COMPANIES if name == company)

    if st.button("Explore Data"):
        with st.spinner("Fetching and analyzing data..."):
            try:
                data = yf.download(ticker, period=period)

                if data is not None and not data.empty:
                    st.subheader(f"{company_name} Stock Data")
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "Price History", "OHLC", "Technical Indicators", 
                        "Volume Analysis", "Returns Distribution", "Moving Averages"
                    ])
                    
                    with tab1:
                        st.markdown("""
                        ### Price History
                        This chart shows the historical closing prices of the stock over time. 
                        
                        **How to use**: 
                        - Use the slider at the bottom to zoom in on specific time periods.
                        - Hover over the line to see exact prices at different dates.
                        
                        **How it helps**: 
                        - Identify long-term trends in the stock's price.
                        - Spot key price levels and potential support/resistance areas.
                        - Understand the overall price movement and volatility of the stock.
                        """)
                        
                        fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
                    
                    # Add rolling mean and standard deviation
                    data['Rolling_Mean'] = data['Close'].rolling(window=20).mean()
                    data['Rolling_Std'] = data['Close'].rolling(window=20).std()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Rolling_Mean'], mode='lines', name='20-day Rolling Mean', line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Rolling_Std'], mode='lines', name='20-day Rolling Std', line=dict(dash='dot')))
                    
                    fig.update_layout(title=f"{company_name} Stock Price History",
                                      xaxis_title="Date",
                                      yaxis_title="Price",
                                      hovermode="x unified",
                                      template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        st.markdown("""
                        ### OHLC (Open-High-Low-Close) Chart
                        This candlestick chart shows the opening, high, low, and closing prices for each trading day.
                        
                        **How to use**:
                        - Green candles indicate days where the closing price was higher than the opening price.
                        - Red candles indicate days where the closing price was lower than the opening price.
                        - The thin lines (wicks) show the high and low prices for the day.
                        
                        **How it helps**:
                        - Identify daily price movements and volatility.
                        - Spot potential reversal patterns or trends.
                        - Understand the intraday price action of the stock.
                        """)
                        
                        ohlc_fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                                  open=data['Open'],
                                                                  high=data['High'],
                                                                  low=data['Low'],
                                                                  close=data['Close'])])
                        ohlc_fig.update_layout(title=f"{company_name} OHLC Chart",
                                               xaxis_title="Date",
                                               yaxis_title="Price",
                                               template="plotly_dark",
                                               xaxis_rangeslider_visible=False)
                        st.plotly_chart(ohlc_fig, use_container_width=True)

                    with tab3:
                        st.markdown("""
                        ### Technical Indicators
                        This chart displays various technical indicators alongside the stock price.
                        
                        **How to use**:
                        - The top chart shows the stock price with SMA (Simple Moving Average), EMA (Exponential Moving Average), and Bollinger Bands.
                        - The bottom chart shows the RSI (Relative Strength Index).
                        
                        **How it helps**:
                        - Identify potential buy/sell signals using moving average crossovers.
                        - Spot overbought or oversold conditions using RSI.
                        - Understand price volatility using Bollinger Bands.
                        - Combine multiple indicators for more robust trading decisions.
                        """)
                        
                        data['SMA_20'] = data['Close'].rolling(window=20).mean()
                        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
                        data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data['Close'])
                        data['RSI'] = calculate_rsi(data['Close'])

                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, 
                                            subplot_titles=("Price and Indicators", "RSI"),
                                            row_heights=[0.7, 0.3])

                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_BB'], mode='lines', name='Upper BB'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_BB'], mode='lines', name='Lower BB'), row=1, col=1)

                        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        fig.update_layout(height=800, title_text=f"{company_name} Technical Indicators",
                                          hovermode="x unified", template="plotly_dark")
                        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
                        fig.update_yaxes(title_text="Price", row=1, col=1)
                        fig.update_yaxes(title_text="RSI", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        st.markdown("""
                        ### Volume Analysis
                        This chart shows the trading volume alongside the stock price.
                        
                        **How to use**:
                        - The top chart displays the daily trading volume.
                        - The bottom chart shows the corresponding stock price.
                        
                        **How it helps**:
                        - Identify periods of high and low trading activity.
                        - Confirm price trends (higher volume often validates price movements).
                        - Spot potential reversals (e.g., high volume at price extremes).
                        - Understand the liquidity of the stock.
                        """)
                        
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, 
                                            subplot_titles=("Volume", "Price"),
                                            row_heights=[0.7, 0.3])

                        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'), row=2, col=1)

                        fig.update_layout(height=600, title_text=f"{company_name} Volume Analysis",
                                          hovermode="x unified", template="plotly_dark")
                        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
                        fig.update_yaxes(title_text="Volume", row=1, col=1)
                        fig.update_yaxes(title_text="Price", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True)

                    with tab5:
                        st.markdown("""
                        ### Returns Distribution
                        This histogram shows the distribution of daily returns for the stock.
                        
                        **How to use**:
                        - The x-axis represents the daily return percentages.
                        - The y-axis shows the frequency of each return range.
                        
                        **How it helps**:
                        - Understand the volatility and risk profile of the stock.
                        - Identify the most common daily return ranges.
                        - Spot any skewness or unusual patterns in returns.
                        - Compare the stock's return distribution to a normal distribution for risk assessment.
                        """)
                        
                        returns = data['Close'].pct_change().dropna()
                        fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50)])
                        fig.update_layout(title=f"{company_name} Returns Distribution",
                                          xaxis_title="Daily Returns",
                                          yaxis_title="Frequency",
                                          template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab6:
                        st.markdown("""
                        ### Moving Averages
                        This chart shows the stock price along with 50-day and 200-day Simple Moving Averages (SMA).
                        
                        **How to use**:
                        - The blue line represents the stock's closing price.
                        - The orange line is the 50-day SMA.
                        - The green line is the 200-day SMA.
                        
                        **How it helps**:
                        - Identify long-term trends in the stock price.
                        - Spot potential buy/sell signals when the price crosses above/below the moving averages.
                        - Recognize "Golden Cross" (50-day crosses above 200-day) and "Death Cross" (50-day crosses below 200-day) signals.
                        - Understand the overall momentum of the stock.
                        """)
                        
                        data['SMA_50'] = data['Close'].rolling(window=50).mean()
                        data['SMA_200'] = data['Close'].rolling(window=200).mean()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-day SMA'))
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-day SMA'))

                        fig.update_layout(title=f"{company_name} Moving Averages",
                                          xaxis_title="Date",
                                          yaxis_title="Price",
                                          template="plotly_dark",
                                          hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

                    # Display key statistics
                    st.subheader("Key Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                        st.metric("52 Week High", f"${data['High'].max():.2f}")
                    with col2:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
                        st.metric("52 Week Low", f"${data['Low'].min():.2f}")
                    with col3:
                        returns = (data['Close'].pct_change() * 100).dropna()
                        st.metric("Avg Daily Return", f"{returns.mean():.2f}%")
                        st.metric("Return Volatility", f"{returns.std():.2f}%")

                    # Display news
                    st.subheader("Latest News")
                    news = fetch_news(company_name)
                    for item in news:
                        st.markdown(f"[{item['title']}]({item['link']}) ({item['pubDate']})")
                    
                else:
                    st.error("Failed to fetch data. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please try a different time period or check your internet connection.")

if __name__ == "__main__":
    main()