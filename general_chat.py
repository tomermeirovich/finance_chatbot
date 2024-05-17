import json
import streamlit as st
import yfinance as yf
import openai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from io import BytesIO

plt.style.use('seaborn-v0_8-dark')

# Set your OpenAI API key
openai.api_key = 'sk-proj-7poJ0QSmvWxvPTPxFDjcT3BlbkFJ1a47knYMxWcxEjDIwSmO'

# Define FinancialInstrument class
class FinancialInstrument:
    def __init__(self, ticker, start, end):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.data = self.get_data()
        self.log_returns()

    def __repr__(self):
        return f"FinancialInstrument(ticker={self._ticker}, start={self.start}, end={self.end})"

    def get_data(self):
        raw = yf.download(self._ticker, self.start, self.end)['Close'].to_frame()
        raw.rename(columns={'Close': 'Price'}, inplace=True)
        return raw

    def log_returns(self):
        self.data['log_returns'] = np.log(self.data['Price'] / self.data['Price'].shift(1))

    def plot_prices(self):
        plt.figure(figsize=(12, 8))
        self.data['Price'].plot()
        plt.title(f"Price Chart: {self._ticker}", fontsize=15)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    def plot_returns(self, kind='ts'):
        plt.figure(figsize=(12, 8))
        if kind == 'ts':
            self.data['log_returns'].plot()
            plt.title(f"Returns: {self._ticker}", fontsize=15)
        elif kind == 'hist':
            self.data['log_returns'].hist(bins=int(np.sqrt(len(self.data))))
            plt.title(f"Frequency of Returns: {self._ticker}", fontsize=15)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    def mean_return(self, freq=None):
        if freq is None:
            return self.data['log_returns'].mean()
        else:
            resampled_price = self.data['Price'].resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()

    def std_return(self, freq=None):
        if freq is None:
            return self.data['log_returns'].std()
        else:
            resampled_price = self.data['Price'].resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()

    def annualized_perf(self):
        mean_return = round(self.data['log_returns'].mean() * 252, 3)
        risk = round(self.data['log_returns'].std() * np.sqrt(252), 3)
        return {"Return": mean_return, "Risk": risk}

# Define RiskReturn class
class RiskReturn(FinancialInstrument):
    def __init__(self, ticker, start, end, freq=None):
        super().__init__(ticker, start, end)
        self.freq = freq

    def __repr__(self):
        return f"RiskReturn(ticker={self._ticker}, start={self.start}, end={self.end}, freq={self.freq})"

    def mean_return(self):
        if self.freq is None:
            return super().mean_return()
        else:
            resampled_price = self.data['Price'].resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()

    def std_return(self):
        if self.freq is None:
            return super().std_return()
        else:
            resampled_price = self.data['Price'].resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()

    def annualized_perf(self):
        mean_return = round(self.mean_return() * 252, 3)
        risk = round(self.std_return() * np.sqrt(252), 3)
        return {"Return": mean_return, "Risk": risk}

# Define additional functions
def plot_stock_with_indicators(ticker, sma_window=None, ema_window=None):
    stock = FinancialInstrument(ticker, '2010-01-01', dt.datetime.now())
    plt.figure(figsize=(12, 8))
    plt.plot(stock.data.index, stock.data['Price'], label='Close Price')

    if sma_window:
        sma = stock.data['Price'].rolling(window=sma_window).mean()
        plt.plot(stock.data.index, sma, label=f'SMA {sma_window}')
    
    if ema_window:
        ema = stock.data['Price'].ewm(span=ema_window, adjust=False).mean()
        plt.plot(stock.data.index, ema, label=f'EMA {ema_window}')

    plt.title(f'{ticker} Stock Price with Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

# Function mapping
functions = {
    'get_stock_price': {
        "function": lambda ticker: FinancialInstrument(ticker, '2010-01-01', dt.datetime.now()).data['Price'].iloc[-1],
        "description": "Get the current stock price",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    'plot_prices': {
        "function": lambda ticker: FinancialInstrument(ticker, '2010-01-01', dt.datetime.now()).plot_prices(),
        "description": "Plot the stock price",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    'plot_returns': {
        "function": lambda ticker, kind='ts': FinancialInstrument(ticker, '2010-01-01', dt.datetime.now()).plot_returns(kind),
        "description": "Plot the stock returns",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "kind": {"type": "string", "enum": ["ts", "hist"], "default": "ts"}
            },
            "required": ["ticker"]
        }
    },
    'annualized_perf': {
        "function": lambda ticker: FinancialInstrument(ticker, '2010-01-01', dt.datetime.now()).annualized_perf(),
        "description": "Get annualized performance metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    'plot_stock_with_indicators': {
        "function": plot_stock_with_indicators,
        "description": "Plot the stock price with SMA and/or EMA",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "sma_window": {"type": ["integer", "null"]},
                "ema_window": {"type": ["integer", "null"]}
            },
            "required": ["ticker"]
        }
    }
}

# Initialize Streamlit app
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('Stock Analysis Chatbot Assistant - SACA')
user_input = st.text_input('Your input: ')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages'],
            max_tokens=150,
            functions=[{
                "name": name,
                "description": details["description"],
                "parameters": details["parameters"]
            } for name, details in functions.items()]
        )

        # Display the model's response
        if response.choices:
            choice = response.choices[0]
            message_content = choice['message']['content']
            st.write(f'Response: {message_content}')

            # Check for function calls
            if 'function_call' in choice['message']:
                function_call = choice['message']['function_call']
                function_name = function_call['name']
                function_args = json.loads(function_call['arguments'])

                # Execute the function
                if function_name in functions:
                    function_result = functions[function_name]["function"](**function_args)
                    if isinstance(function_result, BytesIO):
                        st.image(function_result)
                    else:
                        st.write(f"Function {function_name} result: {function_result}")
                else:
                    st.error("Function not found.")
        else:
            st.write("No response from the model.")
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')