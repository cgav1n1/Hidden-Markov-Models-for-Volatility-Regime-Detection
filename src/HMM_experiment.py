import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import pandas as pd


# Experiment Parameters :

def initiate_universe():
    stock_universe = "^GSPC"
    df = yf.download(stock_universe, start="2005-01-01", end="2025-12-31")
    df.columns = df.columns.get_level_values(0)

    df['Log Return'] = 100*np.log(df['Close']/df['Close'].shift(1))
    df = df.drop_duplicates()
    df = df.dropna(how='any')
    return df

def run_model(n_components) :
    df = initiate_universe()
    