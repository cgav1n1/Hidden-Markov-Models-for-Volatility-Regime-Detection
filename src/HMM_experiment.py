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
    X = df.loc[:'2019-12-31', 'Log Return'].dropna().to_numpy().reshape(-1, 1)

    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=42)

    model.fit(X)
    print("Converged:", model.monitor_.converged)
    print("Log-likelihood:", model.score(X))
    
    volatilities = np.sqrt(model.covars_.reshape(-1))
    low_vol_state = np.argmin(volatilities)
    high_vol_state = np.argmax(volatilities)
    
    avg_lasting_low_vol_state = 1/(1-model.transmat_[low_vol_state][low_vol_state])
    avg_lasting_high_vol_state = 1/(1-model.transmat_[high_vol_state][high_vol_state])
    
    #Print Training Results :
    print("Training Results :")
    
    print(f"Low-vol state: {low_vol_state}")
    print(f"Daily volatility: {volatilities[low_vol_state]:.3f}%")
    print(f"Expected duration: {avg_lasting_low_vol_state} days")

    print(f"\nHigh-vol state: {high_vol_state}")
    print(f"Daily volatility: {volatilities[high_vol_state]:.3f}%")
    print(f"Expected duration: {avg_lasting_high_vol_state} days")
    
    state_probabilities = model.predict_proba(X)
    high_vol_probabilities = state_probabilities[:, high_vol_state]
    
    # X : données utilisées pour entraîner le HMM
    # Shape attendu : (n_observations, 1)

    hidden_states = model.predict(X)

    # Identification automatique des régimes
    volatilities = np.sqrt(model.covars_.reshape(-1))
    low_vol_state = np.argmin(volatilities)
    high_vol_state = np.argmax(volatilities)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(X[:, 0], color="lightgray", alpha=0.6)

    for state, label, color in [
        (low_vol_state, "Low volatility", "steelblue"),
        (high_vol_state, "High volatility", "tomato")
    ]:
        mask = hidden_states == state
        ax.scatter(
            np.where(mask)[0],
            X[mask, 0],
            s=7,
            color=color,
            label=label
        )

    ax.set_title("HMM Volatility Regimes — Viterbi Decoding")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Log return")
    ax.legend()
    plt.show()
    
    
    
    