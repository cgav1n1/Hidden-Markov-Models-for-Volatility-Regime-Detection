# The goal of this file is to see how well a Gaussian Distribution can model our returns
# Goal : do our multi-states regime classification model the return better than a Gaussian distribution


from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import norm


# ============================================================
# CONFIGURATION
# ============================================================

stock_universe = "^GSPC"

start_date = "2005-01-01"
end_date = "2026-01-01"  # Exclusive in yfinance
train_end_date = "2019-12-31"

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"


# ============================================================
# 1. DATA
# ============================================================

def initiate_universe():
    """
    Download S&P 500 prices and calculate daily log returns
    in percentage points.
    """

    df = yf.download(
        stock_universe,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")].copy()

    df["Log Return"] = (
        100 * np.log(df["Close"] / df["Close"].shift(1))
    )

    df = df.dropna(subset=["Log Return"])

    return df


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

def prepare_data(df):
    """
    Training: 2005-2019
    Testing:  2020-2025
    """

    train_df = df.loc[:train_end_date].copy()

    test_df = df.loc[
        df.index > pd.Timestamp(train_end_date)
    ].copy()

    X_train = train_df["Log Return"].to_numpy()
    X_test = test_df["Log Return"].to_numpy()

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or testing dataset is empty.")

    return X_train, X_test


# ============================================================
# 3. GAUSSIAN BENCHMARK
# ============================================================

def run_gaussian_benchmark(df=None):
    """
    Fit a single Gaussian distribution using MLE.

    Returns
    -------
    metrics : dict
        Estimated parameters and evaluation metrics.
    """

    if df is None:
        df = initiate_universe()

    X_train, X_test = prepare_data(df)

    # --------------------------------------------------------
    # Estimate Gaussian parameters on TRAIN ONLY
    # --------------------------------------------------------

    mu = np.mean(X_train)

    # Maximum likelihood variance estimator: ddof=0
    variance = np.var(X_train, ddof=0)
    sigma = np.sqrt(variance)

    if sigma <= 0:
        raise ValueError("Estimated volatility must be positive.")

    # --------------------------------------------------------
    # Calculate log-likelihoods
    # --------------------------------------------------------

    train_loglik = np.sum(
        norm.logpdf(X_train, loc=mu, scale=sigma)
    )

    test_loglik = np.sum(
        norm.logpdf(X_test, loc=mu, scale=sigma)
    )

    train_loglik_per_obs = train_loglik / len(X_train)
    test_loglik_per_obs = test_loglik / len(X_test)

    # --------------------------------------------------------
    # AIC / BIC
    # --------------------------------------------------------

    # Gaussian model estimates two free parameters:
    # mean (mu) and variance (sigma^2).
    n_parameters = 2

    train_aic = (
        2 * n_parameters - 2 * train_loglik
    )

    train_bic = (
        np.log(len(X_train)) * n_parameters
        - 2 * train_loglik
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    metrics = {
        "Model": "Gaussian i.i.d.",
        "Mean (%)": mu,
        "Variance": variance,
        "Volatility (%)": sigma,
        "Train Observations": len(X_train),
        "Test Observations": len(X_test),
        "Train Log-Likelihood": train_loglik,
        "Train Log-Likelihood per Observation": (
            train_loglik_per_obs
        ),
        "Train AIC": train_aic,
        "Train BIC": train_bic,
        "Test Log-Likelihood": test_loglik,
        "Test Log-Likelihood per Observation": (
            test_loglik_per_obs
        )
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("=" * 55)
    print("GAUSSIAN IID BENCHMARK — S&P 500")
    print("=" * 55)

    print("\nESTIMATED PARAMETERS")
    print("-" * 55)
    print(f"Mean daily return:       {mu:.4f}%")
    print(f"Daily variance:          {variance:.4f}")
    print(f"Daily volatility:        {sigma:.4f}%")

    print("\nMODEL EVALUATION")
    print("-" * 55)
    print(f"Train observations:      {len(X_train)}")
    print(f"Test observations:       {len(X_test)}")
    print(f"Train Log-Likelihood:    {train_loglik:.2f}")
    print(f"Train LL / Observation:  {train_loglik_per_obs:.4f}")
    print(f"Train AIC:               {train_aic:.2f}")
    print(f"Train BIC:               {train_bic:.2f}")
    print(f"Test Log-Likelihood:     {test_loglik:.2f}")
    print(f"Test LL / Observation:   {test_loglik_per_obs:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "gaussian_benchmark.csv"

    pd.DataFrame([metrics]).to_csv(
        output_path,
        index=False
    )

    print(f"\nResults saved in: {output_path}")

    return metrics


# ============================================================
# 4. EXECUTION
# ============================================================

if __name__ == "__main__":
    run_gaussian_benchmark()
