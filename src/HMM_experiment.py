
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from hmmlearn.hmm import GaussianHMM


# ============================================================
# CONFIGURATION
# ============================================================

STOCK_UNIVERSE = "^GSPC"

START_DATE = "2005-01-01"
END_DATE = "2026-01-01"  # yfinance: end date is exclusive
TRAIN_END_DATE = "2019-12-31"

RANDOM_STATE = 42
N_ITER = 1000
TOL = 1e-4

# Results are saved relative to the project root.
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================
# 1. DATA
# ============================================================

def initiate_universe():
    """
    Download S&P 500 data and compute daily log returns
    expressed in percentage points.
    """

    df = yf.download(
        STOCK_UNIVERSE,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded from yfinance.")

    # Handle yfinance MultiIndex columns.
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
# 2. PREPARE TRAIN / TEST DATA
# ============================================================

def prepare_data(df):
    """
    Chronological split:
    - Training: until December 2019
    - Testing: after December 2019
    """

    train_df = df.loc[:TRAIN_END_DATE].copy()
    test_df = df.loc[TRAIN_END_DATE:].copy()

    # Avoid including the training boundary in the test set.
    test_df = test_df.loc[test_df.index > pd.Timestamp(TRAIN_END_DATE)]

    X_train = train_df["Log Return"].to_numpy().reshape(-1, 1)
    X_test = test_df["Log Return"].to_numpy().reshape(-1, 1)

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    return train_df, test_df, X_train, X_test


# ============================================================
# 3. MODEL PARAMETERS
# ============================================================

def extract_parameters(model):
    """
    Return:
    - DataFrame with regime characteristics
    - Transition matrix ordered by increasing volatility
    - Original state IDs sorted by volatility
    """

    means = model.means_.reshape(-1)
    variances = model.covars_.reshape(-1)
    volatilities = np.sqrt(variances)

    # HMM state labels are arbitrary.
    # Sort them by estimated daily volatility.
    sorted_states = np.argsort(volatilities)

    results = []

    for rank, state in enumerate(sorted_states):

        persistence = model.transmat_[state, state]

        expected_duration = (
            1 / (1 - persistence)
            if persistence < 1
            else np.inf
        )

        results.append({
            "Regime": rank + 1,
            "Original State": int(state),
            "Mean (%)": means[state],
            "Variance": variances[state],
            "Volatility (%)": volatilities[state],
            "Persistence": persistence,
            "Expected Duration (days)": expected_duration
        })

    parameters_df = pd.DataFrame(results)

    # Reorder both rows and columns of the transition matrix.
    sorted_transition_matrix = model.transmat_[
        np.ix_(sorted_states, sorted_states)
    ]

    regime_labels = [
        f"Regime {i + 1}"
        for i in range(model.n_components)
    ]

    transition_df = pd.DataFrame(
        sorted_transition_matrix,
        index=regime_labels,
        columns=regime_labels
    )

    return parameters_df, transition_df, sorted_states


# ============================================================
# 4. HISTORICAL REGIMES PLOT
# ============================================================

def plot_historical_regimes(
    model,
    train_df,
    X_train,
    sorted_states,
    output_path
):
    """
    Plot historical daily log returns colored by Viterbi state.

    The plot uses the full training sequence, so it represents
    retrospective regime classification, not real-time signals.
    """

    hidden_states = model.predict(X_train)

    dates = train_df.index
    returns = X_train[:, 0]

    n_components = model.n_components

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        dates,
        returns,
        color="lightgray",
        alpha=0.5,
        linewidth=0.8,
        zorder=1
    )

    cmap = plt.get_cmap("viridis", n_components)

    for rank, state in enumerate(sorted_states):

        mask = hidden_states == state

        volatility = np.sqrt(
            model.covars_.reshape(-1)[state]
        )

        ax.scatter(
            dates[mask],
            returns[mask],
            s=8,
            color=cmap(rank),
            label=(
                f"Regime {rank + 1} "
                f"(vol={volatility:.2f}%)"
            ),
            alpha=0.85,
            zorder=2
        )

    ax.axhline(
        y=0,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6
    )

    ax.set_title(
        f"Gaussian HMM — {n_components} Volatility Regimes\n"
        f"S&P 500 | Viterbi Decoding | Training Period"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Log Return (%)")

    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 5. MODEL COMPARISON
# ============================================================

def update_model_comparison(metrics):
    """
    Save model metrics in model_comparison.csv.

    Each column represents a number of states.
    Running the same n_components again updates its column.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    n_components = metrics["n_components"]
    column_name = f"{n_components} states"

    new_results = pd.Series({
        "Train Log-Likelihood": metrics["train_loglik"],
        "Train BIC": metrics["bic"],
        "Train AIC": metrics["aic"],
        "Test Log-Likelihood": metrics["test_loglik"],
        "Test Log-Likelihood per Observation": (
            metrics["test_loglik_per_obs"]
        ),
        "Converged": metrics["converged"],
        "Iterations": metrics["iterations"],
        "Random State": metrics["random_state"]
    }, name=column_name)

    if comparison_path.exists():
        comparison_df = pd.read_csv(
            comparison_path,
            index_col=0
        )
    else:
        comparison_df = pd.DataFrame()

    # Overwrites only the selected model column.
    comparison_df[column_name] = new_results

    # Sort model columns by number of states.
    comparison_df = comparison_df.reindex(
        columns=sorted(
            comparison_df.columns,
            key=lambda col: int(col.split()[0])
        )
    )

    comparison_df.to_csv(comparison_path)

    return comparison_df


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def run_model(n_components, df=None, random_state=RANDOM_STATE):
    """
    Train a Gaussian HMM and save experiment results.

    Parameters
    ----------
    n_components : int
        Number of hidden states.

    df : pandas.DataFrame, optional
        Preloaded market data. If None, download data.

    random_state : int
        Seed used for HMM initialization.

    Returns
    -------
    model : GaussianHMM
        Trained HMM.

    parameters_df : pandas.DataFrame
        Characteristics of the estimated regimes.

    metrics : dict
        Model evaluation metrics.
    """

    if n_components < 2:
        raise ValueError("n_components must be >= 2.")

    if df is None:
        df = initiate_universe()

    train_df, test_df, X_train, X_test = prepare_data(df)

    print("=" * 55)
    print(f"GAUSSIAN HMM — {n_components} STATES")
    print("=" * 55)

    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations:  {len(X_test)}")

    # --------------------------------------------------------
    # Train HMM
    # --------------------------------------------------------

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=random_state
    )

    model.fit(X_train)

    # --------------------------------------------------------
    # Evaluate HMM
    # --------------------------------------------------------

    train_loglik = model.score(X_train)

    bic = model.bic(X_train)
    aic = model.aic(X_train)

    if len(X_test) > 0:
        test_loglik = model.score(X_test)
        test_loglik_per_obs = test_loglik / len(X_test)
    else:
        test_loglik = np.nan
        test_loglik_per_obs = np.nan

    metrics = {
        "n_components": n_components,
        "train_loglik": train_loglik,
        "bic": bic,
        "aic": aic,
        "test_loglik": test_loglik,
        "test_loglik_per_obs": test_loglik_per_obs,
        "converged": model.monitor_.converged,
        "iterations": model.monitor_.iter,
        "random_state": random_state
    }

    # --------------------------------------------------------
    # Extract model parameters
    # --------------------------------------------------------

    (
        parameters_df,
        transition_df,
        sorted_states
    ) = extract_parameters(model)

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    model_dir = (
        RESULTS_DIR / f"model_{n_components}_states"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save model parameters
    # --------------------------------------------------------

    parameters_df.to_csv(
        model_dir / "parameters.csv",
        index=False
    )

    transition_df.to_csv(
        model_dir / "transition_matrix.csv"
    )

    # --------------------------------------------------------
    # Save historical regimes plot
    # --------------------------------------------------------

    plot_historical_regimes(
        model=model,
        train_df=train_df,
        X_train=X_train,
        sorted_states=sorted_states,
        output_path=model_dir / "historical_regimes.png"
    )

    # --------------------------------------------------------
    # Update model comparison
    # --------------------------------------------------------

    comparison_df = update_model_comparison(metrics)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nMODEL EVALUATION")
    print("-" * 55)

    print(f"Converged:           {metrics['converged']}")
    print(f"Iterations:          {metrics['iterations']}")
    print(f"Train Log-Likelihood: {train_loglik:.2f}")
    print(f"BIC:                 {bic:.2f}")
    print(f"AIC:                 {aic:.2f}")

    if len(X_test) > 0:
        print(
            f"Test Log-Likelihood / obs: "
            f"{test_loglik_per_obs:.4f}"
        )

    print("\nREGIME CHARACTERISTICS")
    print("-" * 55)

    print(
        parameters_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nTRANSITION MATRIX")
    print("-" * 55)

    print(transition_df.round(4))

    print("\nMODEL COMPARISON")
    print("-" * 55)

    print(comparison_df.to_string())

    print(f"\nResults saved in: {model_dir}")

    return model, parameters_df, metrics


# ============================================================
# 7. RUN EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    # Download market data only once.
    df = initiate_universe()

    # Compare multiple model specifications.
    for n_states in [2, 3, 4, 5]:
        run_model(
            n_components=n_states,
            df=df
        )
