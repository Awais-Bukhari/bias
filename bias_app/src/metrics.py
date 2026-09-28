import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(obs, sim):
    obs, sim = np.asarray(obs), np.asarray(sim)
    if len(obs) < 2:
        return {"RMSE_mm": np.nan, "R2": np.nan, "NSE": np.nan}
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    r2 = np.corrcoef(obs, sim)[0, 1] ** 2
    nse = 1 - (np.sum((obs - sim) ** 2) / (np.sum((obs - np.mean(obs)) ** 2) + 1e-9))
    return {"RMSE_mm": rmse, "R2": r2, "NSE": nse}

def stats_scatter_fig(obs, pred, title):
    """Returns a matplotlib figure (does NOT call plt.show())."""
    obs, pred = np.asarray(obs), np.asarray(pred)
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    r2 = np.corrcoef(obs, pred)[0, 1] ** 2
    bias = np.mean(pred - obs)
    nse = 1 - (np.sum((obs - pred) ** 2) / (np.sum((obs - np.mean(obs)) ** 2) + 1e-9))

    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.scatter(obs, pred, s=14, alpha=0.35, color="#5b8db8", edgecolor="none")
    ax.plot(lims, lims, ls="--", color="#999999", lw=1, label="1:1 Line")
    m, b = np.polyfit(obs, pred, 1)
    xs = np.linspace(*lims, 50)
    ax.plot(xs, m * xs + b, color="#c0392b", lw=1.8, label="Regression Line")
    ax.set_title(f"{title}\nR²={r2:.2f} | NSE={nse:.2f} | RMSE={rmse:.2f} | Bias={bias:.2f}", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig