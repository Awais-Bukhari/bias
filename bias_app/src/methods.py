import numpy as np
from scipy.optimize import brentq
from scipy.stats import gamma

def linear_scaling(train, apply_df):
    f = train["OBS_mm"].mean() / (train["ERA5_mm"].mean() + 1e-9)
    return {"factor": f}, np.clip(apply_df["ERA5_mm"].values * f, 0, None)

def variance_scaling(train, apply_df):
    mu_o, mu_s = train["OBS_mm"].mean(), train["ERA5_mm"].mean()
    sd_o, sd_s = train["OBS_mm"].std(), train["ERA5_mm"].std()
    p = {"mu_o": mu_o, "mu_s": mu_s, "sd_o": sd_o, "sd_s": sd_s}
    return p, np.clip(mu_o + (apply_df["ERA5_mm"].values - mu_s) * (sd_o / (sd_s + 1e-9)), 0, None)

def power_transformation(train, apply_df):
    obs, sim = train["OBS_mm"].values, train["ERA5_mm"].values
    cv_o = obs.std() / (obs.mean() + 1e-9)
    f = lambda b: (np.power(np.clip(sim, 0, None) + 1e-9, b).std() /
                   (np.power(np.clip(sim, 0, None) + 1e-9, b).mean() + 1e-9)) - cv_o
    b = brentq(f, 0.1, 3.0)
    a = obs.mean() / (np.power(np.clip(sim, 0, None) + 1e-9, b).mean() + 1e-9)
    return {"a": a, "b": b}, np.clip(a * np.power(np.clip(apply_df["ERA5_mm"].values, 0, None) + 1e-9, b), 0, None)

def loci(train, apply_df, wet_day=0.1):
    obs, sim = train["OBS_mm"].values, train["ERA5_mm"].values
    wf = (obs > wet_day).mean()
    pth = np.quantile(sim, 1 - wf)
    wobs = obs[obs > wet_day].mean()
    wsim = sim[sim > pth].mean() if (sim > pth).any() else 1e-9
    S = wobs / (wsim - pth + 1e-9)
    a = apply_df["ERA5_mm"].values
    return {"S": S, "wt_sim": pth, "wt_obs": wobs}, np.clip(np.where(a > pth, S * (a - pth), 0.0), 0, None)

def parametric_qm(train, apply_df, wet_day=0.1):
    ow = train.loc[train["OBS_mm"] > wet_day, "OBS_mm"].values
    sw = train.loc[train["ERA5_mm"] > wet_day, "ERA5_mm"].values
    op, sp = gamma.fit(ow, floc=0), gamma.fit(sw, floc=0)
    a = apply_df["ERA5_mm"].values
    wm = a > wet_day
    out = np.zeros_like(a, dtype=float)
    cdf = np.clip(gamma.cdf(a[wm], *sp), 1e-6, 1 - 1e-6)
    out[wm] = gamma.ppf(cdf, *op)
    return {"obs_params": op, "sim_params": sp}, np.clip(out, 0, None)

METHODS = {
    "Linear_Scaling": linear_scaling,
    "Variance_Scaling": variance_scaling,
    "Power_Transformation": power_transformation,
    "LOCI": loci,
    "Parametric_QM": parametric_qm,
}

def apply_correction_to_grid(raw, method, p, wet_day=0.1):
    if method == "Linear_Scaling":
        return np.clip(raw * p["factor"], 0, None)
    if method == "Variance_Scaling":
        return np.clip(p["mu_o"] + (raw - p["mu_s"]) * (p["sd_o"] / p["sd_s"]), 0, None)
    if method == "Power_Transformation":
        return np.clip(p["a"] * np.power(np.clip(raw, 0, None) + 1e-9, p["b"]), 0, None)
    if method == "LOCI":
        return np.clip(np.where(raw > p["wt_sim"], p["S"] * (raw - p["wt_sim"]), 0.0), 0, None)
    if method == "Parametric_QM":
        wet = raw > wet_day
        out = np.zeros_like(raw, dtype=float)
        cdf = np.clip(gamma.cdf(raw[wet], *p["sim_params"]), 1e-6, 1 - 1e-6)
        out[wet] = gamma.ppf(cdf, *p["obs_params"])
        return np.clip(out, 0, None)
    raise ValueError(method)