"""Rolling-origin backtest: for each test year T, fit on years < T, predict T."""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("prophet.plot").disabled = True


def _site_stats(train):
    g = train.groupby("site")
    last = train.sort_values("year").groupby("site").alt_cm.last().rename("last")
    return pd.concat([g.alt_cm.mean().rename("site_mean"), last], axis=1)


def persistence(train, test):
    return test.site.map(_site_stats(train)["last"])


def site_mean(train, test):
    return test.site.map(_site_stats(train)["site_mean"])


def _fe_design(df, site_means, year_means, use_features, scaler=None):
    """Site fixed effect (as an offset) + common linear trend (+ winter features)."""
    X = pd.DataFrame({"t": df.year - 2000})
    if use_features:
        X[FEATURE_COLS] = df[FEATURE_COLS].values
    return X


def fixed_effects(train, test, use_features=True):
    """y_it - site_mean_i = b*t + c'x_it, fitted with ridge on within-site variation."""
    sm = train.groupby("site").alt_cm.mean()
    tm = train.groupby("site").year.mean()
    cols = ["year"] + (FEATURE_COLS if use_features else [])
    tr = train[cols].copy()
    tr_means = train.groupby("site")[cols].transform("mean")
    Xtr = (tr - tr_means).values
    ytr = (train.alt_cm - train.site.map(sm)).values
    sc = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13), fit_intercept=False).fit(sc.transform(Xtr), ytr)
    site_x_means = train.groupby("site")[cols].mean()
    Xte = (test[cols].values - site_x_means.reindex(test.site).values)
    pred = test.site.map(sm).values + model.predict(sc.transform(Xte))
    coefs = pd.Series(model.coef_ / sc.scale_, index=cols)
    return pd.Series(pred, index=test.index), coefs


def gbm(train, test):
    """Gradient boosting on site history + location + winter features."""
    def design(df, hist):
        st = _site_stats(hist)
        X = df[["lat", "lon"] + FEATURE_COLS].copy()
        X["t"] = df.year - 2000
        X["site_mean"] = df.site.map(st.site_mean)
        X["last"] = df.site.map(st["last"])
        return X
    # build training rows with features computed only from each row's own past
    rows = []
    for y in sorted(train.year.unique())[2:]:
        hist, cur = train[train.year < y], train[train.year == y]
        rows.append((design(cur, hist), cur.alt_cm))
    Xtr = pd.concat([r[0] for r in rows]); ytr = pd.concat([r[1] for r in rows])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    return pd.Series(m.predict(design(test, train)), index=test.index)


def prophet_pooled(train, test):
    """The paper's setup, with the feature bug fixed: one Prophet on all sites."""
    from prophet import Prophet
    df = train.rename(columns={"alt_cm": "y"}).assign(ds=pd.to_datetime(train.year, format="%Y"))
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    for f in FEATURE_COLS:
        m.add_regressor(f)
    m.fit(df[["ds", "y"] + FEATURE_COLS])
    fut = test.assign(ds=pd.to_datetime(test.year, format="%Y"))[["ds"] + FEATURE_COLS]
    return pd.Series(m.predict(fut).yhat.values, index=test.index)


def prophet_anomaly(train, test):
    """Prophet on the regional mean anomaly series, added back to each site's mean."""
    from prophet import Prophet
    sm = train.groupby("site").alt_cm.mean()
    anom = train.assign(a=train.alt_cm - train.site.map(sm))
    reg = anom.groupby("year").agg(y=("a", "mean"), **{f: (f, "mean") for f in FEATURE_COLS}).reset_index()
    reg["ds"] = pd.to_datetime(reg.year, format="%Y")
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    for f in FEATURE_COLS:
        m.add_regressor(f)
    m.fit(reg[["ds", "y"] + FEATURE_COLS])
    te = test.groupby("year")[FEATURE_COLS].mean().reset_index()
    te["ds"] = pd.to_datetime(te.year, format="%Y")
    yhat = dict(zip(te.year, m.predict(te[["ds"] + FEATURE_COLS]).yhat))
    return test.site.map(sm) + test.year.map(yhat)


MODELS = {
    "Persistence (last year)": persistence,
    "Site mean": site_mean,
    "Site mean + trend": lambda tr, te: fixed_effects(tr, te, use_features=False)[0],
    "Site mean + trend + winter features": lambda tr, te: fixed_effects(tr, te, use_features=True)[0],
    "Gradient boosting": gbm,
    "Prophet, pooled (paper setup, fixed)": prophet_pooled,
    "Prophet on regional anomaly": prophet_anomaly,
}


def rolling_backtest(panel, test_years=range(2008, 2016), models=MODELS):
    out = []
    for T in test_years:
        train, test = panel[panel.year < T], panel[panel.year == T]
        test = test[test.site.isin(train.site)]
        for name, fn in models.items():
            pred = fn(train, test)
            out.append(pd.DataFrame({"model": name, "year": T, "site": test.site,
                                     "y": test.alt_cm, "yhat": pred.values}))
    return pd.concat(out, ignore_index=True)


def score(preds):
    e = preds.yhat - preds.y
    return pd.DataFrame({"MAE": e.abs().groupby(preds.model).mean(),
                         "RMSE": np.sqrt((e ** 2).groupby(preds.model).mean()),
                         "Bias": e.groupby(preds.model).mean()})
