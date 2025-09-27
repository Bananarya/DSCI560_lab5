import os, sys, io, base64, json, argparse, shutil, logging, datetime as dt
import numpy as np
import pandas as pd

def _silence_logging():
    for name in ("cmdstanpy", "prophet", "prophet.plot", "prophet.stan_backend"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = False
        # remove existing handlers that may echo INFO
        for h in list(lg.handlers):
            lg.removeHandler(h)
    logging.getLogger().setLevel(logging.WARNING)
_silence_logging()

# Matplotlib non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
try:
    import pymysql  # noqa: F401
except Exception:
    pass

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_NAME = os.getenv("DB_NAME", "dsci560_lab3")

DEFAULT_START = os.getenv("START_DATE", "2018-01-01")
DEFAULT_END   = os.getenv("END_DATE",   "2025-09-01")
DEFAULT_INIT  = float(os.getenv("INITIAL_CAPITAL", "100000"))
DEFAULT_FEE   = float(os.getenv("FEE_BPS", "10"))
DEFAULT_RF    = float(os.getenv("RISK_FREE_RATE", "0.00"))
DEFAULT_MAX   = int(os.getenv("MAX_TICKERS", "0"))

def make_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True, future=True)

def safe_name(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

def fetch_portfolio_tickers(engine, name, max_tickers=0):
    sql = text("""
        SELECT t.id AS ticker_id, t.symbol
        FROM portfolios p
        JOIN portfolio_stocks ps ON ps.portfolio_id = p.id
        JOIN tickers t ON t.id = ps.ticker_id
        WHERE p.name = :pname
        ORDER BY t.symbol
    """)
    df = pd.read_sql(sql, engine, params={"pname": name})
    if max_tickers and max_tickers > 0:
        df = df.head(max_tickers)
    if df.empty:
        raise RuntimeError(f"No stocks found in portfolio '{name}'.")
    return df

def fetch_prices_for_portfolio(engine, name, start_date, end_date):
    sql = text("""
        SELECT pr.ticker_id, pr.dt, pr.`open`, pr.`close`, pr.adj_close
        FROM prices pr
        JOIN portfolio_stocks ps ON ps.ticker_id = pr.ticker_id
        JOIN portfolios p ON p.id = ps.portfolio_id
        WHERE p.name = :pname AND pr.dt BETWEEN :s AND :e
        ORDER BY pr.ticker_id, pr.dt
    """)
    df = pd.read_sql(sql, engine, params={"pname": name, "s": start_date, "e": end_date})
    if df.empty:
        raise RuntimeError("No price data in the given range.")
    df["dt"] = pd.to_datetime(df["dt"])
    # build adj_open consistently with adj_close
    adj_factor = df["adj_close"] / df["close"]
    df["adj_open"] = df["open"] * adj_factor
    return df

def signals_sma(prices, fast=10, slow=30):
    """Simple Moving Average crossover -> desired_position in {0,1} (end-of-day)."""
    out = []
    for tid, g in prices.groupby("ticker_id", sort=True):
        g = g.sort_values("dt").copy()
        sma_f = g["adj_close"].rolling(fast).mean()
        sma_s = g["adj_close"].rolling(slow).mean()
        desired = (sma_f > sma_s).fillna(False).astype(int)  # long when fast>slow
        out.append(pd.DataFrame({
            "ticker_id": tid,
            "dt": g["dt"].values,
            "desired_position": desired.values
        }))
    return pd.concat(out, ignore_index=True)

def _prophet_insample(df_hist):
    """
    Fit Prophet once on history and return in-sample predictions on same ds
    along with yhat_lower/upper. No console spam thanks to _silence_logging().
    """
    from prophet import Prophet
    dfp = df_hist.rename(columns={"dt": "ds", "adj_close": "y"})[["ds", "y"]].copy()
    m = Prophet(yearly_seasonality="auto")
    m.fit(dfp)
    fcst = m.predict(dfp[["ds"]])  # in-sample only
    fcst = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    return fcst

def signals_prophet_snr(prices, horizon=5, vol_lookback=20, entry=0.5, exit=0.1):
    """
    SNR = expected price change over horizon / residual volatility
    Enter long when SNR>entry; exit when SNR<exit
    """
    results = []
    for tid, g in prices.groupby("ticker_id", sort=True):
        g = g.sort_values("dt").copy()
        fcst = _prophet_insample(g.rename(columns={"close":"_close"}))  # keep adj_close
        df = g.merge(fcst, left_on="dt", right_on="ds", how="left").drop(columns=["ds"])
        resid = df["adj_close"] - df["yhat"]
        sigma = resid.rolling(vol_lookback, min_periods=vol_lookback//2).std()
        # using in-sample yhat to approximate future h-step expectation
        yhat_lead = df["yhat"].shift(-horizon)
        exp_change = yhat_lead - df["yhat"]
        snr = exp_change / sigma
        desired = []
        state = 0
        for z in snr.fillna(0).values:
            if state == 0 and z > entry:
                state = 1
            elif state == 1 and z < exit:
                state = 0
            desired.append(state)
        res = pd.DataFrame({"ticker_id": tid, "dt": df["dt"].values,
                            "desired_position": desired})
        results.append(res)
    return pd.concat(results, ignore_index=True)

def signals_prophet_bands(prices, alpha=0.0, take_profit="mid"):
    """
    Dynamic bands from Prophet yhat_lower/upper. Buy when price < lower*(1-alpha),
    exit when price > upper*(1+alpha) or take-profit at mid(yhat) if specified.
    """
    out = []
    for tid, g in prices.groupby("ticker_id", sort=True):
        g = g.sort_values("dt").copy()
        fcst = _prophet_insample(g.rename(columns={"close":"_close"}))
        df = g.merge(fcst, left_on="dt", right_on="ds", how="left").drop(columns=["ds"])
        low  = df["yhat_lower"] * (1.0 - alpha)
        high = df["yhat_upper"] * (1.0 + alpha)
        mid  = df["yhat"]
        px   = df["adj_close"]
        desired = []
        state = 0
        for i in range(len(df)):
            p = px.iloc[i]
            if state == 0:
                if p < low.iloc[i]:
                    state = 1
            else:  # state==1, holding
                if p > high.iloc[i]:
                    state = 0
                elif take_profit == "mid" and p > mid.iloc[i]:
                    state = 0
            desired.append(state)
        out.append(pd.DataFrame({"ticker_id": tid, "dt": df["dt"].values,
                                 "desired_position": desired}))
    return pd.concat(out, ignore_index=True)


def backtest_one_slot(price_df, signal_df, symbol, slot_cash0, fee_bps=10.0):
    fee_rate = fee_bps / 1e4
    df = price_df.merge(signal_df, on=["ticker_id","dt"], how="left").sort_values("dt")
    df["desired_position"] = df["desired_position"].fillna(0).astype(int)
    df["desired_prev"] = df["desired_position"].shift(1).fillna(0).astype(int)
    cash, shares = slot_cash0, 0.0
    pv_rows, tr_rows = [], []
    for _, row in df.iterrows():
        dtv = pd.Timestamp(row["dt"])
        open_px = float(row["adj_open"])
        close_px = float(row["adj_close"])
        want = int(row["desired_prev"])
        pos = 1 if shares > 0 else 0
        if want != pos:
            if want == 1 and pos == 0 and open_px > 0 and cash > 0:
                buy_sh = cash / open_px
                notional = buy_sh * open_px
                fee = notional * fee_rate
                cash = cash - notional - fee
                shares += buy_sh
                tr_rows.append(dict(dt=dtv, symbol=symbol, action="BUY",
                                    price=open_px, shares=buy_sh, fee=fee,
                                    cash_after=cash, shares_after=shares))
            elif want == 0 and pos == 1 and open_px > 0 and shares > 0:
                notional = shares * open_px
                fee = notional * fee_rate
                cash = cash + notional - fee
                tr_rows.append(dict(dt=dtv, symbol=symbol, action="SELL",
                                    price=open_px, shares=shares, fee=fee,
                                    cash_after=cash, shares_after=0.0))
                shares = 0.0
        pv_rows.append(dict(dt=dtv, portfolio_value=cash + shares * close_px))
    pv_df = pd.DataFrame(pv_rows).sort_values("dt")
    trades = pd.DataFrame(tr_rows).sort_values("dt")
    return pv_df, trades

def aggregate_portfolio(pv_list):
    agg = None
    for idx, pv in enumerate(pv_list):
        col = f"slot_{idx+1}"
        pv_renamed = pv.rename(columns={"portfolio_value": col})
        agg = pv_renamed if agg is None else agg.merge(pv_renamed, on="dt", how="outer")
    agg = agg.sort_values("dt")
    value_cols = [c for c in agg.columns if c != "dt"]
    agg[value_cols] = agg[value_cols].ffill().bfill()
    agg["portfolio_value"] = agg[value_cols].sum(axis=1)
    return agg[["dt","portfolio_value"]]

def compute_metrics(pv_df, rf=0.0):
    pv_df = pv_df.sort_values("dt")
    # ensure daily uniqueness
    pv_df["dt_day"] = pv_df["dt"].dt.normalize()
    pv = pv_df.drop_duplicates("dt_day")["portfolio_value"].to_numpy()
    if len(pv) < 2:
        return {}
    # daily returns
    ret = np.zeros_like(pv, dtype=float)
    ret[1:] = pv[1:] / pv[:-1] - 1.0
    daily = ret[1:]
    mu = float(np.nanmean(daily))
    sd = float(np.nanstd(daily, ddof=1))
    ann_vol = float(sd * np.sqrt(252)) if np.isfinite(sd) else None
    rf_daily = rf / 252.0
    sharpe = float(((mu - rf_daily) / sd) * np.sqrt(252)) if (sd and sd > 1e-12) else None
    # drawdown on equity curve
    equity = np.cumprod(1 + daily) if daily.size else np.array([])
    if equity.size:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0
    # CAGR
    days = max(len(pv) - 1, 1)
    cagr = float((pv[-1] / pv[0])**(252.0 / days) - 1.0)
    return {
        "start_value": float(pv[0]),
        "end_value": float(pv[-1]),
        "total_return": float(pv[-1] / pv[0] - 1.0),
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_days": int(days)  # <== correct Num Days (unique dates - 1)
    }


def _fig_to_uri():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def plot_single_ticker_uri(price_df, signal_df, trades_df, symbol, label):
    df = price_df.sort_values("dt").copy()
    plt.figure(figsize=(12,5))
    plt.plot(df["dt"], df["adj_close"], label=f"{symbol} Adj Close", linewidth=1.5)
    plt.plot(df["dt"], df["adj_close"].rolling(10).mean(), linewidth=1.0, alpha=0.7, label="SMA 10")
    plt.plot(df["dt"], df["adj_close"].rolling(30).mean(), linewidth=1.0, alpha=0.7, label="SMA 30")
    if trades_df is not None and not trades_df.empty:
        buys  = trades_df[trades_df["action"]=="BUY"]["dt"]
        sells = trades_df[trades_df["action"]=="SELL"]["dt"]
        px_map = df.set_index("dt")["adj_open"]
        if len(buys):  plt.scatter(buys,  px_map.reindex(buys).values,  marker="^", s=45, edgecolors="k", label="BUY",  zorder=4)
        if len(sells): plt.scatter(sells, px_map.reindex(sells).values, marker="v", s=45, edgecolors="k", label="SELL", zorder=4)
    plt.title(f"{symbol} | {label} Signals & Trades")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.grid(True, alpha=0.3); plt.legend()
    return _fig_to_uri()

def plot_portfolio_uri(pv_df, title="Portfolio Equity"):
    df = pv_df.sort_values("dt").copy()
    plt.figure(figsize=(10,4))
    plt.plot(df["dt"], df["portfolio_value"], linewidth=1.6, label="Portfolio")
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Value")
    plt.grid(True, alpha=0.3); plt.legend()
    return _fig_to_uri()


def _clean_plotly_html(html_text: str) -> str:
    """Hide the range slider bar; tiny cosmetic cleanups without touching forecast.py."""
    if not html_text:
        return html_text
    html_text = html_text.replace('"rangeslider":{"visible":true}', '"rangeslider":{"visible":false}')
    html_text = html_text.replace('"title":{"text":"ds"}', '"title":{"text":"Date"}')
    return html_text

def embed_forecast_html(symbol: str):
    """Try to read plots/forecast_{symbol}.html and components_{symbol}.html and inline them."""
    blocks = []
    f1 = f"plots/forecast_{safe_name(symbol)}.html"
    f2 = f"plots/components_{safe_name(symbol)}.html"
    for fp in (f1, f2):
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                blocks.append(f"<div class='embed-plotly'>\n{_clean_plotly_html(f.read())}\n</div>")
    return "\n".join(blocks)


def run_strategy(prices, strategy_name, **kwargs):
    if strategy_name == "sma":
        sig = signals_sma(prices, fast=kwargs.get("fast", 10), slow=kwargs.get("slow", 30))
        label = f"SMA({kwargs.get('fast',10)},{kwargs.get('slow',30)})"
    elif strategy_name == "prophet_snr":
        sig = signals_prophet_snr(
            prices,
            horizon=kwargs.get("horizon", 5),
            vol_lookback=kwargs.get("vol_lookback", 20),
            entry=kwargs.get("snr_entry", 0.5),
            exit=kwargs.get("snr_exit", 0.1),
        )
        label = f"PROPHET_SNR(h={kwargs.get('horizon',5)})"
    elif strategy_name == "prophet_bands":
        sig = signals_prophet_bands(
            prices,
            alpha=kwargs.get("band_alpha", 0.0),
            take_profit=kwargs.get("take_profit", "mid"),
        )
        label = f"PROPHET_BANDS(α={kwargs.get('band_alpha',0.0)})"
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return sig, label

def make_html_report(portfolio_name, args_dict, per_strategy_results, per_ticker_sections, best_strategy_name):
    os.makedirs("reports", exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    outpath = f"reports/REPORT_{safe_name(portfolio_name)}_{ts}.html"

    def kv(metrics):
        if not metrics:
            return "<div class='card'><i>No metrics</i></div>"
        def _pct(x):
            return "-" if x is None or (isinstance(x,float) and not np.isfinite(x)) else f"{x*100:.2f}%"
        rows = [
            ("Start Value", f"{metrics['start_value']:,.2f}"),
            ("End Value",   f"{metrics['end_value']:,.2f}"),
            ("Total Return", _pct(metrics["total_return"])),
            ("CAGR",         _pct(metrics["cagr"])),
            ("Ann. Vol",     _pct(metrics["annualized_volatility"])),
            ("Sharpe",       "-" if metrics["sharpe_ratio"] is None else f"{metrics['sharpe_ratio']:.2f}"),
            ("Max DD",       _pct(metrics["max_drawdown"])),
            ("Num Days",     f"{metrics['num_days']}"),  # <== unified label
        ]
        tr = "".join([f"<tr><th>{k}</th><td>{v}</td></tr>" for k,v in rows])
        return f"<div class='card'><table class='kv'>{tr}</table></div>"

    # HTML skeleton
    head = f"""<html><head><meta charset='utf-8'><title>DSCI-560 Lab 4 | Portfolio: {portfolio_name} | Strategies Report</title>
    <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Microsoft Yahei", sans-serif; margin: 20px; }}
    h1, h2, h3 {{ margin: 0.6em 0 0.3em; }}
    .meta {{ color: #555; margin-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
    .kv {{ border-collapse: collapse; width: 100%; }}
    .kv th {{ text-align: left; color: #444; width: 160px; padding: 6px 8px; border-bottom: 1px solid #eee; background: #fafafa; }}
    .kv td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
    .trades {{ border-collapse: collapse; width: 100%; }}
    .trades th, .trades td {{ border-bottom: 1px solid #eee; padding: 6px 8px; font-size: 13px; }}
    .section {{ margin: 24px 0; }}
    .embed-plotly {{ border: 1px dashed #ddd; border-radius: 8px; padding: 6px; }}
    img.plot {{ display:block; max-width:100%; height:auto; border:1px solid #eee; border-radius:8px; }}
    .pill {{ display:inline-block; font-size:12px; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; margin-left:6px; }}
    .ok {{ color:#047857; }} .bad {{ color:#b91c1c; }}
    </style>
    </head><body>
    <h1>DSCI-560 Lab 4 | Portfolio: {portfolio_name} | Strategies Report</h1>
    <div class='meta'>Generated at {ts} | Args: <span style='font-family: ui-monospace,monospace; font-size:12px; background:#f8fafc; padding:4px 6px; border-radius:6px;'>{json.dumps(args_dict, ensure_ascii=False)}</span></div>
    """

    # Portfolio summary per strategy
    blocks = [head, "<div class='section'><h2>Portfolio Summary by Strategy</h2><div class='grid'>"]
    for strat_name, info in per_strategy_results.items():
        blocks.append(f"<div class='card'><h3>{strat_name.upper()}</h3>{kv(info['metrics'])}"
                      f"<div style='margin-top:8px'><img class='plot' src='{info['equity_uri']}'/></div></div>")
    blocks.append("</div></div>")

    # Winner banner
    blocks.append(f"<div class='section'><h3>Best strategy on this run: "
                  f"<span class='pill'>{best_strategy_name.upper()}</span></h3></div>")

    # Per-ticker sections (charts + trades + prophet embeds)
    for sec in per_ticker_sections:
        blocks.append("<hr/>")
        blocks.append(f"<div class='section'><h2>{sec['symbol']}</h2>")
        # Strategy charts & mini-metrics per symbol
        blocks.append("<div class='grid'>")
        for card in sec["strategy_cards"]:
            blocks.append(f"<div class='card'><h3>{card['label']}</h3>"
                          f"{kv(card['metrics'])}"
                          f"<div style='margin-top:8px'><img class='plot' src='{card['chart_uri']}'/></div>"
                          f"<div style='margin-top:8px'><b>Trades:</b> {card['num_trades']} | "
                          f"<b class='{ 'ok' if card['profit']>=0 else 'bad' }'>Profit: {card['profit']:,.2f}</b></div>"
                          "</div>")
        blocks.append("</div>")
        # Prophet forecast (interactive) from forecast.py (rangeslider removed here)
        if sec["forecast_embed"]:
            blocks.append("<div class='section'><h3>Prophet Forecast & Components (interactive)</h3>")
            blocks.append(sec["forecast_embed"])
            blocks.append("</div>")
        blocks.append("</div>")  # end ticker section

    blocks.append("</body></html>")
    html = "\n".join(blocks)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)
    return outpath


def run_forecast_sidecar(engine, symbol, period=365, test_rows=0):
    """
    Call your forecast.py minimally to generate the two HTML files.
    We do NOT change forecast.py; we just read its HTML and embed it.
    """
    try:
        from forecast import get_forecast_df, price_forecasting, calculate_best_strategy
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": f"import forecast failed: {e}", "embed": ""}

    # Ensure folders exist
    os.makedirs("plots", exist_ok=True)

    # Pull data & run Prophet once
    df = get_forecast_df(engine, symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "ok": False, "error": "no data", "embed": ""}

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt")
    _ = price_forecasting(df, period)               # forecast.py writes plots/forecast.html & components.html
    # copy to per-symbol filenames so multiple tickers won't override
    if os.path.exists("plots/forecast.html"):
        shutil.copyfile("plots/forecast.html",   f"plots/forecast_{safe_name(symbol)}.html")
    if os.path.exists("plots/components.html"):
        shutil.copyfile("plots/components.html", f"plots/components_{safe_name(symbol)}.html")
    # embed
    embed = embed_forecast_html(symbol)
    return {"symbol": symbol, "ok": True, "embed": embed}


def parse_selection(text, n):
    if text is None or str(text).strip() == "": return list(range(1, n+1))
    s = text.strip().lower().replace(",", " ")
    if s in ("all","a","*"): return list(range(1, n+1))
    picks = set()
    for tok in s.split():
        if tok in ("all","a","*"): return list(range(1, n+1))
        if "-" in tok:
            try:
                a, b = tok.split("-", 1); a, b = int(a), int(b)
                if a > b: a, b = b, a
                for k in range(max(1,a), min(n,b)+1): picks.add(k)
            except: continue
        else:
            try:
                k = int(tok)
                if 1 <= k <= n: picks.add(k)
            except: continue
    return sorted(picks) if picks else list(range(1, n+1))

def main():
    parser = argparse.ArgumentParser(description="Lab 4: portfolio-level backtests (SMA + Prophet SNR/BANDS) with single HTML report")
    # data & period
    parser.add_argument("--portfolio", default=None)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--max_tickers", type=int, default=DEFAULT_MAX)
    # base capital & costs
    parser.add_argument("--initial", type=float, default=DEFAULT_INIT)
    parser.add_argument("--fee_bps", type=float, default=DEFAULT_FEE)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    # strategies
    parser.add_argument("--strategies", default="sma,prophet_snr,prophet_bands")
    parser.add_argument("--fast", type=int, default=10)
    parser.add_argument("--slow", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--snr_entry", type=float, default=0.5)
    parser.add_argument("--snr_exit", type=float, default=0.1)
    parser.add_argument("--band_alpha", type=float, default=0.0)
    parser.add_argument("--take_profit", default="mid", choices=["mid","upper"])
    # forecast sidecar
    parser.add_argument("--forecast_period", type=int, default=365)
    args = parser.parse_args()

    # Preconditions
    if args.fast >= args.slow:
        raise SystemExit(f"--fast ({args.fast}) must be < --slow ({args.slow})")
    os.makedirs("reports", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Connect & load
    name = args.portfolio or input("Enter the portfolio name: ").strip()
    engine = make_engine()
    tickers = fetch_portfolio_tickers(engine, name, args.max_tickers)
    prices = fetch_prices_for_portfolio(engine, name, args.start, args.end)
    prices = prices[prices["ticker_id"].isin(set(tickers["ticker_id"]))].copy()

    # Select subset
    print("\nAvailable tickers in portfolio:")
    for i, (_, row) in enumerate(tickers.iterrows(), start=1):
        print(f"{i}. {row['symbol']}")
    sel_text = input("\nSelect tickers (e.g., 'all' or '3 4 5' or '3-6'; Enter for all):\n> ").strip()
    picks = parse_selection(sel_text, len(tickers))
    chosen = tickers.iloc[[i-1 for i in picks]].reset_index(drop=True)

    # Prepare
    strategy_keys = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    strat_params = dict(
        sma=dict(fast=args.fast, slow=args.slow),
        prophet_snr=dict(horizon=args.horizon, vol_lookback=args.vol_lookback,
                         snr_entry=args.snr_entry, snr_exit=args.snr_exit),
        prophet_bands=dict(band_alpha=args.band_alpha, take_profit=args.take_profit),
    )
    per_ticker_sections = []
    per_strategy_results = {s: {"pv_list": [], "metrics": None, "equity_uri": ""} for s in strategy_keys}
    slot_cash = args.initial / len(chosen)

    print("\nRunning backtests & forecasts...\n")
    for _, r in chosen.iterrows():
        tid, sym = int(r.ticker_id), str(r.symbol)
        sub_px = prices[prices["ticker_id"] == tid].copy().sort_values("dt")
        section = {"symbol": sym, "strategy_cards": [], "forecast_embed": ""}

        # forecast sidecar (produces forecast_{sym}.html & components_{sym}.html; we embed after)
        fc_info = run_forecast_sidecar(engine, sym, period=args.forecast_period)
        section["forecast_embed"] = fc_info.get("embed", "")

        # run each strategy on this ticker
        for skey in strategy_keys:
            sig, label = run_strategy(sub_px, skey, **strat_params.get(skey, {}))
            pv_df, tr_df = backtest_one_slot(sub_px, sig[sig["ticker_id"]==tid], sym, slot_cash0=slot_cash, fee_bps=args.fee_bps)
            metr = compute_metrics(pv_df, rf=args.rf)
            chart_uri = plot_single_ticker_uri(sub_px, sig[sig["ticker_id"]==tid], tr_df, sym, label)

            # Profit for this ticker under this strategy
            profit = pv_df["portfolio_value"].iloc[-1] - pv_df["portfolio_value"].iloc[0]
            print(f"[{sym}][{skey.upper()}] Trades: {len(tr_df):d} | Profit: {profit:,.2f} | Total Return: {metr['total_return']*100:.2f}%")

            section["strategy_cards"].append({
                "label": skey.upper(),
                "metrics": metr,
                "chart_uri": chart_uri,
                "num_trades": int(len(tr_df)),
                "profit": float(profit),
            })
            per_strategy_results[skey]["pv_list"].append(pv_df)

        per_ticker_sections.append(section)

    # Aggregate at portfolio level for each strategy
    for skey, info in per_strategy_results.items():
        pv_port = aggregate_portfolio(info["pv_list"])
        info["metrics"] = compute_metrics(pv_port, rf=args.rf)
        info["equity_uri"] = plot_portfolio_uri(pv_port, title=f"Portfolio Equity | Strategy: {skey.upper()}")

    # Pick best strategy by ending value
    best_name = max(per_strategy_results.items(), key=lambda kv: kv[1]["metrics"]["end_value"])[0]

    # Final portfolio-level summary line (quiet & clean)
    print("\n=== Summary by Strategy (Portfolio) ===")
    for skey, info in per_strategy_results.items():
        m = info["metrics"]
        print(f"{skey.upper()}: End={m['end_value']:,.2f} | Profit={m['end_value']-m['start_value']:,.2f} | "
              f"Sharpe={ '-' if m['sharpe_ratio'] is None else f'{m['sharpe_ratio']:.2f}'} | "
              f"MaxDD={m['max_drawdown']*100:.2f}% | Num Days={m['num_days']}")

    # Build one consolidated HTML report
    args_dict = dict(
        portfolio=name, start=args.start, end=args.end,
        strategies=",".join(strategy_keys), horizon=args.horizon,
        vol_lookback=args.vol_lookback, snr_entry=args.snr_entry, snr_exit=args.snr_exit,
        band_alpha=args.band_alpha, take_profit=args.take_profit,
        initial=args.initial, fee_bps=args.fee_bps, rf=args.rf
    )
    report_path = make_html_report(name, args_dict, per_strategy_results, per_ticker_sections, best_name)
    print(f"\n🏁 Report saved to: {report_path}")

if __name__ == "__main__":
    main()