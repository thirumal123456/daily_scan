import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Institutional Sector Rotation Dashboard",
    layout="wide"
)

st.title("📊 Institutional Sector Rotation Dashboard")

# =========================================================
# ETF MAP
# FIX #1: NIFTY50 removed — was causing self-comparison
# in RS matrix (RS always = 0). Benchmark is NIFTYBEES
# separately.
# =========================================================

ETF_MAP = {
    "BANK":     "BANKBEES.NS",
    "IT":       "ITBEES.NS",
    "AUTO":     "AUTOBEES.NS",
    "PHARMA":   "PHARMABEES.NS",
    "PSU BANK": "PSUBNKBEES.NS",
    "METAL":    "METALIETF.NS",
    "FMCG":     "FMCGIETF.NS",
    "CPSE":     "CPSEETF.NS",
}

# =========================================================
# LOOKBACKS
# =========================================================

LOOKBACKS = {
    "1D":  1,
    "3D":  3,
    "5D":  5,
    "15D": 15,
    "30D": 30,
    "60D": 60,
    "90D": 90,
}

MIN_BARS = 60  # minimum rows needed for reliable indicators

# =========================================================
# DATA LOADER
# FIX #2: bare except -> except Exception with st.warning
# FIX #5: validate shape; extract first column explicitly
# FIX #9: ttl=300 (5 min) for near-live data
# =========================================================

@st.cache_data(ttl=300)
def load_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df.empty:
            st.warning(f"⚠️ No data returned for {symbol}")
            return pd.DataFrame()

        # Flatten MultiIndex columns (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Validate required columns
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            st.warning(f"⚠️ Missing columns for {symbol}: {required - set(df.columns)}")
            return pd.DataFrame()

        # Ensure all price columns are 1-D Series (not DataFrame)
        # FIX #5: explicit column selection instead of .squeeze()
        for col in required:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        return df

    except Exception as e:
        st.warning(f"⚠️ Failed to load {symbol}: {e}")
        return pd.DataFrame()


# =========================================================
# RETURN CALCULATION
#
# ROOT CAUSE OF WRONG NUMBERS:
# The old code used iloc[-(days+1)] which counts TRADING days.
# "30D" in the label means 30 CALENDAR days to any user.
# But iloc[-31] goes back 30 TRADING days ≈ 6 calendar weeks.
# At 90D this error can exceed 10 percentage points.
#
# FIX: use the DatetimeIndex to find the actual calendar date
# N days ago, then pick the nearest prior trading session.
# This matches what NSE/BSE, TradingView, and Bloomberg show.
# =========================================================

def calculate_return(close: pd.Series, days: int) -> float:
    """
    Return % change over the last `days` CALENDAR days.
    Finds the nearest trading session on-or-before (today - days).
    """
    try:
        if close.empty or len(close) < 2:
            return np.nan

        last_date = close.index[-1]
        target    = last_date - pd.DateOffset(days=days)

        # All sessions on or before the target calendar date
        prior = close[close.index <= target]

        if prior.empty:
            return np.nan

        current  = float(close.iloc[-1])
        previous = float(prior.iloc[-1])

        return round(((current / previous) - 1) * 100, 2)

    except Exception:
        return np.nan


# =========================================================
# COLOR SCALE
# =========================================================

def heat_color(val) -> str:
    try:
        v = float(val)
        if v >= 10:  return "background-color: darkgreen; color: white"
        if v >= 5:   return "background-color: green; color: white"
        if v > 0:    return "background-color: lightgreen; color: black"
        if v <= -10: return "background-color: darkred; color: white"
        if v <= -5:  return "background-color: red; color: white"
        if v < 0:    return "background-color: pink; color: black"
        return ""
    except Exception:
        return ""


# FIX #10: apply signal_color only to string columns
def signal_color(val) -> str:
    mapping = {
        "BUY":     "background-color: green; color: white",
        "SELL":    "background-color: red; color: white",
        "HOLD":    "background-color: orange; color: black",
        "Bullish": "background-color: green; color: white",
        "Bearish": "background-color: red; color: white",
        "YES":     "background-color: #1a7a1a; color: white",
        "NO":      "background-color: #555; color: white",
    }
    return mapping.get(str(val), "")


# =========================================================
# LOAD BENCHMARK  (separate from sector ETFs)
# FIX #2: validate before proceeding
# =========================================================

benchmark_df    = load_data("NIFTYBEES.NS")
benchmark_valid = not benchmark_df.empty

if not benchmark_valid:
    st.error("🚨 Benchmark NIFTYBEES.NS failed to load. RS matrix unavailable.")

benchmark_close = benchmark_df["Close"] if benchmark_valid else pd.Series(dtype=float)

# =========================================================
# LOAD SECTOR ETF DATA
# =========================================================

etf_data: dict[str, pd.DataFrame] = {}

for sector, symbol in ETF_MAP.items():
    df = load_data(symbol)
    if not df.empty:
        etf_data[sector] = df

if not etf_data:
    st.error("🚨 No ETF data loaded. Check your connection or ticker symbols.")
    st.stop()

# =========================================================
# PRICE PERFORMANCE MATRIX
# =========================================================

st.subheader("📊 Sector ETF Price Performance Matrix")

price_rows = []

for sector, df in etf_data.items():
    close = df["Close"]
    row = {"Sector": sector}
    for label, days in LOOKBACKS.items():
        row[label] = calculate_return(close, days)
    price_rows.append(row)

price_df = pd.DataFrame(price_rows)

styled_price_df = price_df.style.map(
    heat_color, subset=list(LOOKBACKS.keys())
)

st.dataframe(styled_price_df, use_container_width=True)

# =========================================================
# PRICE HEATMAP
# =========================================================

st.subheader("🔥 ETF Performance Heatmap")

heatmap_df = price_df.set_index("Sector")

fig_heatmap = px.imshow(
    heatmap_df,
    text_auto=True,
    aspect="auto",
    zmin=-10,
    zmax=10,
    color_continuous_scale=[[0, "red"], [0.5, "white"], [1, "green"]],
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# =========================================================
# RELATIVE STRENGTH MATRIX
# =========================================================

st.subheader("⚡ Relative Strength vs NIFTY50 (NIFTYBEES)")

if benchmark_valid:
    benchmark_returns = {
        label: calculate_return(benchmark_close, days)
        for label, days in LOOKBACKS.items()
    }

    rs_rows = []

    for sector, df in etf_data.items():
        close = df["Close"]
        row = {"Sector": sector}
        for label, days in LOOKBACKS.items():
            etf_ret   = calculate_return(close, days)
            bench_ret = benchmark_returns[label]
            row[label] = round(etf_ret - bench_ret, 2) if not pd.isna(etf_ret) else np.nan
        rs_rows.append(row)

    rs_df = pd.DataFrame(rs_rows)

    styled_rs_df = rs_df.style.map(
        heat_color, subset=list(LOOKBACKS.keys())
    )

    st.dataframe(styled_rs_df, use_container_width=True)

    st.subheader("🔥 Relative Strength Heatmap")

    rs_heatmap = rs_df.set_index("Sector")

    fig_rs = px.imshow(
        rs_heatmap,
        text_auto=True,
        aspect="auto",
        zmin=-10,
        zmax=10,
        color_continuous_scale=[[0, "red"], [0.5, "white"], [1, "green"]],
    )

    st.plotly_chart(fig_rs, use_container_width=True)

else:
    st.warning("RS matrix skipped — benchmark unavailable.")

# =========================================================
# ETF TREND COMPARISON
# FIX #7: align on DATE index, not positional .values
# =========================================================

st.subheader("📈 ETF Relative Trend Comparison")

chart_choice = st.selectbox(
    "Select Comparison Window",
    ["15D", "30D", "60D", "90D"],
    index=1,
)

chart_days = LOOKBACKS[chart_choice]

trend_series = {}

for sector, df in etf_data.items():
    try:
        close = df["Close"].tail(chart_days + 1)
        if len(close) < 2:
            continue
        # FIX #7: normalize and keep DatetimeIndex — join on date
        normalized = (close / close.iloc[0]) * 100
        trend_series[sector] = normalized
    except Exception:
        pass

if trend_series:
    # Outer join preserves all dates; NaN where an ETF had no trade
    trend_df = pd.concat(trend_series, axis=1)
    trend_df.index = pd.to_datetime(trend_df.index)
    trend_df = trend_df.sort_index()

    fig_trend = px.line(
        trend_df,
        title=f"ETF Relative Trend ({chart_choice})",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.warning("No trend data available.")

# =========================================================
# ETF SIGNAL MATRIX
# FIX #8: BUY threshold lowered to >= 8 (max score = 12)
# FIX #10: signal_color applied only to string columns
# FIX #11: skip ETFs with < MIN_BARS rows
# =========================================================

st.subheader("🚦 Sector ETF Trading Signal Matrix")

signal_rows = []

for sector, df in etf_data.items():
    try:
        # FIX #11: guard for minimum data length
        if len(df) < MIN_BARS:
            st.warning(f"{sector}: insufficient data ({len(df)} rows < {MIN_BARS}), skipping signals.")
            continue

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        current_price = float(close.iloc[-1])

        # Indicators
        rsi_val      = float(RSIIndicator(close).rsi().iloc[-1])
        macd_obj     = MACD(close)
        macd_val     = float(macd_obj.macd().iloc[-1])
        macd_sig_val = float(macd_obj.macd_signal().iloc[-1])
        adx_val      = float(ADXIndicator(high=high, low=low, close=close).adx().iloc[-1])

        ema9   = float(close.ewm(span=9,   adjust=False).mean().iloc[-1])
        ema21  = float(close.ewm(span=21,  adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        atr_val    = float(AverageTrueRange(high=high, low=low, close=close).average_true_range().iloc[-1])
        support    = float(low.tail(20).min())
        resistance = float(high.tail(20).max())

        breakout        = "YES" if current_price > resistance else "NO"
        avg_vol         = float(volume.tail(20).mean())
        volume_breakout = "YES" if float(volume.iloc[-1]) > avg_vol * 1.5 else "NO"

        # Score  (max = 12)
        score = 0
        if rsi_val > 60:          score += 2
        if macd_val > macd_sig_val: score += 2
        if adx_val > 25:          score += 2
        if ema9 > ema21:          score += 1
        if ema21 > ema50:         score += 1
        if current_price > ema200: score += 1
        if breakout == "YES":     score += 2
        if volume_breakout == "YES": score += 1

        # FIX #8: calibrated thresholds
        # BUY  >= 8  (was 10 — almost never triggered)
        # HOLD >= 5  (was 6)
        # SELL < 5
        if score >= 8:
            final_signal = "BUY"
        elif score >= 5:
            final_signal = "HOLD"
        else:
            final_signal = "SELL"

        signal_rows.append({
            "Sector":         sector,
            "Signal":         final_signal,
            "Score":          score,
            "Price":          round(current_price, 2),
            "RSI":            round(rsi_val, 2),
            "ADX":            round(adx_val, 2),
            "MACD":           "Bullish" if macd_val > macd_sig_val else "Bearish",
            "EMA9":           round(ema9, 2),
            "EMA21":          round(ema21, 2),
            "EMA50":          round(ema50, 2),
            "EMA200":         round(ema200, 2),
            "Support":        round(support, 2),
            "Resistance":     round(resistance, 2),
            "ATR":            round(atr_val, 2),
            "Breakout":       breakout,
            "Vol Breakout":   volume_breakout,
        })

    except Exception as e:
        st.warning(f"{sector}: {e}")

if signal_rows:
    signal_df = pd.DataFrame(signal_rows).sort_values("Score", ascending=False)

    # FIX #10: apply signal_color ONLY to categorical string columns
    string_cols = ["Signal", "MACD", "Breakout", "Vol Breakout"]

    styled_signal = signal_df.style.map(
        signal_color, subset=string_cols
    )

    st.dataframe(styled_signal, use_container_width=True)
else:
    st.warning("No signals generated.")

# =========================================================
# RRG (Relative Rotation Graph)
# FIX #6: proper smoothed RS Ratio using EMA10
#         momentum = smoothed_rs[-1] - smoothed_rs[-2]
#         guard for < 15 rows
# =========================================================

st.subheader("🌀 Relative Rotation Graph")

rrg_rows = []

if benchmark_valid:
    for sector, df in etf_data.items():
        try:
            close = df["Close"]

            min_len = min(len(close), len(benchmark_close))

            # FIX #6: need enough rows for EMA10 to be meaningful
            if min_len < 15:
                continue

            c_aligned = close.tail(min_len).reset_index(drop=True)
            b_aligned = benchmark_close.tail(min_len).reset_index(drop=True)

            rs_ratio_raw = (c_aligned / b_aligned) * 100

            # Smooth with EMA-10 (closer to original RRG methodology)
            rs_smoothed = rs_ratio_raw.ewm(span=10, adjust=False).mean()

            rs_value = float(rs_smoothed.iloc[-1])
            # FIX #6: momentum = 1-period change on smoothed ratio
            momentum = float(rs_smoothed.iloc[-1] - rs_smoothed.iloc[-2])

            if rs_value >= 100 and momentum >= 0:
                quadrant = "Leading"
            elif rs_value >= 100 and momentum < 0:
                quadrant = "Weakening"
            elif rs_value < 100 and momentum < 0:
                quadrant = "Lagging"
            else:
                quadrant = "Improving"

            rrg_rows.append({
                "Sector":   sector,
                "RS Ratio": round(rs_value, 2),
                "Momentum": round(momentum, 2),
                "Quadrant": quadrant,
            })

        except Exception as e:
            st.warning(f"RRG {sector}: {e}")

if rrg_rows:
    rrg_df = pd.DataFrame(rrg_rows)
    rrg_df["Bubble Size"] = rrg_df["Momentum"].abs().clip(lower=0.1)

    fig_rrg = px.scatter(
        rrg_df,
        x="RS Ratio",
        y="Momentum",
        color="Quadrant",
        size="Bubble Size",
        text="Sector",
        color_discrete_map={
            "Leading":   "green",
            "Weakening": "orange",
            "Lagging":   "red",
            "Improving": "steelblue",
        },
    )

    fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray")
    fig_rrg.add_hline(y=0,   line_dash="dash", line_color="gray")

    fig_rrg.update_traces(textposition="top center")

    st.plotly_chart(fig_rrg, use_container_width=True)
else:
    st.warning("RRG skipped — insufficient data or benchmark unavailable.")

st.caption("Institutional Sector Rotation Dashboard — fixed build")
