import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import altair as alt
import datetime

# ===== Настройки страницы =====
st.set_page_config(page_title="WFM: Прогноз и расчёт сотрудников", layout="wide")

# ===== Стилизация =====
st.markdown("""
<style>
    .stDataFrame { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ===== Перевод дней недели =====
weekday_map = {
    "Monday": "Понедельник",
    "Tuesday": "Вторник",
    "Wednesday": "Среда",
    "Thursday": "Четверг",
    "Friday": "Пятница",
    "Saturday": "Суббота",
    "Sunday": "Воскресенье"
}

# ===== Рабочее время =====
WORK_START = 9
WORK_END_EXCL = 20
WORK_HOURS = list(range(WORK_START, WORK_END_EXCL))
SLOTS_PER_DAY = len(WORK_HOURS)
PEAK_HOURS = [9, 10]

# ===== Сайдбар =====
st.sidebar.header("⚙️ Параметры")
horizon_days = st.sidebar.slider("Горизонт прогноза (дней)", 1, 31, 7, 1)
efficiency = st.sidebar.number_input("Эффективность (доля)", 0.10, 1.0, 0.85, 0.01, format="%.2f")

# AHT ручной ввод в формате мм:сс
aht_str = st.sidebar.text_input("AHT (мм:сс)", "03:00")
try:
    mm, ss = map(int, aht_str.split(":"))
    aht_min = mm + ss/60
except:
    st.sidebar.error("Введите время в формате мм:сс")
    aht_min = 3.0

# Дата начала прогноза
default_start = datetime.date(datetime.date.today().year, 8, 15)
start_date = st.sidebar.date_input("📅 Дата начала прогноза", default_start)

# Дата для сравнения (факт vs прогноз)
compare_date = st.sidebar.date_input("📅 Дата для сравнения (Факт vs Прогноз)", None)

# ===== Источник данных (Google Sheets) =====
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRUo-9C-CLiNCnB9CfTXphTbrPknJjjrgogM3dqruy66S6_oKyoeiSUHe0hPMClIGmcmyqLwsYgyupG/pub?gid=1587628005&single=true&output=csv"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True, errors="coerce")
    df["Час"] = pd.to_numeric(df["Час"], errors="coerce")
    df["Звонки"] = pd.to_numeric(df["Звонки"], errors="coerce")

    def parse_aht(val):
        if pd.isna(val): return np.nan
        try:
            parts = str(val).split(":")
            if len(parts) == 2:
                m, s = parts; return int(m) + int(s)/60
            return float(val)
        except:
            return np.nan

    df["Среднее время (мин)"] = df["Среднее время (мин)"].apply(parse_aht)
    df = df.dropna(subset=["Дата", "Час", "Звонки"])
    df["Час"] = df["Час"].astype(int)
    df["Datetime"] = pd.to_datetime(df["Дата"].dt.strftime("%Y-%m-%d") + " " + df["Час"].astype(str) + ":00:00")
    df = df.set_index("Datetime").sort_index()
    df = df[df["Час"].isin(WORK_HOURS)]
    if "День недели" not in df.columns:
        df["День недели"] = df.index.day_name().map(weekday_map)
    return df

df = load_data()

# ===== Утилиты =====
def format_aht(minutes: float):
    if pd.isna(minutes): return "—"
    m = int(minutes)
    s = int(round((minutes - m) * 60))
    return f"{m:02d}:{s:02d}"

def dates_where_month_ahead_is_weekend(dates) -> set:
    dt = pd.to_datetime(dates)
    uniq = sorted(set(dt.date if isinstance(dt, pd.DatetimeIndex) else dt.dt.date))
    res = set()
    for d in pd.to_datetime(uniq):
        future = d + relativedelta(months=+1)
        if future.weekday() in (5, 6):
            res.add(d.date())
    return res

def generate_future_business_hours(start_exclusive: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    out, cur = [], start_exclusive
    for _ in range(periods):
        nxt = cur + pd.Timedelta(hours=1)
        if nxt.hour >= WORK_END_EXCL or nxt.hour < WORK_START:
            nxt = (pd.Timestamp(cur.date()) + pd.Timedelta(days=1)).replace(hour=WORK_START)
        out.append(nxt)
        cur = nxt
    return pd.DatetimeIndex(out)

def trimmed_mean(series, lower=0.1, upper=0.9):
    if series.empty: return np.nan
    q_low, q_high = series.quantile([lower, upper])
    return series[(series >= q_low) & (series <= q_high)].mean()

# ===== Функция прогноза =====
def build_forecast(base_df, horizon_days, efficiency, aht_min, ref_date=None):
    today = pd.Timestamp.today().normalize()
    if ref_date:
        ref_date = pd.Timestamp(ref_date)

    if ref_date and ref_date < today:
        start_forecast = ref_date.replace(hour=WORK_START)
    else:
        start_forecast = (today + pd.Timedelta(days=1)).replace(hour=WORK_START)

    df_win = base_df[base_df.index >= pd.Timestamp(start_date)]
    if df_win.empty:
        return pd.DataFrame()

    df_win["weekday"] = df_win.index.weekday
    df_win["hour"] = df_win["Час"]
    pivot_med = df_win.pivot_table(index="hour", columns="weekday", values="Звонки", aggfunc="median")

    opening_days_hist = dates_where_month_ahead_is_weekend(df_win.index)
    is_open_hist = pd.Series(df_win.index.date, index=df_win.index).isin(opening_days_hist)
    is_weekday_hist = df_win.index.weekday < 5

    hour_coefs = {}
    for h in PEAK_HOURS:
        open_vals = df_win[(df_win["hour"] == h) & is_open_hist]["Звонки"]
        base_vals = df_win[(df_win["hour"] == h) & (~is_open_hist) & is_weekday_hist]["Звонки"]
        if len(open_vals) >= 2 and len(base_vals) >= 2 and base_vals.median() > 0:
            coef = trimmed_mean(open_vals) / trimmed_mean(base_vals)
            coef = float(np.clip(coef, 2.0, 5.0))
        else:
            coef = 2.0
        hour_coefs[h] = coef

    steps = horizon_days * SLOTS_PER_DAY
    future_idx = generate_future_business_hours(start_forecast - pd.Timedelta(hours=1), steps)

    base_vals = []
    for t in future_idx:
        w, h = t.weekday(), t.hour
        v = pivot_med.loc[h, w] if (h in pivot_med.index and w in pivot_med.columns) else np.nan
        if pd.isna(v): v = df_win.groupby("hour")["Звонки"].median().get(h, df_win["Звонки"].median())
        base_vals.append(v)

    out = pd.DataFrame({"Datetime": future_idx, "База": base_vals})
    out["Date"] = out["Datetime"].dt.strftime("%d-%m-%Y")
    out["Hour"] = out["Datetime"].dt.hour
    out["День недели"] = out["Datetime"].dt.day_name().map(weekday_map)
    out["Открытие?"] = out["Datetime"].dt.date.isin(dates_where_month_ahead_is_weekend(out["Datetime"]))

    out["Прогноз звонков"] = out["База"]
    for h in PEAK_HOURS:
        mask = (out["Открытие?"]) & (out["Hour"] == h)
        out.loc[mask, "Прогноз звонков"] *= hour_coefs[h]

    hist_max = df_win["Звонки"].max()
    out["Прогноз звонков"] = np.minimum(out["Прогноз звонков"], hist_max * 2.5)
    out["Прогноз звонков"] = np.maximum(np.round(out["Прогноз звонков"]), 0).astype(int)

    out["Сотрудники"] = np.ceil((out["Прогноз звонков"] * aht_min) / (60.0 * efficiency)).astype(int)

    return out.sort_values("Datetime").reset_index(drop=True)

# ===== MAIN =====
st.markdown("<h1>📞 WFM-модуль</h1>", unsafe_allow_html=True)

# История
with st.expander("📅 История по дням"):
    daily = df.groupby(df["Дата"].dt.date).agg({"Звонки": "sum", "Среднее время (мин)": "mean"}).reset_index()
    daily["Среднее время (AHT)"] = daily["Среднее время (мин)"].apply(format_aht)
    daily["Месяц"] = pd.to_datetime(daily["Дата"]).dt.strftime("%B %Y")
    daily["День недели"] = pd.to_datetime(daily["Дата"]).dt.day_name().map(weekday_map)
    sel_month = st.selectbox("Выберите месяц", sorted(daily["Месяц"].unique()))
    daily_month = daily[daily["Месяц"] == sel_month]
    st.dataframe(daily_month[["Дата","День недели","Звонки","Среднее время (AHT)"]].reset_index(drop=True),
                 use_container_width=True, height=400)

# Прогноз
forecast = build_forecast(df, horizon_days, efficiency, aht_min)

st.header("🔮 Прогноз по дням")
for d, grp in forecast.groupby("Date"):
    with st.expander(f"{d} ({grp['День недели'].iloc[0]})"):
        total_calls = grp["Прогноз звонков"].sum()
        st.markdown(f"**Всего звонков: {total_calls}**")
        st.dataframe(grp[["Hour", "Прогноз звонков", "Сотрудники"]].reset_index(drop=True),
                     use_container_width=True, height=400)

# Сравнение факт vs прогноз
if compare_date:
    compare_date = pd.Timestamp(compare_date)
    st.header(f"📊 Сравнение факта и прогноза: {compare_date.date()}")
    fact = df[df["Дата"].dt.date == compare_date.date()]
    pred = build_forecast(df, 1, efficiency, aht_min, ref_date=compare_date)
    if not fact.empty and not pred.empty:
        merged = pd.merge(fact.reset_index(), pred, left_on="Час", right_on="Hour", how="outer")
        merged["Среднее время (AHT)"] = merged["Среднее время (мин)"].apply(format_aht)
        merged["% отклонения"] = ((merged["Звонки"] - merged["Прогноз звонков"]) / merged["Прогноз звонков"] * 100).round(1)
        merged = merged[["Час", "Звонки", "Прогноз звонков", "% отклонения", "Среднее время (AHT)", "Сотрудники"]]
        st.dataframe(merged.reset_index(drop=True), use_container_width=True, height=400)
    else:
        st.warning("Нет данных для сравнения.")
