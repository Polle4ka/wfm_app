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

# ===== Заголовок =====
st.markdown("<h1 style='margin-bottom:0'>📞 WFM-модуль</h1><p style='color:#666;margin-top:4px'>Прогноз нагрузки и расчёт штата для контакт-центра</p>", unsafe_allow_html=True)

# ===== Рабочее время =====
WORK_START = 9
WORK_END_EXCL = 20            # конец НЕ включительно
WORK_HOURS = list(range(WORK_START, WORK_END_EXCL))  # 9..19
SLOTS_PER_DAY = len(WORK_HOURS)
PEAK_HOURS = [9, 10]

# ===== Сайдбар: параметры =====
st.sidebar.header("⚙️ Параметры прогноза")
horizon_days = st.sidebar.slider("Горизонт прогноза (дней)", 1, 31, 7, 1)
efficiency = st.sidebar.number_input("Эффективность (доля)", 0.10, 1.0, 0.85, 0.01, format="%.2f")

# Календарь для выбора даты начала анализа
default_start = datetime.date(datetime.date.today().year, 8, 15)  # 15 августа текущего года
start_date = st.sidebar.date_input("📅 Дата начала анализа", default_start)

uploaded = st.sidebar.file_uploader("CSV: Дата, Час, [День недели], Звонки, Среднее время (мин)", type=["csv"])

# ===== Утилиты =====
def parse_aht(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    try:
        parts = str(val).strip().split(":")
        if len(parts) == 2:
            m, s = parts; return int(m) + int(s)/60
        if len(parts) == 3:
            h, m, s = parts; return int(h)*60 + int(m) + int(s)/60
        return float(val)
    except Exception:
        return np.nan

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
    """ Усечённое среднее """
    if series.empty:
        return np.nan
    q_low, q_high = series.quantile([lower, upper])
    return series[(series >= q_low) & (series <= q_high)].mean()

# ===== MAIN =====
if uploaded is None:
    st.info("Загрузите CSV. Обязательные колонки: Дата, Час, Звонки, Среднее время (мин).")
else:
    df = pd.read_csv(uploaded)

    # === Очистка ===
    df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True, errors="coerce")
    df["Час"] = pd.to_numeric(df["Час"], errors="coerce")
    df["Звонки"] = pd.to_numeric(df["Звонки"], errors="coerce")
    df["Среднее время (мин)"] = df["Среднее время (мин)"].apply(parse_aht)
    df = df.dropna(subset=["Дата", "Час", "Звонки"])
    df["Час"] = df["Час"].astype(int)
    df["Datetime"] = pd.to_datetime(df["Дата"].dt.strftime("%Y-%m-%d") + " " + df["Час"].astype(str) + ":00:00")
    df = df.set_index("Datetime").sort_index()
    df = df[df["Час"].isin(WORK_HOURS)]
    if "День недели" not in df.columns:
        df["День недели"] = df.index.day_name(locale="Russian")

    # ===== История по дням =====
    with st.expander("📅 История по дням"):
        daily = df.groupby(df["Дата"].dt.date).agg({"Звонки": "sum", "Среднее время (мин)": "mean"}).reset_index()
        daily["Месяц"] = pd.to_datetime(daily["Дата"]).dt.strftime("%B %Y")
        daily["День недели"] = pd.to_datetime(daily["Дата"]).dt.day_name(locale="Russian")
        months = daily["Месяц"].unique().tolist()
        sel_month = st.selectbox("Выберите месяц", months)
        daily_month = daily[daily["Месяц"] == sel_month]
        st.dataframe(daily_month, use_container_width=True, height=350)
        chart_daily = alt.Chart(daily_month).mark_bar(color="#4C78A8").encode(
            x="Дата:T", y="Звонки:Q", tooltip=["Дата","День недели","Звонки"]
        )
        st.altair_chart(chart_daily, use_container_width=True)

    # ===== Фильтр по дате начала анализа =====
    df_win = df[df.index >= pd.Timestamp(start_date)]
    if df_win.empty:
        st.error("Нет данных после выбранной даты.")
        st.stop()

    # === Профиль по часам ===
    df_win["weekday"] = df_win.index.weekday
    df_win["hour"] = df_win["Час"]
    pivot_med = df_win.pivot_table(index="hour", columns="weekday", values="Звонки", aggfunc="median")

    # === Коэффициенты открытия ===
    opening_days_hist = dates_where_month_ahead_is_weekend(df_win.index)
    is_open_hist = pd.Series(df_win.index.date, index=df_win.index).isin(opening_days_hist)
    is_weekday_hist = df_win.index.weekday < 5
    hour_coefs = {}
    for h in PEAK_HOURS:
        open_vals = df_win[(df_win["hour"] == h) & is_open_hist]["Звонки"]
        base_vals = df_win[(df_win["hour"] == h) & (~is_open_hist) & is_weekday_hist]["Звонки"]
        if len(open_vals) >= 2 and len(base_vals) >= 2 and base_vals.median() > 0:
            coef = trimmed_mean(open_vals) / trimmed_mean(base_vals)
            coef = float(np.clip(coef, 2.0, 5.0))  # ограничиваем
        else:
            coef = 2.0
        hour_coefs[h] = coef

    # === Прогноз ===
    today = pd.Timestamp.today().normalize()
    start_forecast = (today + pd.Timedelta(days=1)).replace(hour=WORK_START)
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
    out["День недели"] = out["Datetime"].dt.day_name(locale="Russian")
    out["Открытие?"] = out["Datetime"].dt.date.isin(dates_where_month_ahead_is_weekend(out["Datetime"]))

    out["Прогноз звонков"] = out["База"]
    for h in PEAK_HOURS:
        mask = (out["Открытие?"]) & (out["Hour"] == h)
        out.loc[mask, "Прогноз звонков"] *= hour_coefs[h]
    out["Прогноз звонков"] = np.maximum(np.round(out["Прогноз звонков"]), 0).astype(int)

    default_aht = df_win["Среднее время (мин)"].median() if not np.isnan(df_win["Среднее время (мин)"].median()) else 3.0
    aht_min = st.sidebar.number_input("AHT (мин)", 0.1, 60.0, round(default_aht,2), 0.1)
    out["Сотрудники"] = np.ceil((out["Прогноз звонков"] * aht_min) / (60.0 * efficiency)).astype(int)

    # === Показать рассчитанные коэффициенты ===
    st.markdown("### 📊 Коэффициенты для пиковых часов (дни открытия записи)")
    st.write(f"09:00 → **{hour_coefs.get(9, 2.0):.2f}×**")
    st.write(f"10:00 → **{hour_coefs.get(10, 2.0):.2f}×**")

    # ===== Прогноз по дням =====
    for d, grp in out.groupby("Date"):
        with st.expander(f"🔮 Прогноз на {d} ({grp['День недели'].iloc[0]})"):
            st.dataframe(grp[["Hour","Прогноз звонков","Сотрудники"]], use_container_width=True)
            chart_day = alt.Chart(grp).mark_line(point=True, color="#4C78A8").encode(
                x=alt.X("Hour:O", title="Час"),
                y=alt.Y("Прогноз звонков:Q", title="Звонки"),
                tooltip=["Hour","Прогноз звонков","Сотрудники"]
            )
            staff_day = alt.Chart(grp).mark_bar(opacity=0.3, color="#72B7B2").encode(
                x="Hour:O", y="Сотрудники:Q"
            )
            st.altair_chart(chart_day + staff_day, use_container_width=True)

    # ===== Скачать =====
    st.download_button("⬇️ Скачать прогноз (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name="wfm_result.csv", mime="text/csv")
