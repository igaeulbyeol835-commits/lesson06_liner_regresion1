import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025   # 수업의 기준 기간: 이 해까지만 사용
MIN_DAYS = 300     # 관측일이 이보다 적은 해는 제외

st.set_page_config(page_title="기온 예측기", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중입니다.")
def load_annual() -> pd.DataFrame:
    """일별 자료 -> 연도별 평균기온(관측일수 포함)."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = df.columns.str.replace("﻿", "", regex=False).str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    annual = (
        df.groupby("연도")["평균기온"]
        .agg(관측일수="count", 연평균기온="mean")
        .reset_index()
    )
    return annual[annual["관측일수"] >= MIN_DAYS].reset_index(drop=True)


st.title("기온 예측기")
st.caption(
    f"서울 일별 기온 자료에서 연평균기온을 구하고, 연도에 따른 변화를 직선으로 "
    f"근사하여 선택한 해의 기온을 예측합니다. "
    f"({LAST_YEAR}년까지의 자료만 사용, 관측일 {MIN_DAYS}일 미만인 해 제외)"
)

try:
    annual = load_annual()
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.stop()

if len(annual) < 3:
    st.error("조건을 만족하는 연도가 너무 적어 회귀 직선을 만들 수 없습니다.")
    st.stop()

years = annual["연도"].to_numpy(dtype=float)
temps = annual["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(years, temps, 1)
corr = float(np.corrcoef(years, temps)[0, 1])

n_years = len(annual)
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())

# 직선 정보
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{corr:.3f}")

sign = "+" if intercept >= 0 else "-"
st.markdown(
    f"회귀 직선: **연평균기온 = {slope:.4f} × 연도 {sign} {abs(intercept):.2f}**"
    f" (10년당 약 {slope * 10:+.2f}℃)"
)

# 연도 선택과 예상 기온
st.subheader("예상 기온")
year = st.slider("연도를 선택하십시오.", 1900, 2100, 2026, 1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0 1.6rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">{year}년 서울의 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:700; line-height:1.2;">{pred:.2f}℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if year < start_year or year > end_year:
    st.caption(
        f"선택한 연도는 직선을 만든 기간({start_year}~{end_year}년) 밖에 있으므로, "
        "이 값은 직선을 연장한 외삽 결과입니다."
    )

# 산점도 + 회귀 직선
x_lo = min(start_year, year)
x_hi = max(end_year, year)
line_x = np.array([x_lo, x_hi], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f}℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1)),
        hovertemplate=f"{year}년 예상<br>%{{y:.2f}}℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (r = {corr:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=80, b=40),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        annual.rename(columns={"연평균기온": "연평균기온(℃)"}).round({"연평균기온(℃)": 2}),
        hide_index=True,
    )
