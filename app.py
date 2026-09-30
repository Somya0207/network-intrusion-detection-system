import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import plotly.graph_objects as go
import joblib
from streamlit_autorefresh import st_autorefresh
import shap

st.set_page_config(page_title="NIDS Live Dashboard", layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }
        div.stButton > button {
            font-size: 13px;
            padding: 0.25rem 0.75rem;
        }
        section[data-testid="stSidebar"] {
            font-size: 13px;
        }
        section[data-testid="stSidebar"] .stRadio label p {
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🛡️ Network Intrusion Detection System")

# ---- Load model, explainer, and sample data (cached) ----
@st.cache_resource
def load_model():
    return joblib.load("nids_model.pkl")

model = load_model()  # must load BEFORE the explainer, which needs it

@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model)

explainer = load_explainer()

@st.cache_data
def load_sample_pool():
    return pd.read_csv("dashboard_sample_data.csv")

sample_pool = load_sample_pool()
feature_cols = [c for c in sample_pool.columns if c != "Label"]

SEVERITY_MAP = {
    "BENIGN": "Low", "PortScan": "Medium", "Bot": "High",
    "DDoS": "High", "DoS Hulk": "High", "DoS GoldenEye": "High",
    "DoS slowloris": "High", "DoS Slowhttptest": "High",
    "FTP-Patator": "Medium", "SSH-Patator": "Medium",
    "Web Attack - Brute Force": "Medium", "Web Attack - XSS": "Medium",
    "Web Attack - Sql Injection": "Critical",
    "Infiltration": "Critical", "Heartbleed": "Critical",
}

def generate_real_alert():
    row = sample_pool.sample(n=1).iloc[0]
    X_row = row[feature_cols].values.reshape(1, -1)
    pred = model.predict(X_row)[0]
    proba = model.predict_proba(X_row).max()
    return {
        "timestamp": datetime.now(),
        "src_ip": f"192.168.1.{random.randint(1,254)}",
        "dst_ip": f"10.0.0.{random.randint(1,254)}",
        "protocol": "tcp",
        "attack_type": pred,
        "confidence": round(float(proba), 2),
        "severity": SEVERITY_MAP.get(pred, "Medium"),
        "true_label": row["Label"],
        "raw_features": row[feature_cols],  # kept for SHAP explanations, not for display
    }

# ---- Live mode / session state ----
live_mode = st.sidebar.toggle("🔴 Live Mode (real model predictions)", value=False)
if live_mode:
    st_autorefresh(interval=3000, key="live_refresh")

if "alerts" not in st.session_state:
    st.session_state.alerts = [generate_real_alert() for _ in range(15)]

if st.sidebar.button("🔄 Reset Data"):
    st.session_state.alerts = [generate_real_alert() for _ in range(15)]

if live_mode:
    st.session_state.alerts.insert(0, generate_real_alert())
    st.session_state.alerts = st.session_state.alerts[:200]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ NIDS")
nav_selection = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Alerts", "Traffic Monitor", "System Analytics", "Reports", "Settings"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")

if nav_selection != "Dashboard":
    st.sidebar.info(f"'{nav_selection}' page not built yet — coming in a later step.")

alerts_df = pd.DataFrame(st.session_state.alerts)

st.divider()

# ---- KPI Metrics ----
total_alerts = len(alerts_df)
high_severity = len(alerts_df[alerts_df["severity"].isin(["High", "Critical"])])
avg_confidence = alerts_df["confidence"].mean()
unique_ips = alerts_df["src_ip"].nunique()

def color_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return ",".join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))

def sparkline_kpi(label, value, delta, delta_positive_is_good, trend_data, color):
    delta_color = "#22c55e" if (delta_positive_is_good == (delta.startswith("↑"))) else "#ef4444"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=trend_data, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"rgba({color_to_rgb(color)}, 0.15)"
    ))
    fig.update_layout(
        height=30, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    with st.container(border=True):
        st.markdown(f"<p style='font-size:13px; color:#9ca3af; margin:0'>{label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:24px; font-weight:700; margin:0; line-height:1.3'>{value}</p>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size:12px; color:{delta_color}'>{delta}</span>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=label)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    sparkline_kpi("Total Alerts", total_alerts, "↑ 8.4%", True, [3,5,4,7,6,8,total_alerts], "#3b82f6")
with kpi2:
    sparkline_kpi("High Severity Alerts", high_severity, "↑ 15.3%", False, [2,3,2,4,5,6,high_severity], "#ef4444")
with kpi3:
    sparkline_kpi("Avg Detection Confidence", f"{avg_confidence:.0%}", "↑ 2.1%", True, [65,68,70,69,72,70,int(avg_confidence*100)], "#22c55e")
with kpi4:
    sparkline_kpi("Unique Source IPs", unique_ips, "↓ 3.2%", False, [18,17,16,15,14,15,unique_ips], "#a855f7")
st.divider()

# ---- Donut charts ----
donut_col1, donut_col2 = st.columns(2)

with donut_col1:
    st.subheader("Attack Categories")
    attack_dist = alerts_df["attack_type"].value_counts().reset_index()
    attack_dist.columns = ["attack_type", "count"]

    fig_donut1 = go.Figure(data=[go.Pie(
        labels=attack_dist["attack_type"],
        values=attack_dist["count"],
        hole=0.65,
        marker=dict(colors=["#ef4444", "#3b82f6", "#f59e0b", "#22c55e", "#a855f7"])
    )])
    fig_donut1.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        annotations=[dict(
            text=f"{attack_dist['count'].sum()}<br>Total",
            x=0.5, y=0.5, font_size=18, showarrow=False
        )]
    )
    st.plotly_chart(fig_donut1, use_container_width=True)

with donut_col2:
    st.subheader("Alerts by Severity")
    sev_dist = alerts_df["severity"].value_counts().reset_index()
    sev_dist.columns = ["severity", "count"]

    severity_colors = {"Critical": "#ef4444", "High": "#f59e0b", "Medium": "#3b82f6", "Low": "#22c55e"}
    colors = [severity_colors.get(s, "#888888") for s in sev_dist["severity"]]

    fig_donut2 = go.Figure(data=[go.Pie(
        labels=sev_dist["severity"],
        values=sev_dist["count"],
        hole=0.65,
        marker=dict(colors=colors)
    )])
    fig_donut2.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        annotations=[dict(
            text=f"{sev_dist['count'].sum()}<br>Total",
            x=0.5, y=0.5, font_size=18, showarrow=False
        )]
    )
    st.plotly_chart(fig_donut2, use_container_width=True)

st.markdown("<p style='font-size:18px; font-weight:600; margin-bottom:4px'>Recent Alerts</p>", unsafe_allow_html=True)

if st.button("Simulate New Alert"):
    st.session_state.alerts.insert(0, generate_real_alert())  # fixed: was calling a function that no longer exists

# ---- Filters ----
col1, col2 = st.columns(2)
with col1:
    severity_filter = st.multiselect(
        "Filter by Severity",
        options=alerts_df["severity"].unique(),
        default=alerts_df["severity"].unique()
    )
with col2:
    attack_filter = st.multiselect(
        "Filter by Attack Type",
        options=alerts_df["attack_type"].unique(),
        default=alerts_df["attack_type"].unique()
    )

filtered_df = alerts_df[
    (alerts_df["severity"].isin(severity_filter)) &
    (alerts_df["attack_type"].isin(attack_filter))
]

# Hide internal-only columns (true_label = answer key, raw_features = SHAP data) from the visible table
display_df = filtered_df.drop(columns=["true_label", "raw_features"], errors="ignore")
st.dataframe(display_df, use_container_width=True)
st.caption(f"Showing {len(filtered_df)} of {len(alerts_df)} alerts")

# ---- SHAP: Explain an Alert ----
st.markdown("### 🔍 Explain an Alert")
alert_options = [
    f"#{i} — {a['attack_type']} from {a['src_ip']} ({a['timestamp'].strftime('%H:%M:%S')})"
    for i, a in enumerate(st.session_state.alerts[:20])  # limit dropdown to most recent 20
]
selected = st.selectbox("Select an alert to explain", alert_options, index=0)
selected_idx = int(selected.split("—")[0].replace("#", "").strip())

if st.button("Generate Explanation"):
    with st.spinner("Computing SHAP explanation..."):
        alert = st.session_state.alerts[selected_idx]
        row = pd.DataFrame([alert["raw_features"]])
        pred_class = alert["attack_type"]
        class_idx = list(model.classes_).index(pred_class)

        shap_vals = explainer.shap_values(row)
        if isinstance(shap_vals, list):
            vals = shap_vals[class_idx][0]
        else:
            vals = shap_vals[0, :, class_idx]

        impact = pd.Series(vals, index=feature_cols).sort_values(key=abs, ascending=False).head(8)

        st.markdown(f"**Why was this flagged as `{pred_class}`?**")
        fig_shap = go.Figure(go.Bar(
            x=impact.values,
            y=impact.index,
            orientation='h',
            marker_color=["#ef4444" if v > 0 else "#22c55e" for v in impact.values]
        ))
        fig_shap.update_layout(
            template="plotly_dark", height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Impact on prediction (SHAP value)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Red = pushed toward this prediction, Green = pushed away from it")

st.divider()
st.subheader("Attack Overview")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Alerts by Attack Type**")
    attack_counts = filtered_df["attack_type"].value_counts().reset_index()
    attack_counts.columns = ["attack_type", "count"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=attack_counts["attack_type"],
        y=attack_counts["count"],
        marker_color="#3b82f6"
    ))
    fig_bar.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    st.markdown("**Alerts Over Time**")
    time_series = (
        filtered_df
        .set_index("timestamp")
        .resample("5min")
        .size()
        .reset_index(name="count")
    )

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=time_series["timestamp"],
        y=time_series["count"],
        mode="lines",
        line=dict(color="#ef4444", width=2),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.15)"
    ))
    fig_line.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
    )
    st.plotly_chart(fig_line, use_container_width=True)