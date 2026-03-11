import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Chapel Hill Football Analytics", layout="wide")

# -------------------------
# Custom CSS (TCU-style Purple)
# -------------------------
st.markdown("""
<style>
/* Metric Cards */
.metric-card {
    background-color: #4B2E83;       /* TCU-style purple */
    padding: 10px 15px;
    border-radius: 8px;
    text-align: center;
    color: white;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.4);
    transition: transform 0.15s;
    margin-bottom: 8px;
}
.metric-card:hover {
    transform: scale(1.03);
}
.metric-number {
    font-size: 22px;
    font-weight: 700;
    color: white;                     /* highlight number */
    margin-bottom: 4px;
}
.metric-label {
    font-size: 12px;
    color: #FFFFFFCC;                 /* subtle label */
    font-weight: 500;
}
.metric-column {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Upload
# -------------------------
st.sidebar.title("Har‑Ber Football Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Hudl Excel File", type=["xlsx","xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    # Column renaming
    COLUMN_MAP = {
        "down": ["down","dn"],
        "distance":["dist","togo","yards to go","ydstogo"],
        "yardline":["yard ln","spot","ball on"],
        "concept":["off play"],
        "play_type":["play type","playtype","type"],
        "play_direction":["play dir"],
        "gain_loss":["gn/ls"]
    }
    rename_dict = {col:s for s,v in COLUMN_MAP.items() for col in df.columns if col in v}
    df = df.rename(columns=rename_dict)

    # Custom yard groups
    def custom_yard_group(yardline):
        if pd.isna(yardline):
            return "Unknown"
        if 0 >= yardline >= -9:
            return "0 - -9"
        elif -10 >= yardline >= -19:
            return "-10 - -19"
        elif -20 >= yardline >= -29:
            return "-20 - -29"
        elif -30 >= yardline >= -39:
            return "-30 - -39"
        elif -40 >= yardline >= -50:
            return "-40 - -50"
        elif 50 >= yardline >= 40:
            return "+50 - +40"
        elif 39 >= yardline >= 30:
            return "+39 - +30"
        elif 29 >= yardline >= 20:
            return "+29 - +20"
        elif 19 >= yardline >= 10:
            return "+19 - +10"
        elif 9 >= yardline >= 0:
            return "+9 - 0"
        else:
            return "Other"
    df["yard_group"] = df["yardline"].apply(custom_yard_group)

    yard_order = [
        "0 - -9", "-10 - -19", "-20 - -29", "-30 - -39", "-40 - -49",
        "+50 - +40", "+39 - +30", "+29 - +20", "+19 - +10", "+9 - 0"
    ]

    # Sidebar Filters
    st.sidebar.header("Filters")
    down_choices = sorted(df["down"].dropna().unique())
    down_selected = st.sidebar.selectbox("Down", down_choices)
    df_down = df[df["down"] == down_selected]
    yard_choices = [yg for yg in yard_order if yg in df_down["yard_group"].unique()]
    yard_choice = st.sidebar.selectbox("Yard Group", yard_choices)
    selected = df_down[df_down["yard_group"] == yard_choice]
    if selected.empty:
        st.warning("No plays for this selection.")
        st.stop()

    # -------------------------
    # Tabs
    # -------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Snapshot",
        "Filtered by Down/Yardline",
        "Success Heatmaps",
        "Concept Effectiveness",
        "Play Prediction"
    ])

    # Tab 1 - whole dataset
    with tab1:
        avg_gain_all = round(df["gain_loss"].mean(),1)
        max_gain_all = df["gain_loss"].max()
        min_gain_all = df["gain_loss"].min()
        c1, c2, c3 = st.columns(3)
        for col, val, label in zip([c1,c2,c3], [avg_gain_all,max_gain_all,min_gain_all], ["Average Gain","Max Gain","Min Gain"]):
            col.markdown(f'<div class="metric-card"><div class="metric-number">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        # Gain/Loss Distribution
        gain_summary_all = df.groupby("gain_loss").size().reset_index(name="plays").sort_values("gain_loss")
        gain_fig_all = px.bar(
            gain_summary_all, x="gain_loss", y="plays",
            labels={"gain_loss":"Yards Gained","plays":"Number of Plays"},
            title="Gain / Loss Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#FFFFFF"]
        )

        # Most frequent concepts
        top_concepts_all = df.groupby(["concept","play_direction"]).size().reset_index(name="count").sort_values("count",ascending=False).head(8)
        concept_fig_all = px.bar(
            top_concepts_all, x="count", y="concept", color="play_direction", orientation="h",
            title="Most Frequent Concepts by Play Direction",
            template="plotly_dark",
            color_discrete_sequence=["#FFFFFF","#000000","#CCCCCC"]
        )

        # Run/Pass Pie
        play_type_summary_all = df["play_type"].value_counts().reset_index()
        play_type_summary_all.columns = ["play_type","count"]
        run_pass_fig_all = px.pie(
            play_type_summary_all, names="play_type", values="count", title="Run vs Pass %",
            color="play_type", color_discrete_map={"Run":"#000000","Pass":"#FFFFFF"}, template="plotly_dark"
        )

        # Concept Pie
        concept_summary_all = df["concept"].value_counts().head(6).reset_index()
        concept_summary_all.columns = ["concept","count"]
        concept_pie_fig_all = px.pie(
            concept_summary_all, names="concept", values="count",
            title="Most Frequent Concepts",
            color_discrete_sequence=["#FFFFFF","#CCCCCC","#000000","#FFFFFF","#CCCCCC","#000000"],
            template="plotly_dark"
        )

        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(gain_fig_all, use_container_width=True)
        r1c2.plotly_chart(concept_fig_all, use_container_width=True)
        st.markdown('<div class="section-header">Run/Pass & Concept Distribution</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(run_pass_fig_all, use_container_width=True)
        r2c2.plotly_chart(concept_pie_fig_all, use_container_width=True)

    # Tabs 2–5 remain largely unchanged; just update Plotly color sequences to match:
    # White (#FFFFFF), Black (#000000), Purple (#4B2E83)
    # Wherever "#7FDBFF", "#0A2342", "#AAAAAA" appeared, replace with the above colors.




