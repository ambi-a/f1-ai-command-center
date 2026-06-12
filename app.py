import streamlit as st
import pandas as pd
import plotly.express as px
import fastf1
from commentary import generate_commentary

from model import train_tyre_model
from strategy import simulate_strategy, optimize_strategy

st.set_page_config(
    page_title="F1 AI Command Center",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #050505 0%,
            #111827 45%,
            #1F2937 100%
        );
        color: #F9FAFB;
    }

    section[data-testid="stSidebar"] {
        background: #070707;
        border-right: 1px solid #E50914;
    }

    section[data-testid="stSidebar"] * {
        color: #F9FAFB !important;
    }

    section[data-testid="stSidebar"] label {
        color: #FCA5A5 !important;
        font-weight: 700 !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: white !important;
        border: 1px solid #374151 !important;
        border-radius: 14px !important;
    }

    div[data-baseweb="popover"] {
        background-color: #111827 !important;
    }

    div[data-baseweb="menu"] {
        background-color: #111827 !important;
    }

    ul[role="listbox"] {
        background-color: #111827 !important;
    }

    li[role="option"] {
        background-color: #111827 !important;
        color: #FFFFFF !important;
    }

    li[role="option"] div {
        color: #FFFFFF !important;
    }

    li[role="option"]:hover {
        background-color: #E50914 !important;
    }

    li[role="option"]:hover div {
        color: #FFFFFF !important;
    }

    .stButton button {
        background: #E50914;
        color: white !important;
        border-radius: 999px;
        border: none;
        font-weight: 800;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    p, label, span {
        color: #E5E7EB !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 10px 22px;
        color: white !important;
    }

    .stTabs [aria-selected="true"] {
        background: #E50914 !important;
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="
        background: rgba(229,9,20,0.15);
        border-left: 4px solid #E50914;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
        font-size: 15px;
        line-height: 1.6;
    ">
        <b>Version 0.8 Beta</b><br>
        This project is under active development.
        Predictions, strategy recommendations, and AI-generated analysis are experimental and may be inaccurate.
    </div>
    """,
    unsafe_allow_html=True
)
fastf1.Cache.enable_cache("cache")

# ---------------- HELPERS ----------------
def format_lap_time(seconds):
    if pd.isna(seconds):
        return "N/A"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes}:{remaining_seconds:06.3f}"


def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font_color="#FFFFFF",
        title_font_size=24,
        legend=dict(
            font=dict(
                color="white",
                size=16
            ),
            title=dict(
                font=dict(
                    color="white",
                    size=18
                )
            ),
            bgcolor="rgba(0,0,0,0.25)"
        ),
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.22)"
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.22)"
    )

    return fig


team_colors = {
    "VER": "#1E5BC6",
    "PER": "#1E5BC6",
    "HAM": "#00D2BE",
    "RUS": "#00D2BE",
    "LEC": "#DC0000",
    "SAI": "#DC0000",
    "NOR": "#FF8700",
    "PIA": "#FF8700",
    "ALO": "#006F62",
    "STR": "#006F62",
    "OCO": "#0090FF",
    "GAS": "#0090FF",
    "ALB": "#005AFF",
    "SAR": "#005AFF",
    "TSU": "#6692FF",
    "RIC": "#6692FF"
}

tyre_colors = {
    "SOFT": "#FF2D2D",
    "MEDIUM": "#FFD800",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#00FF85",
    "WET": "#0077FF"
}

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style='padding:40px; border-radius:28px; background:linear-gradient(135deg,#050505,#111827); border:1px solid rgba(255,255,255,0.12); box-shadow:0 10px 40px rgba(0,0,0,0.45); margin-bottom:30px;'>

    <h1 style='color:white; font-size:72px; font-weight:900; margin-bottom:10px;'>
    🏎️ F1 AI Command Center
    </h1>

    <p style='color:#EF4444; font-size:20px; font-weight:800; letter-spacing:1px;'>
    LIVE AI RACE ANALYTICS PLATFORM
    </p>

    <p style='color:#D1D5DB; font-size:20px; line-height:1.7; max-width:1000px;'>
    AI-powered Formula 1 telemetry intelligence platform featuring race strategy optimization,
    tyre degradation prediction, pit-stop simulation, telemetry analytics, and machine learning powered race insights.
    </p>

    <div style='margin-top:25px; display:flex; gap:12px; flex-wrap:wrap;'>

    <span style='background:#E50914; padding:10px 18px; border-radius:999px; color:white; font-weight:700;'>
    Real-Time Telemetry
    </span>

    <span style='background:rgba(255,255,255,0.10); padding:10px 18px; border-radius:999px; color:white; font-weight:700;'>
    AI Strategy Engine
    </span>

    <span style='background:rgba(255,255,255,0.10); padding:10px 18px; border-radius:999px; color:white; font-weight:700;'>
    Tyre Intelligence
    </span>

    <span style='background:rgba(255,255,255,0.10); padding:10px 18px; border-radius:999px; color:white; font-weight:700;'>
    ML Predictions
    </span>

    </div>
    </div>
    """,
    unsafe_allow_html=True
)

fastf1.Cache.enable_cache("cache")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Race Controls")

year = st.sidebar.selectbox(
    "Select Year",
    list(range(2018, 2026))[::-1]
)

race_options = {
    2025: [
        "Australia", "China", "Japan", "Bahrain", "Saudi Arabia",
        "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada",
        "Austria", "Great Britain", "Belgium", "Hungary",
        "Netherlands", "Italy", "Azerbaijan", "Singapore",
        "United States", "Mexico City", "Brazil", "Las Vegas",
        "Qatar", "Abu Dhabi"
    ],
    2024: [
        "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
        "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
        "Austria", "Great Britain", "Hungary", "Belgium",
        "Netherlands", "Italy", "Azerbaijan", "Singapore",
        "United States", "Mexico City", "Brazil", "Las Vegas",
        "Qatar", "Abu Dhabi"
    ],
    2023: [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan",
        "Miami", "Monaco", "Spain", "Canada", "Austria",
        "Great Britain", "Hungary", "Belgium", "Netherlands",
        "Italy", "Singapore", "Japan", "Qatar", "United States",
        "Mexico City", "Brazil", "Las Vegas", "Abu Dhabi"
    ],
    2022: [
        "Bahrain", "Saudi Arabia", "Australia", "Emilia Romagna",
        "Miami", "Spain", "Monaco", "Azerbaijan", "Canada",
        "Great Britain", "Austria", "France", "Hungary",
        "Belgium", "Netherlands", "Italy", "Singapore",
        "Japan", "United States", "Mexico City", "Brazil",
        "Abu Dhabi"
    ],
    2021: [
        "Bahrain", "Emilia Romagna", "Portugal", "Spain", "Monaco",
        "Azerbaijan", "France", "Styria", "Austria", "Great Britain",
        "Hungary", "Belgium", "Netherlands", "Italy", "Russia",
        "Turkey", "United States", "Mexico City", "Brazil",
        "Qatar", "Saudi Arabia", "Abu Dhabi"
    ],
    2020: [
        "Austria", "Styria", "Hungary", "Great Britain",
        "70th Anniversary", "Spain", "Belgium", "Italy",
        "Tuscany", "Russia", "Eifel", "Portugal",
        "Emilia Romagna", "Turkey", "Bahrain", "Sakhir",
        "Abu Dhabi"
    ],
    2019: [
        "Australia", "Bahrain", "China", "Azerbaijan", "Spain",
        "Monaco", "Canada", "France", "Austria", "Great Britain",
        "Germany", "Hungary", "Belgium", "Italy", "Singapore",
        "Russia", "Japan", "Mexico", "United States",
        "Brazil", "Abu Dhabi"
    ],
    2018: [
        "Australia", "Bahrain", "China", "Azerbaijan", "Spain",
        "Monaco", "Canada", "France", "Austria", "Great Britain",
        "Germany", "Hungary", "Belgium", "Italy", "Singapore",
        "Russia", "Japan", "United States", "Mexico",
        "Brazil", "Abu Dhabi"
    ]
}

race = st.sidebar.selectbox(
    "Select Race",
    race_options[year]
)

session_type = st.sidebar.selectbox(
    "Session",
    ["R", "Q", "FP1", "FP2", "FP3"]
)

load_button = st.sidebar.button("Load Race Data")

# ---------------- LOAD DATA ----------------
if load_button:
    with st.spinner("Loading F1 data..."):
        session = fastf1.get_session(year, race, session_type)
        session.load()

        st.session_state["laps"] = session.laps
        st.session_state["year"] = year
        st.session_state["race"] = race
        st.session_state["session_type"] = session_type

        st.success("Race data loaded!")
        st.toast("Telemetry loaded successfully 🏎️")

# ---------------- MAIN DASHBOARD ----------------
if "laps" in st.session_state:
    laps = st.session_state["laps"]

    drivers = sorted(laps["Driver"].dropna().unique())

    driver = st.sidebar.selectbox(
        "Select Driver",
        drivers,
        key="driver_selector"
    )

    if "last_driver" not in st.session_state:
        st.session_state["last_driver"] = driver

    if st.session_state["last_driver"] != driver:
        st.session_state["last_driver"] = driver

        for key in list(st.session_state.keys()):
            if key.startswith("strategy_") or key.startswith("commentary_"):
                del st.session_state[key]

        st.rerun()

    driver_laps = laps[laps["Driver"] == driver].copy()
    driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].dt.total_seconds()

    driver_laps = driver_laps.dropna(
        subset=["LapTimeSeconds", "TyreLife", "Compound"]
    )

    if driver_laps.empty:
        st.warning("No valid lap data available for this driver/session.")
        st.stop()

    driver_laps["FormattedLapTime"] = driver_laps[
        "LapTimeSeconds"
    ].apply(format_lap_time)

    line_color = team_colors.get(driver, "#FF1801")

    st.header(f"{driver} Race Analysis")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fastest Lap",
        format_lap_time(driver_laps["LapTimeSeconds"].min())
    )

    col2.metric(
        "Average Race Pace",
        format_lap_time(driver_laps["LapTimeSeconds"].mean())
    )

    col3.metric(
        "Valid Laps",
        len(driver_laps)
    )

    col4.metric(
        "Tyre Compounds Used",
        len(driver_laps["Compound"].dropna().unique())
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Lap Analysis",
            "Tyre AI",
            "Strategy Simulator",
            "Telemetry",
            "Race Summary"
        ]
    )

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Lap Time Trend")

        st.markdown(
            """
            This chart shows the driver's lap time across the race.  
            Lower lap time means faster pace. Spikes usually mean traffic, pit stops, mistakes, or tyre drop-off.
            """
        )

        fig = px.line(
            driver_laps,
            x="LapNumber",
            y="LapTimeSeconds",
            hover_data={
                "FormattedLapTime": True,
                "LapTimeSeconds": False,
                "LapNumber": True,
                "Compound": True,
                "TyreLife": True
            },
            labels={
                "LapNumber": "Race Lap Number",
                "LapTimeSeconds": "Lap Time in Seconds"
            },
            title=f"{driver} Lap Time Trend"
        )

        fig.update_traces(
            line_color=line_color,
            line_width=3
        )

        fig.update_layout(
            yaxis_tickformat=".1f"
        )

        fig = style_plot(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Tyre Compound Pace")

        st.markdown(
            """
            This chart shows how each tyre compound performed.  
            Bigger dots mean older tyres. If lap times rise as tyre life increases, the tyre is degrading.
            """
        )

        fig2 = px.scatter(
            driver_laps,
            x="TyreLife",
            y="LapTimeSeconds",
            color="Compound",
            color_discrete_map=tyre_colors,
            size="LapNumber",
            hover_data={
                "LapNumber": True,
                "FormattedLapTime": True,
                "LapTimeSeconds": False,
                "TyreLife": True,
                "Compound": True
            },
            labels={
                "TyreLife": "Tyre Age in Laps",
                "LapTimeSeconds": "Lap Time in Seconds",
                "Compound": "Tyre Compound"
            },
            title="Tyre Age vs Lap Time"
        )

        fig2.update_traces(
            marker=dict(
                opacity=0.9,
                line=dict(
                    width=1,
                    color="white"
                )
            )
        )

        fig2 = style_plot(fig2)

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("AI Tyre Degradation Intelligence")

        model, columns, mae, clean_df = train_tyre_model(driver_laps)

        if model is None:
            st.warning("Not enough lap data available for ML prediction.")
        else:
            st.metric(
                "AI Prediction Error",
                f"± {mae:.2f} sec",
                help="Mean Absolute Error. Lower is better."
            )

            st.markdown(
                f"""
                This model predicts lap time using tyre age, lap number, stint, and tyre compound.  
                It is currently usually within about **{mae:.2f} seconds** of the real lap time.
                """
            )

            prediction_input = pd.get_dummies(
                clean_df[["LapNumber", "TyreLife", "Stint", "Compound"]],
                columns=["Compound"]
            )

            prediction_input = prediction_input.reindex(
                columns=columns,
                fill_value=0
            )

            clean_df["PredictedLapTime"] = model.predict(prediction_input)
            clean_df["Actual Time"] = clean_df[
                "LapTimeSeconds"
            ].apply(format_lap_time)
            clean_df["Predicted Time"] = clean_df[
                "PredictedLapTime"
            ].apply(format_lap_time)

            degradation_rows = []

            for compound in clean_df["Compound"].dropna().unique():
                compound_df = clean_df[clean_df["Compound"] == compound].copy()

                if len(compound_df) >= 3:
                    early_laps = compound_df.nsmallest(
                        3,
                        "TyreLife"
                    )["LapTimeSeconds"].mean()

                    late_laps = compound_df.nlargest(
                        3,
                        "TyreLife"
                    )["LapTimeSeconds"].mean()

                    degradation = late_laps - early_laps

                    degradation_rows.append(
                        {
                            "Compound": compound,
                            "Early Pace": early_laps,
                            "Late Pace": late_laps,
                            "Degradation Seconds": degradation,
                            "Early Pace Time": format_lap_time(early_laps),
                            "Late Pace Time": format_lap_time(late_laps)
                        }
                    )

            degradation_df = pd.DataFrame(degradation_rows)

            if not degradation_df.empty:
                best_compound_row = degradation_df.loc[
                    degradation_df["Degradation Seconds"].idxmin()
                ]

                worst_compound_row = degradation_df.loc[
                    degradation_df["Degradation Seconds"].idxmax()
                ]

                col_a, col_b, col_c = st.columns(3)

                col_a.metric(
                    "Most Stable Compound",
                    best_compound_row["Compound"],
                    f"{best_compound_row['Degradation Seconds']:.2f}s drop-off"
                )

                col_b.metric(
                    "Highest Degradation",
                    worst_compound_row["Compound"],
                    f"{worst_compound_row['Degradation Seconds']:.2f}s drop-off"
                )

                col_c.metric(
                    "Tyres Analysed",
                    len(degradation_df)
                )

            st.markdown("### Actual vs AI-Predicted Lap Time")

            fig3 = px.line(
                clean_df,
                x="TyreLife",
                y=["LapTimeSeconds", "PredictedLapTime"],
                color_discrete_map={
                    "LapTimeSeconds": "#FFFFFF",
                    "PredictedLapTime": "#E50914"
                },
                labels={
                    "TyreLife": "Tyre Age in Laps",
                    "value": "Lap Time in Seconds",
                    "variable": "Line Type"
                },
                title="AI Tyre Degradation Prediction"
            )

            fig3 = style_plot(fig3)

            st.plotly_chart(
                fig3,
                width="stretch"
            )

            if not degradation_df.empty:
                st.markdown("### Compound Degradation Comparison")

                fig_deg = px.bar(
                    degradation_df,
                    x="Compound",
                    y="Degradation Seconds",
                    color="Compound",
                    color_discrete_map=tyre_colors,
                    hover_data={
                        "Early Pace Time": True,
                        "Late Pace Time": True,
                        "Degradation Seconds": ":.2f"
                    },
                    labels={
                        "Compound": "Tyre Compound",
                        "Degradation Seconds": "Pace Drop-off in Seconds"
                    },
                    title="Tyre Drop-off by Compound"
                )

                fig_deg.update_traces(
                    marker=dict(
                        line=dict(
                            width=1,
                            color="white"
                        )
                    )
                )

                fig_deg = style_plot(fig_deg)

                st.plotly_chart(
                    fig_deg,
                    width="stretch"
                )

            with st.expander("What does this mean?"):
                st.write(
                    """
                    The white line is the real lap time.  
                    The red line is the AI prediction.  
                    If the lap time rises as tyre age increases, the tyre is degrading.  
                    The degradation chart compares early-stint pace with late-stint pace for each compound.
                    """
                )

            st.dataframe(
                clean_df[
                    [
                        "LapNumber",
                        "Compound",
                        "TyreLife",
                        "Actual Time",
                        "Predicted Time"
                    ]
                ],
                width="stretch"
            )

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("AI Pit Strategy Optimizer")

        st.markdown(
            """
            This optimizer tests many possible pit-stop laps and tyre combinations.  
            It predicts the total race time for each strategy and selects the fastest option.
            """
        )

        total_laps = st.slider(
            "Total Race Laps",
            20,
            80,
            53
        )

        pit_loss = st.slider(
            "Pit Stop Time Loss in Seconds",
            15,
            35,
            22
        )

        model, columns, mae, clean_df = train_tyre_model(driver_laps)

        if model is None:
            st.warning("Not enough lap data available for strategy optimization.")
        else:
            run_optimizer = st.button(
                "Run Strategy Optimizer",
                key=f"run_optimizer_{driver}"
            )

            if not run_optimizer:
                st.info("Click Run Strategy Optimizer to calculate the best pit strategy.")
            else:
                with st.spinner("Searching best pit strategy..."):
                    strategy_df, best = optimize_strategy(
                        model=model,
                        columns=columns,
                        total_laps=total_laps,
                        pit_loss=pit_loss
                    )

                strategy_df["Formatted Total Time"] = strategy_df[
                    "Predicted Total Time"
                ].apply(format_lap_time)

                st.metric(
                    "Best Strategy",
                    best["Strategy"],
                    best["Compounds"]
                )

                st.metric(
                    "Best Predicted Race Time",
                    format_lap_time(best["Predicted Total Time"]),
                    f"Pit lap(s): {best['Pit Laps']}"
                )

                st.markdown("### Strategy Search Results")

                top_results = strategy_df.sort_values(
                    "Predicted Total Time"
                ).head(15)

                fig4 = px.bar(
                    top_results,
                    x="Strategy",
                    y="Predicted Total Time",
                    color="Strategy",
                    hover_data={
                        "Pit Laps": True,
                        "Compounds": True,
                        "Formatted Total Time": True,
                        "Predicted Total Time": False
                    },
                    labels={
                        "Predicted Total Time": "Predicted Race Time in Seconds",
                        "Strategy": "Strategy Type"
                    },
                    title="Top 15 Fastest Strategy Options"
                )

                fig4 = style_plot(fig4)

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

                st.markdown("### Best Tyre Stint Plan")

                stint_df = best["StintPlan"].copy()

                stint_df["Formatted Lap Time"] = stint_df[
                    "PredictedLapTime"
                ].apply(format_lap_time)

                fig_stint = px.scatter(
                    stint_df,
                    x="Lap",
                    y="Compound",
                    color="Compound",
                    color_discrete_map=tyre_colors,
                    size="TyreLife",
                    hover_data={
                        "Lap": True,
                        "TyreLife": True,
                        "Formatted Lap Time": True
                    },
                    title="Recommended Tyre Usage Across Race"
                )

                fig_stint.update_traces(
                    marker=dict(
                        opacity=0.9,
                        line=dict(
                            width=1,
                            color="white"
                        )
                    )
                )

                fig_stint.update_layout(
                    xaxis_title="Race Lap",
                    yaxis_title="Tyre Compound"
                )

                fig_stint = style_plot(fig_stint)

                st.plotly_chart(
                    fig_stint,
                    use_container_width=True
                )

                st.dataframe(
                    top_results[
                        [
                            "Strategy",
                            "Pit Laps",
                            "Compounds",
                            "Formatted Total Time"
                        ]
                    ],
                    use_container_width=True
                )

    # ---------------- TAB 4 ----------------
    with tab4:
        st.subheader("Telemetry Speed Trace")

        st.markdown(
            """
            This chart shows the driver's speed across their fastest lap.

            - X-axis: distance around the track in meters  
            - Y-axis: speed in kilometers per hour  
            - High points usually mean straights  
            - Low points usually mean braking zones and corners
            """
        )

        try:
            fastest_lap = driver_laps.pick_fastest()

            if fastest_lap is None:
                st.warning("No fastest lap telemetry available.")
            else:
                car_data = fastest_lap.get_car_data()

                if car_data.empty:
                    st.warning("Telemetry unavailable for this lap.")
                else:
                    car_data = car_data.add_distance()

                    fig5 = px.scatter(
                        car_data,
                        x="Distance",
                        y="Speed",
                        color="Speed",
                        color_continuous_scale=[
                            "#00F5FF",
                            "#00B3FF",
                            "#0066FF",
                            "#7C3AED",
                            "#FF003C"
                        ],
                        labels={
                            "Distance": "Distance Around Track in Meters",
                            "Speed": "Speed in km/h"
                        },
                        title=f"{driver} Fastest Lap Telemetry"
                    )

                    fig5.update_traces(
                        marker=dict(
                            size=8,
                            line=dict(
                                width=0
                            )
                        )
                    )

                    fig5.update_layout(
                        coloraxis_colorbar=dict(
                            title="Speed in km/h"
                        )
                    )

                    fig5 = style_plot(fig5)

                    st.plotly_chart(
                        fig5,
                        use_container_width=True
                    )

                    st.markdown("### Telemetry Summary")

                    col_a, col_b, col_c = st.columns(3)

                    col_a.metric(
                        "Top Speed",
                        f"{car_data['Speed'].max():.1f} km/h"
                    )

                    col_b.metric(
                        "Average Speed",
                        f"{car_data['Speed'].mean():.1f} km/h"
                    )

                    col_c.metric(
                        "Lap Distance",
                        f"{car_data['Distance'].max():.0f} m"
                    )

        except Exception:
            st.warning("Telemetry data not available for this session or driver.")

    # ---------------- TAB 5 ----------------
    with tab5:
        st.subheader("AI Race Engineer Summary")

        best_lap = driver_laps["LapTimeSeconds"].min()
        avg_lap = driver_laps["LapTimeSeconds"].mean()
        tyre_used = driver_laps["Compound"].dropna().unique()

        model, columns, mae, clean_df = train_tyre_model(driver_laps)

        st.markdown(
            """
            This section summarizes the driver's pace, tyre behaviour, and telemetry without running the heavy strategy optimizer automatically.
            """
        )

        if model is None:
            summary = f"""
            ### {driver} Race Engineer Report

            **Pace:**  
            Fastest lap: **{format_lap_time(best_lap)}**  
            Average race pace: **{format_lap_time(avg_lap)}**

            **Tyres Used:**  
            {", ".join(tyre_used)}

            There was not enough clean lap data to generate a reliable AI summary.
            """

            st.info(summary)

        else:
            prediction_input = pd.get_dummies(
                clean_df[["LapNumber", "TyreLife", "Stint", "Compound"]],
                columns=["Compound"]
            )

            prediction_input = prediction_input.reindex(
                columns=columns,
                fill_value=0
            )

            clean_df["PredictedLapTime"] = model.predict(prediction_input)

            degradation_rows = []

            for compound in clean_df["Compound"].dropna().unique():
                compound_df = clean_df[clean_df["Compound"] == compound].copy()

                if len(compound_df) >= 3:
                    early_pace = compound_df.nsmallest(
                        3,
                        "TyreLife"
                    )["LapTimeSeconds"].mean()

                    late_pace = compound_df.nlargest(
                        3,
                        "TyreLife"
                    )["LapTimeSeconds"].mean()

                    degradation_rows.append(
                        {
                            "Compound": compound,
                            "Degradation": late_pace - early_pace,
                            "Early Pace": early_pace,
                            "Late Pace": late_pace
                        }
                    )

            degradation_df = pd.DataFrame(degradation_rows)

            if not degradation_df.empty:
                most_stable = degradation_df.loc[
                    degradation_df["Degradation"].idxmin()
                ]

                highest_drop = degradation_df.loc[
                    degradation_df["Degradation"].idxmax()
                ]

                tyre_text = (
                    f"The most stable compound was **{most_stable['Compound']}**, "
                    f"with about **{most_stable['Degradation']:.2f}s** drop-off. "
                    f"The highest degradation appeared on **{highest_drop['Compound']}**, "
                    f"with about **{highest_drop['Degradation']:.2f}s** drop-off."
                )
            else:
                tyre_text = "There was not enough compound-level data to compare tyre degradation reliably."

            try:
                fastest_lap = driver_laps.pick_fastest()
                car_data = fastest_lap.get_car_data().add_distance()

                telemetry_text = (
                    f"Telemetry shows a top speed of **{car_data['Speed'].max():.1f} km/h** "
                    f"and an average fastest-lap speed of **{car_data['Speed'].mean():.1f} km/h**."
                )
            except Exception:
                telemetry_text = "Telemetry was not available for this driver/session."

            summary = f"""
            ### {driver} Race Engineer Report

            **Pace:**  
            Fastest lap: **{format_lap_time(best_lap)}**  
            Average race pace: **{format_lap_time(avg_lap)}**  
            AI prediction error: **±{mae:.2f} seconds**

            **Tyre Analysis:**  
            {tyre_text}

            **Strategy Note:**  
            Use the **Strategy Simulator** tab and click **Run Strategy Optimizer** to calculate the best pit plan. This summary does not run the optimizer automatically, so switching drivers stays fast.

            **Telemetry Insight:**  
            {telemetry_text}

            **Engineer Takeaway:**  
            {driver}'s strongest performance window appears when tyre age is lower and lap-time variance is controlled. If degradation rises quickly, an earlier stop or a two-stop strategy may be more competitive.
            """

            st.info(summary)

            st.markdown("### AI Live Commentary")

            run_commentary = st.button(
                "Generate AI Commentary",
                key=f"commentary_button_{driver}"
            )

            if run_commentary:

                with st.spinner("Generating AI race commentary..."):

                    ai_commentary = generate_commentary(
                        driver=driver,

                        avg_lap=format_lap_time(avg_lap),

                        best_lap=format_lap_time(best_lap),

                        tyre_text=tyre_text,

                        telemetry_text=telemetry_text
                    )

                st.success(ai_commentary)

            else:

                st.info("Click Generate AI Commentary to create a local AI race summary.")

            if not degradation_df.empty:
                
                st.markdown("### Tyre Degradation Table")

                degradation_df["Early Pace Time"] = degradation_df[
                    "Early Pace"
                ].apply(format_lap_time)

                degradation_df["Late Pace Time"] = degradation_df[
                    "Late Pace"
                ].apply(format_lap_time)

                st.dataframe(
                    degradation_df[
                        [
                            "Compound",
                            "Early Pace Time",
                            "Late Pace Time",
                            "Degradation"
                        ]
                    ],
                    width="stretch"
                )

    st.subheader("Data Preview")

    preview_df = driver_laps[
        [
            "LapNumber",
            "Driver",
            "Compound",
            "TyreLife",
            "LapTimeSeconds",
            "FormattedLapTime"
        ]
    ].copy()

    preview_df = preview_df.rename(
        columns={
            "LapNumber": "Race Lap",
            "Driver": "Driver",
            "Compound": "Tyre Compound",
            "TyreLife": "Tyre Age in Laps",
            "LapTimeSeconds": "Lap Time in Seconds",
            "FormattedLapTime": "Lap Time"
        }
    )

    st.dataframe(
        preview_df,
        use_container_width=True
    )

else:
    st.info("Use the sidebar to select a race and load session data.")
