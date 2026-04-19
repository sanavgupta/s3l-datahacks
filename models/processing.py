import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time  
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Marine Eco-Forecast", layout="wide")

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    conn = sqlite3.connect('marine_observations.db')
    df = pd.read_sql_query("SELECT * FROM observations", conn)
    conn.close()
    
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['year_month'] = df['time'].dt.strftime('%Y-%m')
    df['common_name'] = df['common_name'].fillna('Unknown')
    return df.sort_values('year_month')

df = load_data()

# --- DYNAMIC COLOR GENERATION ---
def create_species_colors(dataframe):
    color_map = {}
    fish_species = dataframe[dataframe['biological_type'] == 'larval_fish']['common_name'].unique()
    zoo_species = dataframe[dataframe['biological_type'] == 'zooplankton']['common_name'].unique()
    
    blue_shades = px.colors.sequential.Blues[3:] + px.colors.sequential.Teal[3:]
    green_shades = px.colors.sequential.Greens[3:] + px.colors.sequential.algae[3:]
    
    for i, sp in enumerate(fish_species):
        color_map[sp] = blue_shades[i % len(blue_shades)]
    for i, sp in enumerate(zoo_species):
        color_map[sp] = green_shades[i % len(green_shades)]
        
    return color_map

species_color_map = create_species_colors(df)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Filter Observations")

st.sidebar.subheader("1. Biological Categories")
selected_species = []
broad_categories = df['biological_type'].dropna().unique()

for category in broad_categories:
    display_cat = category.replace('_', ' ').title()
    with st.sidebar.expander(f"🐟 {display_cat}" if category == 'larval_fish' else f"🔬 {display_cat}"):
        cat_species = df[df['biological_type'] == category]['common_name'].unique().tolist()
        selections = st.multiselect("Select species:", cat_species, default=cat_species[:1])
        selected_species.extend(selections)

st.sidebar.markdown("---")

st.sidebar.subheader("2. Climate Factors")
climate_factors = {'Temperature (°C)': 'T_degC', 'Salinity (PSU)': 'Salnty'}

selected_factor_labels = st.sidebar.multiselect(
    "Select data to map as gradients:", 
    list(climate_factors.keys()), 
    default=['Temperature (°C)']
)
selected_factor_cols = [climate_factors[label] for label in selected_factor_labels]

# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.title("Interactive Trophic Survey")

if not selected_species:
    st.warning("👈 Please select at least one species from the sidebar to begin.")
else:
    # --- ANIMATION STATE MANAGEMENT ---
    if "time_idx" not in st.session_state:
        st.session_state.time_idx = 0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    months = sorted(df['year_month'].unique().tolist())

    # --- CUSTOM PLAY/PAUSE UI ---
    play_col, slider_col = st.columns([1, 10])
    with play_col:
        st.write("") 
        if st.button("⏸ Pause" if st.session_state.is_playing else "▶ Play"):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()
            
    with slider_col:
        selected_time = st.select_slider(
            "⏳ Progress through time:", 
            options=months, 
            value=months[st.session_state.time_idx]
        )
        if selected_time != months[st.session_state.time_idx]:
            st.session_state.time_idx = months.index(selected_time)
            st.session_state.is_playing = False

    # --- DATA FILTERING ---
    current_date = pd.to_datetime(selected_time).tz_localize('UTC')
    cutoff_date = current_date - pd.DateOffset(months=2) 

    rolling_df = df[(df['time'] >= cutoff_date) & (df['time'] <= current_date)].copy()
    rolling_df = rolling_df[rolling_df['common_name'].isin(selected_species)]
    rolling_df = rolling_df.dropna(subset=['latitude', 'longitude', 'count'])

    filtered_df = rolling_df.sort_values('time').groupby(['latitude', 'longitude', 'common_name']).tail(1)
    filtered_df = filtered_df[filtered_df['count'] > 0] 

    historical_env_df = df[df['year_month'] <= selected_time].copy()
    latest_env_df = historical_env_df.sort_values('time').groupby(['latitude', 'longitude']).tail(1)

    # --- LAYOUT SPLIT ---
    map_col, legend_col = st.columns([4, 1])

    # ----------------------------------------
    # THE MAP (Left Column)
    # ----------------------------------------
    with map_col:
        fig = go.Figure()

        # LAYER 1: Translucent, Monochromatic Climate Gradients (Trace Consistency Logic)
        for factor_label, factor_col, c_scale in [('Temperature (°C)', 'T_degC', 'Reds'), ('Salinity (PSU)', 'Salnty', 'Blues')]:
            if factor_label in selected_factor_labels and not latest_env_df.empty and factor_col in latest_env_df.columns:
                env_df = latest_env_df.dropna(subset=[factor_col])
                if not env_df.empty:
                    fig.add_trace(go.Densitymapbox(
                        lat=env_df['latitude'], lon=env_df['longitude'], z=env_df[factor_col],
                        radius=45, colorscale=c_scale, opacity=0.2, showscale=False, hoverinfo='skip', name=factor_label
                    ))
                else:
                    # Inject empty trace if data is missing for this month
                    fig.add_trace(go.Densitymapbox(lat=[], lon=[], z=[], showscale=False, hoverinfo='skip', name=factor_label))
            else:
                # Inject empty trace if user deselected it to prevent WebGL teardown
                fig.add_trace(go.Densitymapbox(lat=[], lon=[], z=[], showscale=False, hoverinfo='skip', name=factor_label))

        # LAYER 2: The Translucent Species Dots (Trace Consistency Logic)
        for sp in selected_species:
            if not filtered_df.empty:
                sp_df = filtered_df[filtered_df['common_name'] == sp]
                if not sp_df.empty:
                    sizes = np.log1p(sp_df['count']) * 4 
                    fig.add_trace(go.Scattermapbox(
                        lat=sp_df['latitude'], lon=sp_df['longitude'], mode='markers',
                        marker=dict(size=sizes, color=species_color_map.get(sp, '#888'), opacity=0.55),
                        name=sp, text=sp_df['common_name'] + "<br>Count: " + sp_df['count'].astype(str), hoverinfo='text'
                    ))
                else:
                    # Inject empty trace if this species isn't present this month
                    fig.add_trace(go.Scattermapbox(lat=[], lon=[], mode='markers', name=sp))
            else:
                # Inject empty trace if NO species are present this month
                fig.add_trace(go.Scattermapbox(lat=[], lon=[], mode='markers', name=sp))

        fig.update_layout(
            uirevision='constant',  # Keeps the camera angle and background locked
            mapbox_style="carto-positron",
            mapbox=dict(center=dict(lat=33.5, lon=-121.0), zoom=4.2),
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=False,
            height=650
        )
        
        st.plotly_chart(fig, use_container_width=True, key="marine_map")

    # ----------------------------------------
    # THE LEGEND (Right Column)
    # ----------------------------------------
    with legend_col:
        st.subheader("Reference Key")
        legend_html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-family: sans-serif;'>"
        
        # Fish Category 
        legend_html += "<h5 style='margin-bottom: 10px; color: #333;'>🐟 Larval Fish</h5>"
        fish_species = df[df['biological_type'] == 'larval_fish']['common_name'].dropna().unique()
        for sp in fish_species:
            color = species_color_map.get(sp, "#888")
            legend_html += f"<div style='display: flex; align-items: center; margin-bottom: 8px;'><div style='width: 14px; height: 14px; background-color: {color}; border-radius: 50%; margin-right: 10px; opacity: 0.7;'></div><span style='font-size: 14px; color: #444;'>{sp}</span></div>"
            
        # Zooplankton Category 
        legend_html += "<h5 style='margin-top: 20px; margin-bottom: 10px; color: #333;'>🔬 Zooplankton</h5>"
        zoo_species = df[df['biological_type'] == 'zooplankton']['common_name'].dropna().unique()
        for sp in zoo_species:
            color = species_color_map.get(sp, "#888")
            legend_html += f"<div style='display: flex; align-items: center; margin-bottom: 8px;'><div style='width: 14px; height: 14px; background-color: {color}; border-radius: 50%; margin-right: 10px; opacity: 0.7;'></div><span style='font-size: 14px; color: #444;'>{sp}</span></div>"
            
        # --- CLIMATE GRADIENTS ---
        if selected_factor_labels:
            legend_html += "<hr style='border: none; border-top: 1px solid #ddd; margin: 15px 0;'>"
            legend_html += "<h5 style='margin-bottom: 10px; color: #333;'>🌊 Climate Layers</h5>"
            
            if 'Temperature (°C)' in selected_factor_labels:
                legend_html += """
                <div style='margin-bottom: 15px;'>
                    <div style='font-size: 14px; color: #444; margin-bottom: 4px;'>Temperature</div>
                    <div style='height: 12px; width: 100%; background: linear-gradient(to right, #fee5d9, #cb181d); border-radius: 4px;'></div>
                    <div style='display: flex; justify-content: space-between; font-size: 12px; color: #777; margin-top: 4px;'><span>Cool</span><span>Warm</span></div>
                </div>
                """
                
            if 'Salinity (PSU)' in selected_factor_labels:
                legend_html += """
                <div style='margin-bottom: 10px;'>
                    <div style='font-size: 14px; color: #444; margin-bottom: 4px;'>Salinity</div>
                    <div style='height: 12px; width: 100%; background: linear-gradient(to right, #eff3ff, #2171b5); border-radius: 4px;'></div>
                    <div style='display: flex; justify-content: space-between; font-size: 12px; color: #777; margin-top: 4px;'><span>Low</span><span>High</span></div>
                </div>
                """

        legend_html += "</div>"
        
        # Safely render the HTML through an iframe component
        components.html(legend_html, height=750, scrolling=True)
        
    # ----------------------------------------
    # DYNAMIC INSIGHTS (Bottom)
    # ----------------------------------------
    st.markdown("---")
    st.subheader(f"Aggregate Summary for {selected_time}")
    
    cols = st.columns(len(selected_factor_cols) + 1)
    
    with cols[0]:
        st.metric("Organisms Found (Last 3 Months)", f"{int(filtered_df['count'].sum()):,}" if not filtered_df.empty else "0")
        
    for i, (label, col_name) in enumerate(zip(selected_factor_labels, selected_factor_cols)):
        with cols[i+1]:
            if not latest_env_df.empty and col_name in latest_env_df.columns:
                avg_val = latest_env_df[col_name].mean()
                st.metric(f"Avg Known {label}", f"{avg_val:.2f}")
            else:
                st.metric(f"Avg Known {label}", "No Data")

    # ==========================================
    # THE AUTO-PLAY ENGINE 
    # ==========================================
    if st.session_state.is_playing:
        time.sleep(0.6)  
        if st.session_state.time_idx < len(months) - 1:
            st.session_state.time_idx += 1
            st.rerun()
        else:
            st.session_state.is_playing = False 
            st.rerun()
