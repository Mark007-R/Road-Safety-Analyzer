import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import ui_chrome
import ui_theme

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="India Road Accidents Analysis", layout="wide")
# mark.dev paper ground, Fraunces/Inter/JetBrains Mono, vermilion accent. The
# same base colours are mirrored in .streamlit/config.toml, which is the only
# way to reach widgets CSS cannot touch — change one, change both.
ui_theme.apply_theme()
ui_chrome.apply_app_css()

# ----------------------------------------------------------
# Data Loading
# ----------------------------------------------------------
@st.cache_data
def load_data():
    ds1 = pd.read_csv("data/dataset1.csv")
    ds2 = pd.read_csv("data/dataset2.csv")
    ds3 = pd.read_csv("data/dataset3.csv")
    ds4 = pd.read_csv("data/dataset4.csv")
    ds5 = pd.read_csv("data/dataset5.csv")
    ds6 = pd.read_csv("data/dataset6.csv")
    return ds1, ds2, ds3, ds4, ds5, ds6

ds1, ds2, ds3, ds4, ds5, ds6 = load_data()

# ----------------------------------------------------------
# Tabs (Dashboard + MapReduce + Forecast)
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 MapReduce Insights", "🚦 Future Forecasting","Prediction"])

# ----------------------------------------------------------
# TAB 1: MAIN DASHBOARD
# ----------------------------------------------------------
with tab1:
    st.sidebar.title("Filters (Select only ONE)")

    states = ["All"] + sorted(ds1['State Name'].dropna().unique().tolist())
    cities = ["All"] + sorted(ds1['City Name'].dropna().unique().tolist())
    years = ["All"] + sorted(ds1['Year'].dropna().unique().tolist())

    if "active_filter" not in st.session_state:
        st.session_state.active_filter = None

    selected_state = st.sidebar.selectbox("Select State", options=states)
    if selected_state != "All":
        st.session_state.active_filter = "state"
    else:
        if st.session_state.active_filter == "state":
            st.session_state.active_filter = None

    selected_city = st.sidebar.selectbox(
        "Select City",
        options=cities,
        disabled=(st.session_state.active_filter not in [None, "city"])
    )
    if selected_city != "All":
        st.session_state.active_filter = "city"
    else:
        if st.session_state.active_filter == "city":
            st.session_state.active_filter = None

    selected_year = st.sidebar.selectbox(
        "Select Year",
        options=years,
        disabled=(st.session_state.active_filter not in [None, "year"])
    )
    if selected_year != "All":
        st.session_state.active_filter = "year"
    else:
        if st.session_state.active_filter == "year":
            st.session_state.active_filter = None

    if st.session_state.active_filter == "state":
        selected_city, selected_year = "All", "All"
    elif st.session_state.active_filter == "city":
        selected_state, selected_year = "All", "All"
    elif st.session_state.active_filter == "year":
        selected_state, selected_city = "All", "All"

    def filter_data(df):
        df_filtered = df.copy()
        if selected_state != "All":
            df_filtered = df_filtered[df_filtered['State Name'] == selected_state]
        elif selected_city != "All":
            df_filtered = df_filtered[df_filtered['City Name'] == selected_city]
        elif selected_year != "All":
            df_filtered = df_filtered[df_filtered['Year'] == selected_year]
        return df_filtered

    ds1_filtered = filter_data(ds1)

    st.title("📊 India Road Accidents Interactive Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("🚗 Total Accidents", len(ds1_filtered))
    col2.metric("☠️ Total Fatalities", int(ds1_filtered['Number of Fatalities'].sum()))
    col3.metric("💥 Total Casualties", int(ds1_filtered['Number of Casualties'].sum()))

    st.markdown("---")

    # One light-to-vermilion ramp for every count-coloured chart; the
    # alternation below is kept as-is, it just alternates between the same ramp.
    chart_gradients = [ui_theme.sequential(), ui_theme.sequential()]
    color_idx = 0

    if not ds1_filtered.empty:
        severity_counts = ds1_filtered['Accident Severity'].value_counts()
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Accident Severity Distribution",
            color_discrete_sequence=ui_theme.colorway()
        )
        fig_severity.update_traces(textinfo='percent+label', pull=[0.05]*len(severity_counts))
        ui_theme.style_fig(fig_severity, title_font=ui_chrome.CHART_TITLE_FONT)
        st.plotly_chart(fig_severity, use_container_width=True, theme=None)
        color_idx = 1 - color_idx 

    if not ds1_filtered.empty:
        vehicle_counts = ds1_filtered['Vehicle Type Involved'].value_counts().reset_index()
        vehicle_counts.columns = ['Vehicle Type', 'Count']
        fig_vehicle = px.bar(
            vehicle_counts,
            x='Vehicle Type',
            y='Count',
            color='Count',
            color_continuous_scale=chart_gradients[color_idx],
            title="🚘 Accidents by Vehicle Type",
            text='Count'
        )
        fig_vehicle.update_traces(texttemplate='%{text}', textposition='outside')
        ui_theme.style_fig(fig_vehicle, title_font=ui_chrome.CHART_TITLE_FONT)
        st.plotly_chart(fig_vehicle, use_container_width=True, theme=None)
        color_idx = 1 - color_idx

    if not ds1_filtered.empty:
        day_counts = ds1_filtered['Day of Week'].value_counts().reset_index()
        day_counts.columns = ['Day of Week', 'Count']
        fig_day = px.bar(
            day_counts,
            x='Day of Week',
            y='Count',
            color='Count',
            color_continuous_scale=chart_gradients[color_idx],
            title="📅 Accidents by Day of the Week",
            text='Count'
        )
        fig_day.update_traces(texttemplate='%{text}', textposition='outside')
        ui_theme.style_fig(fig_day, title_font=ui_chrome.CHART_TITLE_FONT)
        st.plotly_chart(fig_day, use_container_width=True, theme=None)
        color_idx = 1 - color_idx

    # Map & Monthly/Time charts
    if selected_state != "All":
        ds4_filtered = ds4[ds4['STATE/UT'] == selected_state]
        if selected_year != "All":
            ds4_filtered = ds4_filtered[ds4_filtered['YEAR'] == int(selected_year)]
        if not ds4_filtered.empty:
            months = ['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']
            monthly_accidents = ds4_filtered[months].iloc[0]
            fig_month = px.area(
                x=months,
                y=monthly_accidents.values,
                title=f"Monthly Accidents Trend - {selected_state}",
                labels={'x':'Month','y':'Accidents'},
                color_discrete_sequence=ui_theme.colorway()
            )
            ui_theme.style_fig(fig_month, title_font=ui_chrome.CHART_TITLE_FONT)
            st.plotly_chart(fig_month, use_container_width=True, theme=None)
            color_idx = 1 - color_idx

    if selected_state != "All":
        ds5_filtered = ds5[ds5['STATE/UT'] == selected_state]
        if selected_year != "All":
            ds5_filtered = ds5_filtered[ds5_filtered['YEAR'] == int(selected_year)]
        if not ds5_filtered.empty:
            time_cols = ds5.columns[2:-1]
            time_counts = ds5_filtered[time_cols].iloc[0]
            fig_time = px.bar(
                x=time_cols,
                y=time_counts.values,
                color=time_counts.values,
                color_continuous_scale=chart_gradients[color_idx],
                title=f"Accidents by Time of Day - {selected_state}"
            )
            ui_theme.style_fig(fig_time, title_font=ui_chrome.CHART_TITLE_FONT)
            st.plotly_chart(fig_time, use_container_width=True, theme=None)
            color_idx = 1 - color_idx

    st.subheader("🗺️ Accident Hotspots Map")
    if 'Latitude' in ds1_filtered.columns and 'Longitude' in ds1_filtered.columns:
        ds_map = ds1_filtered.dropna(subset=['Latitude','Longitude'])
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        for _, row in ds_map.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                popup=f"<b>{row['City Name']}</b><br>Severity: {row['Accident Severity']}",
                color=ui_chrome.MAP_BAD if row['Accident Severity']=="Serious" else ui_chrome.MAP_WARN,
                fill=True
            ).add_to(m)
        st_folium(m, width=700)
    else:
        st.info("No latitude/longitude available for map plotting.")

    st.subheader("🚦 Top Accident Causes in Million+ Cities")
    if {'Cause category', 'Count'}.issubset(ds2.columns):
        ds2_filtered = ds2.copy()
        if selected_city != "All":
            ds2_filtered = ds2_filtered[ds2_filtered['Million Plus Cities']==selected_city]
        if not ds2_filtered.empty:
            cause_counts = ds2_filtered.groupby('Cause category')['Count'].sum().sort_values(ascending=False)
            cause_df = pd.DataFrame({'Cause': cause_counts.index, 'Count': cause_counts.values})
            fig_cause = px.bar(
                cause_df,
                x='Cause', y='Count',
                color='Count',
                color_continuous_scale=chart_gradients[color_idx],
                title="Top Accident Causes"
            )
            ui_theme.style_fig(fig_cause, title_font=ui_chrome.CHART_TITLE_FONT)
            st.plotly_chart(fig_cause, use_container_width=True, theme=None)
            color_idx = 1 - color_idx

    st.subheader("📈 Long-term Accident Trends in India")
    fig_trend = px.line(
        ds6,
        x='Years',
        y='Total Number of Road Accidents (in numbers)',
        markers=True,
        color_discrete_sequence=ui_theme.colorway(),
        title="Total Road Accidents Over Years"
    )
    fig_trend.update_layout(xaxis_title="Year", yaxis_title="Accidents", hovermode="x unified")
    ui_theme.style_fig(fig_trend, title_font=ui_chrome.CHART_TITLE_FONT)
    st.plotly_chart(fig_trend, use_container_width=True, theme=None)

    if 'Number of Fatalities' in ds1_filtered.columns:
        st.subheader("⚠️ Severity vs Fatalities Scatter Plot")
        fig_scatter = px.scatter(
            ds1_filtered,
            x='Number of Fatalities',
            y='Number of Casualties',
            color='Number of Fatalities',
            color_continuous_scale=ui_theme.sequential(),
            hover_data=['State Name','City Name'],
            title="Fatalities vs Casualties (Light → Vermilion Gradient)"
        )
        ui_theme.style_fig(fig_scatter, title_font=ui_chrome.CHART_TITLE_FONT)
        st.plotly_chart(fig_scatter, use_container_width=True, theme=None)

    st.markdown("---")
    st.header("🤖 Accident Severity Prediction (ML Feature)")

    if 'Accident Severity' in ds1.columns:
        ml_df = ds1[['Vehicle Type Involved', 'Day of Week', 'Number of Fatalities', 'Number of Casualties', 'Accident Severity']].dropna()

        le_vehicle = LabelEncoder()
        le_day = LabelEncoder()
        le_severity = LabelEncoder()

        ml_df['Vehicle Type Involved'] = le_vehicle.fit_transform(ml_df['Vehicle Type Involved'])
        ml_df['Day of Week'] = le_day.fit_transform(ml_df['Day of Week'])
        ml_df['Accident Severity'] = le_severity.fit_transform(ml_df['Accident Severity'])

        X = ml_df[['Vehicle Type Involved', 'Day of Week', 'Number of Fatalities', 'Number of Casualties']]
        y = ml_df['Accident Severity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        st.subheader("Enter Accident Details to Predict Severity")

        col1, col2 = st.columns(2)
        with col1:
            vehicle_input = st.selectbox("Vehicle Type", options=le_vehicle.classes_)
            fatalities_input = st.number_input("Number of Fatalities", min_value=0, max_value=50, value=0)
        with col2:
            day_input = st.selectbox("Day of Week", options=le_day.classes_)
            casualties_input = st.number_input("Number of Casualties", min_value=0, max_value=100, value=0)

        if st.button("Predict Severity 🚦"):
            input_df = pd.DataFrame({
                'Vehicle Type Involved': [le_vehicle.transform([vehicle_input])[0]],
                'Day of Week': [le_day.transform([day_input])[0]],
                'Number of Fatalities': [fatalities_input],
                'Number of Casualties': [casualties_input]
            })

            prediction = model.predict(input_df)
            predicted_label = le_severity.inverse_transform(prediction)[0]
            st.success(f"✅ Predicted Severity: **{predicted_label}**")
    else:
        st.warning("ML Prediction cannot be run — 'Accident Severity' column missing in dataset.")
# ----------------------------------------------------------
# TAB 2: MAPREDUCE INSIGHTS (ENHANCED)
# ----------------------------------------------------------
with tab2:
    st.title("🧠 MapReduce-Based Accident Insights")

    # Filters for MapReduce Tab
    states_map = ["All"] + sorted(ds1['State Name'].dropna().unique())
    years_map = ["All"] + sorted(ds1['Year'].dropna().unique())

    selected_state_map = st.selectbox("Select State for Analysis", states_map)
    selected_year_map = st.selectbox("Select Year for Analysis", years_map)

    map_df = ds1[['State Name', 'Year', 'Number of Fatalities', 'Number of Casualties']].dropna()

    # Apply filters
    if selected_state_map != "All":
        map_df = map_df[map_df['State Name'] == selected_state_map]
    if selected_year_map != "All":
        map_df = map_df[map_df['Year'] == int(selected_year_map)]

    # Reduce Step: Aggregate per State/Year
    reduce_state = map_df.groupby('State Name')[['Number of Fatalities','Number of Casualties']].sum().reset_index()
    reduce_year = map_df.groupby('Year')[['Number of Fatalities','Number of Casualties']].sum().reset_index()

    st.markdown("### 📊 Top States by Fatalities & Casualties")
    top_states = reduce_state.sort_values('Number of Fatalities', ascending=False).head(5)
    fig_top_states = px.bar(top_states, x='State Name', y=['Number of Fatalities','Number of Casualties'],
                            title="Top 5 States by Fatalities & Casualties",
                            color_discrete_sequence=ui_theme.colorway())
    ui_theme.style_fig(fig_top_states, title_font=ui_chrome.CHART_TITLE_FONT)
    st.plotly_chart(fig_top_states, use_container_width=True, theme=None)

    st.markdown("### 📈 Trend Over Years")
    if not reduce_year.empty:
        fig_trend_years = px.line(reduce_year, x='Year', y=['Number of Fatalities','Number of Casualties'],
                                  markers=True, title="Yearly Trend",
                                  color_discrete_sequence=ui_theme.colorway())
        ui_theme.style_fig(fig_trend_years, title_font=ui_chrome.CHART_TITLE_FONT)
        st.plotly_chart(fig_trend_years, use_container_width=True, theme=None)

    st.markdown("### 🌐 Accidents Heatmap by State")
    if 'Latitude' in ds1.columns and 'Longitude' in ds1.columns:
        heatmap_df = ds1.dropna(subset=['Latitude','Longitude'])
        if selected_state_map != "All":
            heatmap_df = heatmap_df[heatmap_df['State Name']==selected_state_map]
        m = folium.Map(location=[20.5937,78.9629], zoom_start=5)
        for _, row in heatmap_df.iterrows():
            folium.CircleMarker(location=[row['Latitude'],row['Longitude']],
                                radius=5,
                                color=ui_chrome.MAP_BAD if row['Number of Fatalities']>0 else ui_chrome.MAP_WARN,
                                fill=True,
                                popup=f"{row['State Name']}<br>Fatalities: {row['Number of Fatalities']}<br>Casualties: {row['Number of Casualties']}"
                               ).add_to(m)
        st_folium(m, width=700)
    else:
        st.info("No latitude/longitude data available for heatmap.")

# ----------------------------------------------------------
# TAB 3: ADVANCED FORECASTING + SCENARIO PREDICTION
# ----------------------------------------------------------
with tab3:
    st.title("🚦 Predict Next Year’s Fatalities & Casualties (ARIMA & Risk Prediction)")

    st.markdown("""
    This section uses **ARIMA** for time-series forecasting  
    and also allows **scenario-based risk prediction** using ML-like rules.
    """)

    forecast_df = ds1.groupby('Year')[['Number of Fatalities', 'Number of Casualties']].sum().reset_index()
    forecast_df = forecast_df.sort_values('Year')

    # -------- ARIMA Forecasting --------
    if len(forecast_df) > 3:
        model_fatal = ARIMA(forecast_df['Number of Fatalities'], order=(1,1,1)).fit()
        forecast_fatal = model_fatal.forecast(steps=1).iloc[0]

        model_casual = ARIMA(forecast_df['Number of Casualties'], order=(1,1,1)).fit()
        forecast_casual = model_casual.forecast(steps=1).iloc[0]

        next_year = int(forecast_df['Year'].max()) + 1
        st.success(f"📅 Forecast Year: {next_year}")
        st.metric("☠️ Predicted Fatalities", f"{int(forecast_fatal):,}")
        st.metric("💥 Predicted Casualties", f"{int(forecast_casual):,}")

        fig_forecast = px.line(forecast_df, x='Year', y=['Number of Fatalities','Number of Casualties'],
                               markers=True, title="Historical vs Predicted Trends",
                               color_discrete_sequence=ui_theme.colorway())
        fig_forecast.add_scatter(
            x=[next_year,next_year],
            y=[forecast_fatal,forecast_casual],
            mode='markers+text',
            text=["Predicted Fatalities","Predicted Casualties"],
            textposition="top center",
            # each predicted point wears its own series' colour
            marker=dict(size=12, color=ui_theme.colorway()[:2])
        )
        ui_theme.style_fig(fig_forecast, title_font=ui_chrome.CHART_TITLE_FONT)
        st.plotly_chart(fig_forecast, use_container_width=True, theme=None)
    else:
        st.warning("Insufficient yearly data for ARIMA forecasting (need at least 4 years).")

    st.markdown("---")
    st.header("⚠️ Scenario-Based Accident Risk Prediction")

    # -------- Scenario-Based Inputs --------
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"])
        terrain = st.selectbox("Terrain Type", ["Flat", "Hilly", "Mountainous"])
    with col2:
        traffic = st.slider("Traffic Density (vehicles/km)", min_value=0, max_value=200, value=50)
        visibility = st.slider("Visibility (meters)", min_value=10, max_value=1000, value=500)

    st.markdown("Click the button below to calculate **risk score** based on your inputs.")

    # -------- Risk Calculation Logic (Custom Simple Model) --------
    def calculate_risk(weather, terrain, traffic, visibility):
        risk = 0
        # Weather impact
        if weather in ["Rainy", "Foggy", "Snowy"]:
            risk += 30
        # Terrain impact
        if terrain in ["Hilly", "Mountainous"]:
            risk += 25
        # Traffic impact
        risk += (traffic / 200) * 30  # normalize to 0-30
        # Visibility impact
        if visibility < 200:
            risk += 15
        elif visibility < 500:
            risk += 10
        return min(round(risk), 100)

    if st.button("Predict Accident Risk 🚦"):
        risk_score = calculate_risk(weather, terrain, traffic, visibility)
        st.subheader(f"⚠️ Predicted Accident Risk: **{risk_score}%**")
        if risk_score > 70:
            st.error("High risk of accidents! Drive cautiously.")
        elif risk_score > 40:
            st.warning("Moderate risk. Be alert on the road.")
        else:
            st.success("Low risk. Safe conditions.")

    st.markdown("---")
    st.info("""
    ✅ **Scenario-based prediction** considers environmental and traffic factors  
    alongside historical trends (from ARIMA) for safer decision-making.
    """)

# ----------------------------------------------------------
# NEW TAB 4: SAFE ROUTE PLANNER
# ----------------------------------------------------------
with tab4:
    st.title("🗺️ Safe Route Planner")
    st.markdown("Get route suggestions based on safety, historical accident data, and simulated traffic.")

    # --- Hardcoded coordinates (for example) ---
    # In a real app, use a geocoding API (like geopy)
    CITY_COORDS = {
        "Mumbai": (19.0760, 72.8777),
        "Goa": (15.2993, 74.1240),
        "Pune": (18.5204, 73.8567),
        "Bangalore": (12.9716, 77.5946),
        "Delhi": (28.7041, 77.1025)
    }

    # --- Mock Functions to Simulate API Calls ---
    
    def get_mock_routes(origin_city, dest_city):
        """
        Simulates a routing API call (e.g., Google Maps, Mapbox).
        Returns a list of route dictionaries.
        """
        st.info(f"Simulating API call for routes from {origin_city} to {dest_city}...")
        
        origin_coords = CITY_COORDS.get(origin_city)
        dest_coords = CITY_COORDS.get(dest_city)
        
        if not origin_coords or not dest_coords:
            st.error("Could not find coordinates for one or both cities.")
            return []

        # Route 1: Coastal Route (NH 66)
        route1_path = [origin_coords, (17.6599, 73.3000), (16.9902, 73.3000), dest_coords]
        route1 = {
            "name": "Coastal Route (NH 66)",
            "path": route1_path,
            "base_accident_risk": 70,  # Simulated historical risk score
            "simulated_traffic": "High",
            "distance_km": 590,
            "color": ui_chrome.MAP_BAD
        }

        # Route 2: Inland Route (NH 48 via Pune)
        pune_coords = CITY_COORDS.get("Pune")
        route2_path = [origin_coords, pune_coords, (17.6868, 74.0182), (15.8497, 74.4977), dest_coords]
        route2 = {
            "name": "Inland Route (NH 48)",
            "path": route2_path,
            "base_accident_risk": 45, # Simulated historical risk score
            "simulated_traffic": "Low",
            "distance_km": 650,
            "color": ui_chrome.MAP_OK
        }
        
        return [route1, route2]

    def analyze_route_safety(route, weather, month):
        """
        Analyzes a single route based on user inputs.
        This simulates checking against historical accident data (ds1, ds4)
        """
        final_score = route["base_accident_risk"]
        reasoning = [f"Base Risk: {final_score} (from historical data)"]

        # 1. Weather Penalty
        if weather == "Rainy":
            final_score += 25
            reasoning.append(f"+25 (Weather: {weather})")
        elif weather == "Foggy":
            final_score += 35
            reasoning.append(f"+35 (Weather: {weather})")
        
        # 2. Month Penalty (e.g., Monsoon)
        # In a real app, you'd query ds4 here for this route
        monsoon_months = ["June", "July", "August", "September", "October"]
        if month in monsoon_months:
            final_score += 15
            reasoning.append(f"+15 (Month: {month} - Monsoon Risk)")
            
        # 3. Traffic Penalty
        if route["simulated_traffic"] == "High":
            final_score += 20
            reasoning.append(f"+20 (Traffic: High)")
        elif route["simulated_traffic"] == "Medium":
            final_score += 10
            reasoning.append(f"+10 (Traffic: Medium)")
            
        route["final_score"] = final_score
        route["reasoning"] = reasoning
        return route

    # --- UI Inputs for Tab 4 ---
    
    col1, col2 = st.columns(2)
    with col1:
        origin_city = st.selectbox("Select Origin", options=CITY_COORDS.keys(), index=0)
        month = st.selectbox(
            "Month of Travel", 
            options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            index=9 # Defaults to October
        )
    
    with col2:
        dest_city = st.selectbox("Select Destination", options=CITY_COORDS.keys(), index=1)
        weather = st.selectbox(
            "Weather Conditions",
            options=["Clear", "Rainy", "Foggy"],
            index=1 # Defaults to Rainy
        )

    if st.button("Find Safest Route Find Safest Route 🚦"):
        if origin_city == dest_city:
            st.error("Origin and Destination cannot be the same.")
        else:
            # 1. Simulate API call
            mock_routes = get_mock_routes(origin_city, dest_city)
            
            # 2. Analyze routes
            analyzed_routes = []
            for route in mock_routes:
                analyzed_routes.append(analyze_route_safety(route, weather, month))

            # 3. Find the best route (lowest score)
            best_route = min(analyzed_routes, key=lambda x: x['final_score'])

            # 4. Display Recommendation
            st.success(f"🏆 Recommendation: **{best_route['name']}**")
            st.markdown(f"""
            This route is recommended due to the lowest overall risk score.
            - **Final Risk Score:** `{best_route['final_score']}`
            - **Simulated Traffic:** `{best_route['simulated_traffic']}`
            - **Distance:** `{best_route['distance_km']}` km
            """)
            
            # 5. Display Map
            st.subheader("Route Comparison Map")
            m = folium.Map(location=CITY_COORDS[origin_city], zoom_start=7)

            # Add markers for origin and destination
            folium.Marker(CITY_COORDS[origin_city], popup=f"ORIGIN: {origin_city}", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(CITY_COORDS[dest_city], popup=f"DESTINATION: {dest_city}", icon=folium.Icon(color="red")).add_to(m)

            # Draw route polylines
            for route in analyzed_routes:
                folium.PolyLine(
                    route['path'],
                    color=route['color'],
                    weight=5,
                    opacity=0.8,
                    popup=f"<b>{route['name']}</b><br>Risk Score: {route['final_score']}<br>Traffic: {route['simulated_traffic']}"
                ).add_to(m)

            st_folium(m, width=725, height=500)
            
            # 6. Show detailed scoring
            with st.expander("See Detailed Risk Scoring"):
                for route in analyzed_routes:
                    st.markdown(f"#### {route['name']}")
                    st.markdown(f"- **Final Score: {route['final_score']}**")
                    for reason in route['reasoning']:
                        st.text(f"  - {reason}")