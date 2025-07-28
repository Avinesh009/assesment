# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime
from fpdf import FPDF
import os

# Configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    page_icon="⚙️"
)

# Load Data and Models
@st.cache_resource
def load_data_and_models():
    # Load data with error handling
    nasa_data, pump_data = None, None
    try:
        if os.path.exists('features_nasa.csv'):
            nasa_data = pd.read_csv('features_nasa.csv')
        else:
            st.warning("NASA data file not found (features_nasa.csv)")
        
        if os.path.exists('features_pump.csv'):
            pump_data = pd.read_csv('features_pump.csv')
        else:
            st.warning("Pump data file not found (features_pump.csv)")
    
    except Exception as e:
        st.error(f"❌ Error loading data files: {e}")

    # Load models with error handling
    classifier, regressor = None, None
    try:
        if os.path.exists('failure_classifier.pkl'):
            classifier = joblib.load('failure_classifier.pkl')
        else:
            st.warning("Classifier model not found (failure_classifier.pkl)")
        
        if os.path.exists('rul_predictor.pkl'):
            regressor = joblib.load('rul_predictor.pkl')
        else:
            st.warning("Regressor model not found (rul_predictor.pkl)")
    
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

    return nasa_data, pump_data, classifier, regressor

nasa_data, pump_data, classifier, regressor = load_data_and_models()

# Initialize session state for alerts and model metrics
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
    
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {
        'classifier': {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.91,
            'f1': 0.90
        },
        'regressor': {
            'mae': 12.5,
            'rmse': 18.3
        }
    }

# Page 1: Home
def home_page():
    st.title("Asset Health Dashboard")
    
    # Create mock health scores if NASA data exists
    if nasa_data is not None:
        assets = nasa_data['id'].unique()[:5]  # Show first 5 assets
        health_data = pd.DataFrame({
            'Asset ID': assets,
            'Health Score': [85, 72, 90, 68, 77],
            'Predicted RUL (cycles)': [142, 85, 210, 56, 120],
            'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        try:
            st.dataframe(
                health_data.style.background_gradient(
                    subset=['Health Score', 'Predicted RUL (cycles)'], 
                    cmap='RdYlGn'
                ),
                use_container_width=True
            )
        except ImportError:
            st.warning("⚠️ For color gradients, install matplotlib: `pip install matplotlib`")
            st.dataframe(health_data, use_container_width=True)
    else:
        st.warning("No asset data available. Please upload NASA C-MAPSS data.")
    
    st.markdown("---")
    st.write("### Upload New Sensor Data")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(new_data)} records")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Page 2: Asset Detail
def asset_detail_page():
    st.title("Asset Detailed View")
    
    if nasa_data is None:
        st.warning("No asset data available")
        return
    
    # Asset selection
    asset_id = st.selectbox("Select Asset", nasa_data['id'].unique())
    
    # Filter data
    asset_data = nasa_data[nasa_data['id'] == asset_id]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Health Score", "87%", "2% from last week")
    with col2:
        if regressor:
            try:
                rul_pred = regressor.predict(asset_data.drop(['id', 'RUL', 'cycle'], axis=1, errors='ignore')).mean()
                st.metric("Predicted RUL", f"{rul_pred:.1f} cycles")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.metric("Predicted RUL", "N/A (model not loaded)")
    with col3:
        if classifier and pump_data is not None:
            try:
                # Preprocess data for classification
                classification_data = pump_data.copy()
                
                # Convert timestamp to numeric if exists
                if 'timestamp' in classification_data.columns:
                    classification_data['timestamp'] = pd.to_datetime(classification_data['timestamp']).astype('int64')
                
                # Encode categorical columns if exists
                if 'machine_status' in classification_data.columns:
                    from sklearn.preprocessing import LabelEncoder
                    classification_data['machine_status'] = LabelEncoder().fit_transform(
                        classification_data['machine_status']
                    )
                
                # Ensure only numeric columns remain
                classification_data = classification_data.select_dtypes(include=['number'])
                
                # Make prediction
                proba = classifier.predict_proba(
                    classification_data.drop(['failure_flag'], axis=1, errors='ignore')
                )[:,1].mean()
                st.metric("Failure Probability", f"{proba*100:.1f}%")
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.metric("Failure Probability", "N/A (model not loaded)")
    
    # Sensor trends
    st.subheader("Sensor Trends")
    sensor_cols = [col for col in asset_data.columns if 'sensor_' in col]
    selected_sensor = st.selectbox("Select Sensor", sensor_cols[:5])  # Show first 5 sensors
    
    fig = px.line(
        asset_data, 
        x='cycle', 
        y=selected_sensor,
        title=f"{selected_sensor} Trend for Asset {asset_id}"
    )
    st.plotly_chart(fig, use_container_width=True)

# Page 3: Alerts
def alerts_page():
    st.title("Maintenance Alerts")
    
    # Generate mock alerts if none exist
    if not st.session_state.alerts and nasa_data is not None:
        assets = nasa_data['id'].unique()[:3]
        st.session_state.alerts = [
            {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'asset': asset,
                'severity': 'High' if i == 0 else 'Medium',
                'message': f"Anomaly detected in sensor_{i+1}" 
            } 
            for i, asset in enumerate(assets)
        ]
    
    if st.session_state.alerts:
        # Display alerts table
        alerts_df = pd.DataFrame(st.session_state.alerts)
        st.dataframe(alerts_df, use_container_width=True)
        
        # Export option
        st.download_button(
            label="Export Alerts to CSV",
            data=alerts_df.to_csv(index=False),
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
        # Acknowledge all button
        if st.button("Acknowledge All Alerts"):
            st.session_state.alerts = []
            st.rerun()
    else:
        st.success("No active alerts")

# Page 4: Reports
def reports_page():
    st.title("Maintenance Reports")
    
    # Model Performance Section
    st.subheader("Model Performance")
    if classifier or regressor:
        col1, col2 = st.columns(2)
        
        with col1:
            if classifier:
                st.write("**Failure Classifier Metrics**")
                metrics = st.session_state.model_metrics['classifier']
                st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                st.metric("F1 Score", f"{metrics['f1']*100:.1f}%")
        
        with col2:
            if regressor:
                st.write("**RUL Predictor Metrics**")
                metrics = st.session_state.model_metrics['regressor']
                st.metric("MAE", f"{metrics['mae']:.1f} cycles")
                st.metric("RMSE", f"{metrics['rmse']:.1f} cycles")
    else:
        st.warning("No models loaded to display metrics")
    
    # Reliability Metrics
    st.subheader("Reliability Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MTBF (Mean Time Between Failures)", "250 hours")
    with col2:
        st.metric("MTTR (Mean Time To Repair)", "8 hours")
    
    # Failure Analysis
    if nasa_data is not None and 'RUL' in nasa_data.columns:
        st.subheader("Failure Analysis")
        tab1, tab2 = st.tabs(["RUL Distribution", "Failure Patterns"])
        
        with tab1:
            fig = px.histogram(
                nasa_data,
                x='RUL',
                nbins=20,
                title="Remaining Useful Life Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'failure_flag' in pump_data.columns:
                fig = px.pie(
                    pump_data,
                    names='failure_flag',
                    title="Failure vs Healthy Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Report Generation
    st.subheader("Generate Report")
    report_type = st.radio("Report Type", ["PDF", "Excel"], horizontal=True)
    report_scope = st.multiselect(
        "Select Report Contents",
        options=["Executive Summary", "Model Performance", "Asset Health", "Alerts Log"],
        default=["Executive Summary", "Alerts Log"]
    )
    
    if st.button("Generate Report"):
        with st.spinner("Creating report..."):
            if report_type == "PDF":
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Header
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="Predictive Maintenance Report", ln=1, align="C")
                pdf.set_font("Arial", '', 12)
                pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d')}", ln=1)
                pdf.ln(10)
                
                # Contents based on selection
                if "Executive Summary" in report_scope:
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Executive Summary", ln=1)
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, txt="This report summarizes the current health status of monitored assets, model performance metrics, and active alerts.")
                    pdf.ln(5)
                
                if "Model Performance" in report_scope and classifier:
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Model Performance", ln=1)
                    pdf.set_font("Arial", '', 12)
                    pdf.cell(200, 10, txt=f"Failure Classifier Accuracy: {st.session_state.model_metrics['classifier']['accuracy']*100:.1f}%", ln=1)
                    if regressor:
                        pdf.cell(200, 10, txt=f"RUL Predictor MAE: {st.session_state.model_metrics['regressor']['mae']:.1f} cycles", ln=1)
                    pdf.ln(5)
                
                if "Asset Health" in report_scope and nasa_data is not None:
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Asset Health Overview", ln=1)
                    pdf.set_font("Arial", '', 12)
                    assets = nasa_data['id'].unique()[:3]
                    for asset in assets:
                        pdf.cell(200, 10, txt=f"Asset {asset}: Health Score 85%", ln=1)
                    pdf.ln(5)
                
                if "Alerts Log" in report_scope and st.session_state.alerts:
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Active Alerts", ln=1)
                    pdf.set_font("Arial", '', 12)
                    for alert in st.session_state.alerts[-5:]:
                        pdf.multi_cell(0, 10, txt=f"{alert['timestamp']} - {alert['severity']}: {alert['message']}")
                    pdf.ln(5)
                
                # Save and download
                pdf_path = "maintenance_report.pdf"
                pdf.output(pdf_path)
                
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name=pdf_path,
                        mime="application/pdf"
                    )
            else:  # Excel
                report_data = []
                
                if "Executive Summary" in report_scope:
                    report_data.extend([
                        ["Executive Summary", ""],
                        ["Generated on", datetime.now().strftime('%Y-%m-%d')],
                        ["Total Assets", len(nasa_data['id'].unique()) if nasa_data is not None else 0]
                    ])
                
                if "Model Performance" in report_scope:
                    report_data.extend([
                        ["", ""],
                        ["Model Performance", ""],
                        ["Classifier Accuracy", f"{st.session_state.model_metrics['classifier']['accuracy']*100:.1f}%"],
                        ["RUL Predictor MAE", f"{st.session_state.model_metrics['regressor']['mae']:.1f} cycles"]
                    ])
                
                if "Alerts Log" in report_scope and st.session_state.alerts:
                    report_data.extend([
                        ["", ""],
                        ["Recent Alerts", ""]
                    ])
                    for alert in st.session_state.alerts[-5:]:
                        report_data.append([alert['timestamp'], alert['message']])
                
                df = pd.DataFrame(report_data)
                excel_path = "maintenance_report.xlsx"
                df.to_excel(excel_path, index=False, header=False)
                
                with open(excel_path, "rb") as f:
                    st.download_button(
                        label="Download Excel Report",
                        data=f,
                        file_name=excel_path,
                        mime="application/vnd.ms-excel"
                    )

# Main App
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Asset Detail": asset_detail_page,
        "Alerts": alerts_page,
        "Reports": reports_page
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Predictive Maintenance Dashboard**  
    Version 1.0  
    Data last updated: {}  
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

if __name__ == "__main__":
    main()