"""
Streamlit Dashboard for Network Intrusion Detection System
Interactive web interface for SOC analysts to analyze network flows
"""

import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Helper Functions
# ============================================================================
def check_api_health():
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None

def get_model_info():
    """Get model metadata and performance metrics"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_feature_names():
    """Get list of expected feature names"""
    try:
        response = requests.get(f"{API_URL}/features", timeout=2)
        if response.status_code == 200:
            return response.json()["features"]
        return None
    except:
        return None

def predict_flow(features_dict):
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features_dict},
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def prepare_flow_features(df_row, feature_names):
    """Convert dataframe row to feature dict for API"""
    features_dict = {}
    for i, fname in enumerate(feature_names):
        if i < len(df_row):
            value = df_row.iloc[i]
            # Handle NaN/inf
            if pd.isna(value) or value == float('inf') or value == float('-inf'):
                value = 0.0
            features_dict[fname] = float(value)
        else:
            features_dict[fname] = 0.0
    return features_dict

# ============================================================================
# Sidebar - Model Status and Information
# ============================================================================
with st.sidebar:
    st.title("🛡️ NIDS")
    st.caption("Network Intrusion Detection System")

    st.divider()

    # API Health Check
    st.subheader("System Status")
    is_healthy, health_data = check_api_health()

    if is_healthy:
        st.success("✅ API Online")
        if health_data:
            st.metric("Model", "Loaded" if health_data.get("model_loaded") else "Not Loaded")
            st.metric("Scaler", "Loaded" if health_data.get("scaler_loaded") else "Not Loaded")
            st.metric("SHAP", "Loaded" if health_data.get("shap_loaded") else "Not Loaded")
    else:
        st.error("❌ API Offline")
        st.warning("Please start the API server:\n```bash\npython src/api/app.py\n```")

    st.divider()

    # Model Performance Metrics
    st.subheader("Model Performance")
    model_info = get_model_info()

    if model_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1 Score", f"{model_info['test_f1_score']:.3f}")
            st.metric("Precision", f"{model_info['test_precision']:.1%}")
        with col2:
            st.metric("Recall", f"{model_info['test_recall']:.1%}")
            st.metric("Status", model_info['production_status'])

        with st.expander("Model Details"):
            st.write(f"**Type:** {model_info['model_type']}")
            st.write(f"**Training Samples:** {model_info['training_samples']:,}")
            st.write(f"**Stability:** {model_info['stability']}")
            st.write(f"**Last Updated:** {model_info['last_updated']}")

    st.divider()

    # Quick Links
    st.subheader("Quick Links")
    st.markdown(f"[API Docs]({API_URL}/docs)")
    st.markdown(f"[Health Check]({API_URL}/health)")
    st.markdown("[GitHub](https://github.com)")

# ============================================================================
# Main Content - Tabs
# ============================================================================
st.title("Network Intrusion Detection System")
st.markdown("**Volumetric Attack Detection using One-Class SVM with SHAP Explainability**")

# Check if API is available
if not is_healthy:
    st.error("⚠️ **API is not running.** Please start the FastAPI server before using the dashboard.")
    st.code("python src/api/app.py", language="bash")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Flow Analysis", "Batch Analysis", "Sample Data"])

# ============================================================================
# Tab 1: Single Flow Analysis
# ============================================================================
with tab1:
    st.header("Analyze Individual Network Flow")
    st.markdown("Upload a CSV file with a single network flow (66 features) or paste feature values manually.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV with single flow",
        type=["csv"],
        key="single_flow",
        help="CSV should contain 66 network flow features in the correct order"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Show preview
            st.subheader("Flow Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Loaded {len(df.columns)} features, {len(df)} rows")

            # Get feature names
            feature_names = get_feature_names()

            if feature_names is None:
                st.error("Could not retrieve feature names from API")
            elif len(df.columns) != 66:
                st.warning(f"Expected 66 features, got {len(df.columns)}. Prediction may fail.")

            # Analyze button
            if st.button("🔍 Analyze Flow", type="primary", key="analyze_single"):
                with st.spinner("Analyzing flow..."):
                    # Prepare features
                    features_dict = prepare_flow_features(df.iloc[0], feature_names)

                    # Get prediction
                    success, result = predict_flow(features_dict)

                    if success:
                        # Display prediction result
                        st.divider()
                        st.subheader("Detection Result")

                        # Main prediction
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if result["prediction_label"] == "ATTACK":
                                st.error("🚨 **ATTACK DETECTED**")
                            else:
                                st.success("✅ **BENIGN TRAFFIC**")

                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")

                        with col3:
                            st.metric("Anomaly Score", f"{result['anomaly_score']:.6f}")

                        # Explanation
                        st.info(result["explanation"])

                        # Top Contributing Features (SHAP)
                        st.subheader("Top Contributing Features")
                        st.markdown("*Features that most influenced this prediction (powered by SHAP)*")

                        top_features = result["top_features"][:10]

                        # Create dataframe for display
                        features_df = pd.DataFrame([
                            {
                                "Rank": i + 1,
                                "Feature": feat["feature"],
                                "Value": f"{feat['feature_value']:.4f}",
                                "SHAP Contribution": f"{feat['shap_value']:.8f}",
                                "Importance": f"{feat['importance']:.8f}"
                            }
                            for i, feat in enumerate(top_features)
                        ])

                        st.dataframe(features_df, use_container_width=True, hide_index=True)

                        # Visualize top 5 features
                        st.subheader("Feature Importance Visualization")

                        top_5 = top_features[:5]
                        feature_names_chart = [f["feature"] for f in top_5]
                        shap_values_chart = [f["shap_value"] for f in top_5]

                        fig = go.Figure(go.Bar(
                            x=shap_values_chart,
                            y=feature_names_chart,
                            orientation='h',
                            marker=dict(
                                color=shap_values_chart,
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="SHAP Value")
                            )
                        ))

                        fig.update_layout(
                            title="Top 5 Features by SHAP Importance",
                            xaxis_title="SHAP Value",
                            yaxis_title="Feature",
                            height=400,
                            yaxis=dict(autorange="reversed")
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Model info
                        with st.expander("Model Information"):
                            st.json(result["model_info"])

                        # Raw response
                        with st.expander("Raw API Response"):
                            st.json(result)

                    else:
                        st.error(f"Prediction failed: {result}")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# ============================================================================
# Tab 2: Batch Analysis
# ============================================================================
with tab2:
    st.header("Analyze Multiple Network Flows")
    st.markdown("Upload a CSV file with multiple network flows for batch analysis.")

    # File upload
    uploaded_batch = st.file_uploader(
        "Upload CSV with multiple flows",
        type=["csv"],
        key="batch_flows",
        help="CSV should contain multiple rows, each with 66 network flow features"
    )

    if uploaded_batch:
        try:
            df_batch = pd.read_csv(uploaded_batch)

            # Show preview
            st.subheader("Batch Data Preview")
            st.dataframe(df_batch.head(10), use_container_width=True)
            st.caption(f"Loaded {len(df_batch)} flows with {len(df_batch.columns)} features each")

            # Get feature names
            feature_names = get_feature_names()

            if feature_names is None:
                st.error("Could not retrieve feature names from API")

            # Analyze button
            max_flows = min(len(df_batch), 1000)  # Limit to 1000 flows

            if st.button(f"🔍 Analyze Batch ({max_flows} flows)", type="primary", key="analyze_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []

                for idx in range(max_flows):
                    status_text.text(f"Analyzing flow {idx + 1}/{max_flows}...")
                    progress_bar.progress((idx + 1) / max_flows)

                    # Prepare features
                    features_dict = prepare_flow_features(df_batch.iloc[idx], feature_names)

                    # Get prediction
                    success, result = predict_flow(features_dict)

                    if success:
                        predictions.append({
                            "flow_id": idx,
                            "prediction": result["prediction_label"],
                            "confidence": result["confidence"],
                            "anomaly_score": result["anomaly_score"]
                        })
                    else:
                        predictions.append({
                            "flow_id": idx,
                            "prediction": "ERROR",
                            "confidence": 0,
                            "anomaly_score": 0
                        })

                progress_bar.empty()
                status_text.empty()

                # Display results
                st.divider()
                st.subheader("Batch Analysis Results")

                # Summary metrics
                attack_count = sum(1 for p in predictions if p["prediction"] == "ATTACK")
                benign_count = sum(1 for p in predictions if p["prediction"] == "BENIGN")
                error_count = sum(1 for p in predictions if p["prediction"] == "ERROR")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Flows", len(predictions))

                with col2:
                    st.metric("Attacks Detected", attack_count,
                             delta=f"{(attack_count/len(predictions)*100):.1f}%")

                with col3:
                    st.metric("Benign Traffic", benign_count,
                             delta=f"{(benign_count/len(predictions)*100):.1f}%")

                with col4:
                    if error_count > 0:
                        st.metric("Errors", error_count, delta="warning")

                # Results dataframe
                st.subheader("Detection Details")
                results_df = pd.DataFrame(predictions)

                # Add color coding
                def highlight_attacks(row):
                    if row['prediction'] == 'ATTACK':
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['prediction'] == 'BENIGN':
                        return ['background-color: #ccffcc'] * len(row)
                    else:
                        return ['background-color: #ffffcc'] * len(row)

                styled_df = results_df.style.apply(highlight_attacks, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Show detected attacks
                if attack_count > 0:
                    st.subheader("Detected Attacks")
                    attack_indices = [p["flow_id"] for p in predictions if p["prediction"] == "ATTACK"]
                    st.dataframe(df_batch.iloc[attack_indices], use_container_width=True)

                # Download results
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name="nids_batch_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")

# ============================================================================
# Tab 3: Sample Data
# ============================================================================
with tab3:
    st.header("Test with Sample Data")
    st.markdown("Use real CICIDS2017 data samples to test the detection system.")

    # Check if data exists
    wednesday_file = DATA_DIR / "Wednesday-workingHours.pcap_ISCX.csv"

    if not wednesday_file.exists():
        st.warning(f"Sample data not found at: {wednesday_file}")
        st.info("Please download CICIDS2017 dataset to use sample data.")
    else:
        st.success(f"✅ Sample data available: {wednesday_file.name}")

        # Load sample
        sample_type = st.radio(
            "Select sample type",
            ["Benign Traffic", "DoS Attack"],
            horizontal=True
        )

        if st.button("Load Sample", type="primary"):
            with st.spinner("Loading sample data..."):
                try:
                    df_sample = pd.read_csv(wednesday_file)

                    # Drop metadata
                    metadata_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label']
                    label_col = ' Label'

                    if label_col in df_sample.columns:
                        if sample_type == "Benign Traffic":
                            sample_df = df_sample[df_sample[label_col] == 'BENIGN'].head(1)
                        else:
                            sample_df = df_sample[df_sample[label_col].str.contains('DoS', na=False)].head(1)

                        # Drop metadata
                        for col in metadata_cols:
                            if col in sample_df.columns:
                                sample_df = sample_df.drop(col, axis=1)

                        # Show sample
                        st.subheader(f"Sample: {sample_type}")
                        st.dataframe(sample_df, use_container_width=True)

                        # Save for download
                        csv = sample_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Sample CSV",
                            data=csv,
                            file_name=f"sample_{sample_type.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )

                        st.info("💡 Download this sample and upload it in the 'Single Flow Analysis' tab to test the system.")

                except Exception as e:
                    st.error(f"Error loading sample: {str(e)}")

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption("Network Intrusion Detection System | Powered by One-Class SVM + SHAP | Built with Streamlit")
