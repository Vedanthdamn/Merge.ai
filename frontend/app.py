"""
Streamlit Frontend for Healthcare ML Model - DEMO ONLY

IMPORTANT:
This frontend is for DEMONSTRATION purposes only. It:
- Accepts manual patient feature input
- Displays prediction results
- Shows model accuracy (if available)

This frontend does NOT:
- Participate in model training
- Access raw hospital data
- Store any patient information
- Replace clinical decision-making

WARNING:
This is a research prototype. NOT for actual clinical use.
Always consult qualified healthcare professionals for medical decisions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from baseline.model import BaselineHealthcareModel
except ImportError:
    st.error("Could not import baseline model. Make sure the project structure is correct.")


def load_model(model_path: str = "models/baseline_model.h5"):
    """
    Load trained model.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        Loaded model or None
    """
    if not os.path.exists(model_path):
        return None
    
    try:
        model = BaselineHealthcareModel()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_feature_inputs(n_features: int = 10) -> List[float]:
    """
    Create input widgets for patient features.
    
    This is a generic interface for any healthcare dataset.
    In practice, features would have specific names and meanings.
    
    Args:
        n_features (int): Number of features to input
        
    Returns:
        List of feature values
    """
    st.subheader("Patient Features")
    st.info("""
    **Demo Mode**: Enter generic numerical features.
    
    In a real deployment:
    - Features would have specific medical meanings (age, blood pressure, lab values, etc.)
    - Input validation would ensure clinical plausibility
    - Units and ranges would be clearly specified
    """)
    
    features = []
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    for i in range(n_features):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            value = st.number_input(
                f"Feature {i+1}",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"feature_{i}"
            )
            features.append(value)
    
    return features


def display_prediction(prediction: float, threshold: float = 0.5):
    """
    Display prediction result.
    
    Args:
        prediction (float): Prediction probability
        threshold (float): Classification threshold
    """
    st.subheader("Prediction Result")
    
    # Display probability
    st.metric("Predicted Probability", f"{prediction:.2%}")
    
    # Display classification
    predicted_class = "Positive" if prediction >= threshold else "Negative"
    color = "🔴" if prediction >= threshold else "🟢"
    
    st.markdown(f"### {color} Classification: **{predicted_class}**")
    
    # Confidence indicator
    confidence = abs(prediction - 0.5) * 2  # 0 = uncertain, 1 = very confident
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.1%}")
    
    # Important disclaimer
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**
    
    This prediction is from a research prototype and should NOT be used for:
    - Clinical diagnosis
    - Treatment decisions
    - Medical advice
    
    Always consult qualified healthcare professionals for medical decisions.
    """)


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Healthcare ML Demo",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title and description
    st.title("🏥 Privacy-Preserving Healthcare ML Demo")
    st.markdown("---")
    
    # Sidebar information
    with st.sidebar:
        st.header("About This Demo")
        st.info("""
        **Research Prototype**
        
        This interface demonstrates:
        - Manual patient feature input
        - Binary classification prediction
        - Model inference (NOT training)
        
        **Privacy Features:**
        - No data is stored
        - No connection to hospital databases
        - Demonstration purposes only
        """)
        
        st.markdown("---")
        
        st.header("System Components")
        st.markdown("""
        ✓ Baseline ML Model  
        ✓ Federated Learning  
        ✓ Split Learning  
        ✓ Differential Privacy  
        ✓ SMPC  
        ✓ Blockchain Audit  
        """)
        
        st.markdown("---")
        
        st.warning("""
        **NOT FOR CLINICAL USE**
        
        This is a research prototype.
        Do not use for actual medical decisions.
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Model Inference")
        
        # Check if model exists
        model_path = "models/baseline_model.h5"
        model_exists = os.path.exists(model_path)
        
        if not model_exists:
            st.warning("""
            **No trained model found.**
            
            To use this demo:
            1. Train the baseline model first: `python src/baseline/model.py`
            2. Ensure the model is saved to `models/baseline_model.h5`
            3. Reload this page
            
            For demonstration purposes, you can still input features below.
            """)
        
        # Feature input section
        st.markdown("### Enter Patient Features")
        
        # Allow user to set number of features
        n_features = st.number_input(
            "Number of Features",
            min_value=1,
            max_value=50,
            value=10,
            help="Set this to match your trained model's input dimension"
        )
        
        features = create_feature_inputs(n_features)
        
        # Predict button
        if st.button("Make Prediction", type="primary"):
            if model_exists:
                try:
                    # Load model
                    with st.spinner("Loading model..."):
                        model = load_model(model_path)
                    
                    if model is not None:
                        # Make prediction
                        with st.spinner("Computing prediction..."):
                            X_input = np.array([features])
                            prediction = model.predict(X_input)[0][0]
                        
                        # Display result
                        display_prediction(prediction)
                    else:
                        st.error("Failed to load model.")
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Make sure the number of features matches the trained model.")
            else:
                # Mock prediction for demonstration
                st.info("Demo mode: Generating mock prediction (no actual model loaded)")
                mock_prediction = np.random.rand()
                display_prediction(mock_prediction)
    
    with col2:
        st.header("Model Info")
        
        if model_exists:
            st.success("✓ Model loaded successfully")
            
            # Display model metrics (mock data for demo)
            st.subheader("Model Performance")
            st.metric("Training Accuracy", "N/A")
            st.metric("Test Accuracy", "N/A")
            st.metric("AUC", "N/A")
            
            st.caption("Run evaluation to see actual metrics")
        else:
            st.error("✗ Model not found")
            st.caption(f"Expected at: {model_path}")
        
        st.markdown("---")
        
        st.subheader("Privacy Guarantees")
        st.markdown("""
        🔒 **Data Privacy:**
        - Training data stays at hospitals
        - Only model parameters shared
        - Differential privacy applied
        
        🔗 **Audit Trail:**
        - Blockchain logging enabled
        - Transparent training history
        - Immutable records
        
        🤝 **Collaborative:**
        - Federated learning across hospitals
        - No central data repository
        - Fair contribution weighting
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Privacy-Preserving Distributed Healthcare ML Framework</p>
        <p>Research Prototype - Not for Clinical Use</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
