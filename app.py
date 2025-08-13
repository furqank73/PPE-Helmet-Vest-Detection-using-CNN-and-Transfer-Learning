import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import cv2
from io import BytesIO
import base64

# ---------------------- CONFIGURATION ----------------------
HISTORY_FILE = "detection_history.json"
SETTINGS_FILE = "app_settings.json"

# Default settings
DEFAULT_SETTINGS = {
    "theme": "dark",
    "primary_color": "#1f77b4",
    "model_path": "best_model.pth",
    "confidence_threshold": 0.7
}

# ---------------------- UTILITY FUNCTIONS ----------------------
def load_settings():
    """Load app settings from file or return defaults"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Ensure all required keys exist
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save app settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except:
        return False

def load_history():
    """Load detection history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_to_history(image_data, prediction, confidence, timestamp=None):
    """Save detection result to history"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Convert image to base64 for storage
    if hasattr(image_data, 'read'):
        image_data.seek(0)
        image_bytes = image_data.read()
        image_data.seek(0)
    else:
        # Handle PIL Image
        buffer = BytesIO()
        image_data.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
    
    image_b64 = base64.b64encode(image_bytes).decode()
    
    history = load_history()
    history.append({
        'timestamp': timestamp,
        'class': prediction,
        'confidence': confidence,
        'image_data': image_b64
    })
    
    # Keep only last 100 entries to manage file size
    if len(history) > 100:
        history = history[-100:]
    
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
        return True
    except:
        return False

def apply_custom_css():
    """Apply custom CSS styling based on current settings"""
    settings = st.session_state.get('settings', load_settings())
    
    if settings['theme'] == 'dark':
        bg_color = "#0e1117"
        secondary_bg = "#262730"
        text_color = "#ffffff"
        card_bg = "#1e1e1e"
    else:
        bg_color = "#ffffff"
        secondary_bg = "#f0f2f6"
        text_color = "#000000"
        card_bg = "#ffffff"
    
    primary_color = settings['primary_color']
    
    st.markdown(f"""
    <style>
    /* Main app styling */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Card styling */
    .metric-card {{
        background-color: {card_bg};
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {secondary_bg};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(90deg, {primary_color}, #1f77b4);
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }}
    
    /* Status indicators */
    .status-safe {{
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }}
    
    .status-unsafe {{
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }}
    
    /* History card styling */
    .history-card {{
        background-color: {card_bg};
        border: 1px solid {secondary_bg};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    
    .history-card:hover {{
        border-color: {primary_color};
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {secondary_bg};
    }}
    
    /* Button styling */
    .stButton > button {{
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, {primary_color}, #1f77b4);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background-color: {primary_color};
    }}
    
    /* Metric styling */
    [data-testid="metric-container"] {{
        background-color: {card_bg};
        border: 1px solid {secondary_bg};
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------- MODEL CLASS ----------------------
class SafetyClassifier:
    """PPE Safety Classification Model"""
    
    def __init__(self, model_path="best_model.pth"):
        self.class_names = ['Safe', 'Unsafe']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ResNet18 model architecture"""
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification
        self.model = self.model.to(self.device)

    def load_weights(self):
        """Load pre-trained weights"""
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                return True
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return False
        else:
            st.warning(f"Model file not found: {self.model_path}")
            return False

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, (str, BytesIO)):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor, img

    def predict(self, image_input):
        """Make prediction on input image"""
        try:
            img_tensor, original_img = self.preprocess_image(image_input)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            return {
                'class': self.class_names[predicted_class],
                'class_index': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy(),
                'image': original_img
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

@st.cache_resource
def load_model():
    """Load and cache the model"""
    settings = st.session_state.get('settings', load_settings())
    classifier = SafetyClassifier(model_path=settings['model_path'])
    classifier.load_weights()
    return classifier

# ---------------------- VISUALIZATION FUNCTIONS ----------------------
def create_confidence_gauge(confidence, prediction):
    """Create a Plotly gauge chart for confidence visualization"""
    settings = st.session_state.get('settings', load_settings())
    
    # Determine colors based on prediction
    if prediction == "Safe":
        color = "#00b894"
        bar_color = "green"
    else:
        color = "#e74c3c"
        bar_color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 20}},
        delta = {'reference': 70, 'increasing': {'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_probability_bar_chart(probabilities, class_names):
    """Create a bar chart showing class probabilities"""
    settings = st.session_state.get('settings', load_settings())
    
    colors = ['#00b894' if name == 'Safe' else '#e74c3c' for name in class_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Safety Class",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_history_chart(history_data):
    """Create a timeline chart of detection history"""
    if not history_data:
        return None
    
    df = pd.DataFrame(history_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Count detections by date and class
    summary = df.groupby(['date', 'class']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    if 'Safe' in summary.columns:
        fig.add_trace(go.Scatter(
            x=summary.index,
            y=summary['Safe'],
            mode='lines+markers',
            name='Safe',
            line=dict(color='#00b894', width=3),
            marker=dict(size=8)
        ))
    
    if 'Unsafe' in summary.columns:
        fig.add_trace(go.Scatter(
            x=summary.index,
            y=summary['Unsafe'],
            mode='lines+markers',
            name='Unsafe',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Detection History Timeline",
        xaxis_title="Date",
        yaxis_title="Number of Detections",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode='x unified'
    )
    
    return fig

# ---------------------- PAGE FUNCTIONS ----------------------
def home_page():
    """Render the home page"""
    st.header("Welcome to SafeVision AI")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <!-- Logo -->
        <img src="logo.png" alt="SafeVision AI Logo" style="width: 120px; margin-bottom: 10px;">
        <!-- Main Heading -->
        <h1 style="color: #1a3d5d; font-family: Arial, sans-serif;">🛡️ SafeVision AI</h1>
        <p style="font-size: 18px; color: #555;">Advanced PPE Detection & Safety Monitoring System</p>
        <!-- Divider -->
        <hr style="margin-top: 15px; border: 1px solid #ddd;">
        <!-- Author Info -->
        <p style="margin-top: 10px; font-size: 16px; color: #333;">
            Developed by <b>M Furqan Khan</b><br>
            <i>Machine Learning Engineer & Data Scientist</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

    
    # Key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Real-time Analysis</h3>
            <p>Instant PPE detection from images or live camera feed with high accuracy predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Smart Analytics</h3>
            <p>Comprehensive history tracking with visual analytics and exportable reports.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ AI-Powered</h3>
            <p>Advanced deep learning model trained on extensive PPE safety datasets.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    history = load_history()
    if history:
        st.subheader("📈 Quick Statistics")
        
        total_detections = len(history)
        safe_count = sum(1 for h in history if h['class'] == 'Safe')
        unsafe_count = total_detections - safe_count
        avg_confidence = np.mean([h['confidence'] for h in history])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Detections", total_detections)
        col2.metric("Safe Detections", safe_count, f"{safe_count/total_detections*100:.1f}%")
        col3.metric("Unsafe Detections", unsafe_count, f"{unsafe_count/total_detections*100:.1f}%")
        col4.metric("Avg. Confidence", f"{avg_confidence:.1%}")
        
        # History timeline chart
        timeline_fig = create_history_chart(history)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Getting started
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Upload an Image**: Go to the Analyze page and upload an image for PPE detection
    2. **Use Live Camera**: Capture real-time images using your device camera
    3. **View Results**: Get instant predictions with confidence scores and visual analytics
    4. **Track History**: Monitor all detections in the History page with filtering options
    5. **Customize Settings**: Adjust theme, colors, and model parameters in Settings
    """)

def analyze_page():
    """Render the analyze page"""
    st.header("🔍 PPE Safety Analysis")
    
    # Input method selection
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Camera"])
    
    uploaded_file = None
    
    with tab1:
        st.subheader("Upload Image for Analysis")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to analyze PPE safety compliance"
        )
    
    with tab2:
        st.subheader("Capture Live Image")
        camera_image = st.camera_input("Take a photo for analysis")
        if camera_image:
            uploaded_file = camera_image
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Image")
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Show loading spinner
            with st.spinner("Analyzing image... Please wait"):
                try:
                    # Load model and make prediction
                    model = load_model()
                    result = model.predict(uploaded_file)
                    
                    if result:
                        prediction = result['class']
                        confidence = result['confidence']
                        probabilities = result['probabilities']
                        
                        # Display prediction result
                        if prediction == "Safe":
                            st.markdown(f"""
                            <div class="status-safe">
                                ✅ SAFE ENVIRONMENT<br>
                                Confidence: {confidence:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="status-unsafe">
                                ⚠️ UNSAFE ENVIRONMENT<br>
                                Confidence: {confidence:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Save to history
                        timestamp = datetime.now().isoformat()
                        if save_to_history(uploaded_file, prediction, confidence, timestamp):
                            st.success("✅ Result saved to history")
                        else:
                            st.warning("⚠️ Could not save to history")
                        
                        # Show detailed metrics
                        col_metrics1, col_metrics2 = st.columns(2)
                        col_metrics1.metric("Prediction", prediction)
                        col_metrics2.metric("Confidence", f"{confidence:.1%}")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    result = None
        
        # Show visualizations below
        if uploaded_file is not None and 'result' in locals() and result:
            st.subheader("📊 Detailed Analysis")
            
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Confidence gauge
                gauge_fig = create_confidence_gauge(result['confidence'], result['class'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with chart_col2:
                # Probability bar chart
                bar_fig = create_probability_bar_chart(result['probabilities'], model.class_names)
                st.plotly_chart(bar_fig, use_container_width=True)
            
            # Additional insights
            st.subheader("💡 Analysis Insights")
            settings = st.session_state.get('settings', load_settings())
            threshold = settings.get('confidence_threshold', 0.7)
            
            if result['confidence'] >= threshold:
                confidence_level = "High"
                confidence_color = "green"
            elif result['confidence'] >= 0.5:
                confidence_level = "Medium"
                confidence_color = "orange"
            else:
                confidence_level = "Low"
                confidence_color = "red"
            
            st.markdown(f"""
            - **Confidence Level**: :{confidence_color}[{confidence_level}] ({result['confidence']:.1%})
            - **Model Device**: {model.device}
            - **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **Image Size**: {result['image'].size} pixels
            """)

def history_page():
    """Render the history page"""
    st.header("📚 Detection History")
    
    # Load history
    history = load_history()
    
    if not history:
        st.info("📝 No detection history found. Start analyzing images to build your history!")
        return
    
    # History controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        class_filter = st.selectbox(
            "Filter by Class",
            options=["All", "Safe", "Unsafe"],
            index=0
        )
    
    with col2:
        sort_order = st.selectbox(
            "Sort Order",
            options=["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"],
            index=0
        )
    
    with col3:
        items_per_page = st.selectbox(
            "Items per Page",
            options=[10, 25, 50, 100],
            index=1
        )
    
    with col4:
        # Download history as CSV
        if st.button("📥 Download CSV"):
            df = pd.DataFrame(history)
            df = df.drop('image_data', axis=1)  # Remove image data for CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download History CSV",
                data=csv,
                file_name=f"safevision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Filter and sort history
    filtered_history = history.copy()
    
    if class_filter != "All":
        filtered_history = [h for h in filtered_history if h['class'] == class_filter]
    
    # Sort history
    if sort_order == "Newest First":
        filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_order == "Oldest First":
        filtered_history.sort(key=lambda x: x['timestamp'])
    elif sort_order == "Highest Confidence":
        filtered_history.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_order == "Lowest Confidence":
        filtered_history.sort(key=lambda x: x['confidence'])
    
    # Pagination
    total_items = len(filtered_history)
    total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
    
    if total_pages > 1:
        page = st.selectbox(
            f"Page (Total: {total_pages})",
            options=list(range(1, total_pages + 1)),
            index=0
        )
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_items = filtered_history[start_idx:end_idx]
    else:
        page_items = filtered_history
        start_idx, end_idx = 0, total_items
    
    # Display summary
    st.subheader(f"📊 Summary ({total_items} total items, showing {len(page_items)})")
    
    if total_items > 0:
        safe_count = sum(1 for h in filtered_history if h['class'] == 'Safe')
        unsafe_count = total_items - safe_count
        avg_confidence = np.mean([h['confidence'] for h in filtered_history])
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Total", total_items)
        summary_col2.metric("Safe", safe_count, f"{safe_count/total_items*100:.1f}%")
        summary_col3.metric("Unsafe", unsafe_count, f"{unsafe_count/total_items*100:.1f}%")
        summary_col4.metric("Avg. Confidence", f"{avg_confidence:.1%}")
    
    # Display history items
    st.subheader(f"🗂️ History Items ({start_idx+1}-{end_idx} of {total_items})")
    
    for i, item in enumerate(page_items):
        with st.expander(
            f"{datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - "
            f"{item['class']} ({item['confidence']:.1%})"
        ):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image
                try:
                    image_data = base64.b64decode(item['image_data'])
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption=f"Detection #{len(history) - history.index(item)}", use_container_width=True)
                except:
                    st.error("Could not load image")
            
            with col2:
                # Display details
                status_class = "status-safe" if item['class'] == 'Safe' else "status-unsafe"
                st.markdown(f"""
                <div class="{status_class}">
                    {item['class'].upper()} - {item['confidence']:.1%} Confidence
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                **Timestamp**: {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}  
                **Class**: {item['class']}  
                **Confidence**: {item['confidence']:.3f} ({item['confidence']:.1%})  
                **Status**: {'✅ Safe Environment' if item['class'] == 'Safe' else '⚠️ Unsafe Environment'}
                """)
    
    # Clear history option
    st.subheader("🗑️ Data Management")
    if st.button("🚮 Clear All History", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            try:
                with open(HISTORY_FILE, 'w') as f:
                    json.dump([], f)
                st.success("✅ History cleared successfully!")
                st.session_state['confirm_clear'] = False
                st.rerun()
            except:
                st.error("❌ Failed to clear history")
        else:
            st.session_state['confirm_clear'] = True
            st.warning("⚠️ Click again to confirm history deletion")

def model_info_page():
    """Render the model information page"""
    st.header("🤖 Model Information")
    
    # Model Overview
    with st.expander("📋 Model Overview", expanded=True):
        st.markdown("""
        **SafeVision AI** uses a deep learning approach for Personal Protective Equipment (PPE) detection
        and safety compliance monitoring. The model is designed to classify workplace environments as
        either "Safe" or "Unsafe" based on visual analysis.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Architecture**: ResNet-18  
            **Framework**: PyTorch  
            **Input Size**: 224×224 pixels  
            **Classes**: 2 (Safe, Unsafe)  
            """)
        
        with col2:
            st.markdown("""
            **Output**: Binary classification  
            **Activation**: Softmax  
            **Device**: GPU/CPU compatible  
            **Inference Time**: ~50-200ms  
            """)
    
    # Technical Details
    with st.expander("⚙️ Technical Specifications"):
        st.markdown("""
        ### Model Architecture
        - **Base Model**: ResNet-18 (18-layer residual network)
        - **Pre-training**: ImageNet weights (optional)
        - **Custom Head**: Fully connected layer (512 → 2)
        - **Total Parameters**: ~11.2M
        
        ### Image Preprocessing
        1. **Resize**: Images resized to 224×224 pixels
        2. **Normalization**: ImageNet standard (RGB channels)
        3. **Tensor Conversion**: PIL Image → PyTorch Tensor
        4. **Device Transfer**: CPU/CUDA compatible
        
        ### Training Configuration
        - **Loss Function**: CrossEntropy Loss
        - **Optimizer**: Adam/SGD (configuration dependent)
        - **Learning Rate**: Adaptive scheduling
        - **Batch Size**: Variable (typically 16-32)
        """)

    # Dataset Information
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        ### Training Dataset
        The model has been trained on a comprehensive dataset of workplace safety images:
        
        - **Total Images**: Varies based on training configuration
        - **Class Distribution**: Balanced Safe/Unsafe samples
        - **Image Sources**: Workplace environments, construction sites, industrial settings
        - **Augmentation**: Rotation, flipping, color jittering, scaling
        
        ### Safety Classes
        - **Safe (Class 0)**: Proper PPE usage, compliant environments
        - **Unsafe (Class 1)**: Missing/improper PPE, hazardous conditions
        
        ### Data Quality Measures
        - Manual annotation and review
        - Cross-validation across different environments
        - Regular dataset updates and improvements
        """)
    
    # Model Performance
    with st.expander("📈 Performance Metrics"):
        # Create sample performance metrics visualization
        metrics_data = {
            'Metric': ['Accuracy', 'Precision (Safe)', 'Precision (Unsafe)', 'Recall (Safe)', 'Recall (Unsafe)', 'F1-Score'],
            'Value': [0.81, 0.74, 0.78, 0.80, 0.81, 0.85]
        }
        
        fig = px.bar(
            x=metrics_data['Value'],
            y=metrics_data['Metric'],
            orientation='h',
            title="Model Performance Metrics",
            labels={'x': 'Score', 'y': 'Metrics'},
            color=metrics_data['Value'],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note**: These are example metrics. Actual performance may vary based on 
        the specific model weights and training configuration used.
        """)
    
    # Usage Guidelines
    with st.expander("📖 Usage Guidelines"):
        st.markdown("""
        ### Best Practices for Accurate Detection
        
        #### Image Quality
        - **Resolution**: Use high-quality images (minimum 480×480 pixels)
        - **Lighting**: Ensure adequate lighting conditions
        - **Focus**: Images should be clear and in focus
        - **Angle**: Front-facing or side-view angles work best
        
        #### Subject Positioning
        - **Full Body**: Include full body or torso for PPE assessment
        - **Clear View**: Ensure PPE items are clearly visible
        - **Distance**: Subject should be 2-10 meters from camera
        - **Background**: Minimize cluttered backgrounds
        
        #### Environmental Factors
        - **Lighting**: Avoid extreme backlighting or shadows
        - **Weather**: Consider outdoor weather conditions
        - **Context**: Include relevant workplace context
        
        ### Limitations
        - Model accuracy may vary with unseen environments
        - Complex scenes with multiple subjects may be challenging
        - Extreme lighting conditions can affect performance
        - Model requires periodic retraining for optimal performance
        """)
    
    # Example Predictions
    with st.expander("🖼️ Example Predictions"):
        st.markdown("""
        ### Sample Detection Scenarios
        
        #### Safe Classifications ✅
        - Workers wearing hard hats and safety vests,
        - Proper use of fall protection equipment
        - Correct handling of safety equipment
        - Compliant workplace environments
        
        #### Unsafe Classifications ⚠️
        - Missing of one of the required PPE items or improperly worn hard hats
        - Lack of high-visibility clothing
        - Inadequate fall protection
        - Hazardous working conditions
        
        **Note**: Upload your own images in the Analyze section to see real predictions!
        """)
    
    # Model Status
    st.subheader("🔧 Current Model Status")
    
    try:
        settings = st.session_state.get('settings', load_settings())
        model_path = settings.get('model_path', 'best_model.pth')
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Status", "✅ Loaded")
            col2.metric("File Size", f"{file_size:.1f} MB")
            col3.metric("Last Modified", mod_time.strftime("%Y-%m-%d"))
            
            # Test model loading
            try:
                model = load_model()
                device_info = str(model.device)
                st.success(f"🚀 Model successfully loaded on {device_info}")
            except Exception as e:
                st.error(f"❌ Model loading error: {str(e)}")
        else:
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please check the model path in Settings or ensure the model file exists.")
    
    except Exception as e:
        st.error(f"❌ Error checking model status: {str(e)}")

def settings_page():
    """Render the settings page"""
    st.header("⚙️ Application Settings")
    
    # Load current settings
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    
    settings = st.session_state.settings
    
    # Theme Settings
    st.subheader("🎨 Theme & Appearance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_theme = st.selectbox(
            "Color Theme",
            options=["dark", "light"],
            index=0 if settings['theme'] == 'dark' else 1,
            help="Choose between dark and light theme"
        )
    
    with col2:
        color_options = {
            "Blue": "#1f77b4",
            "Green": "#2ca02c", 
            "Red": "#d62728",
            "Purple": "#9467bd",
            "Orange": "#ff7f0e",
            "Teal": "#17becf"
        }
        
        current_color_name = next((name for name, color in color_options.items() 
                                 if color == settings['primary_color']), "Blue")
        
        new_color_name = st.selectbox(
            "Primary Color",
            options=list(color_options.keys()),
            index=list(color_options.keys()).index(current_color_name),
            help="Choose the primary accent color"
        )
        new_primary_color = color_options[new_color_name]
    
    # Model Settings
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_model_path = st.text_input(
            "Model File Path",
            value=settings['model_path'],
            help="Path to the PyTorch model file (.pth)"
        )
    
    with col2:
        new_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=settings['confidence_threshold'],
            step=0.05,
            help="Minimum confidence threshold for predictions"
        )
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("""
        ### Cache Settings
        - **Model Caching**: Models are cached using Streamlit's caching system
        - **History Storage**: Detection history is stored locally in JSON format
        - **Settings Storage**: App settings are saved to local configuration file
        
        ### Performance Options
        - **Device Selection**: Automatic GPU/CPU detection
        - **Batch Processing**: Single image processing for optimal memory usage
        - **Image Preprocessing**: Standardized pipeline for consistent results
        """)
        
        if st.button("🗑️ Clear Model Cache"):
            st.cache_resource.clear()
            st.success("✅ Model cache cleared! Please reload the page.")
    
    # Save Settings
    st.subheader("💾 Save Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            # Update settings
            st.session_state.settings.update({
                'theme': new_theme,
                'primary_color': new_primary_color,
                'model_path': new_model_path,
                'confidence_threshold': new_threshold
            })
            
            # Save to file
            if save_settings(st.session_state.settings):
                st.success("✅ Settings saved successfully!")
                st.info("🔄 Please refresh the page to apply theme changes.")
            else:
                st.error("❌ Failed to save settings")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.session_state.settings = DEFAULT_SETTINGS.copy()
            if save_settings(st.session_state.settings):
                st.success("✅ Settings reset to defaults!")
                st.info("🔄 Please refresh the page to apply changes.")
            else:
                st.error("❌ Failed to reset settings")
    
    with col3:
        if st.button("📋 Export Settings"):
            settings_json = json.dumps(st.session_state.settings, indent=2)
            st.download_button(
                label="💾 Download Settings JSON",
                data=settings_json,
                file_name=f"safevision_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Import Settings
    st.subheader("📂 Import Configuration")
    uploaded_settings = st.file_uploader(
        "Upload Settings File",
        type=["json"],
        help="Import previously exported settings"
    )
    
    if uploaded_settings is not None:
        try:
            imported_settings = json.load(uploaded_settings)
            # Validate imported settings
            valid_keys = set(DEFAULT_SETTINGS.keys())
            imported_keys = set(imported_settings.keys())
            
            if valid_keys.issubset(imported_keys):
                st.session_state.settings = imported_settings
                if save_settings(st.session_state.settings):
                    st.success("✅ Settings imported successfully!")
                    st.info("🔄 Please refresh the page to apply changes.")
                else:
                    st.error("❌ Failed to save imported settings")
            else:
                st.error("❌ Invalid settings file format")
        except Exception as e:
            st.error(f"❌ Error importing settings: {str(e)}")
    
    # Current Settings Preview
    with st.expander("👀 Current Settings Preview"):
        st.json(st.session_state.settings)

# ---------------------- MAIN APPLICATION ----------------------
def main():
    """Main application function"""
    
    # Initialize settings in session state
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Configure page
    st.set_page_config(
        page_title="SafeVision AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🛡️ SafeVision AI")
        st.markdown("---")
        
        page = st.selectbox(
            "Navigate to:",
            options=["🏠 Home", "🔍 Analyze", "📚 History", "🤖 Model Info", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        history = load_history()
        if history:
            st.subheader("📊 Quick Stats")
            total = len(history)
            safe = sum(1 for h in history if h['class'] == 'Safe')
            st.metric("Total Detections", total)
            st.metric("Safe Rate", f"{safe/total*100:.1f}%" if total > 0 else "0%")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; font-size: 12px;'>
            SafeVision AI v2.0<br>
            Advanced PPE Detection System
        </div>
        """, unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "🔍 Analyze":
        analyze_page()
    elif page == "📚 History":
        history_page()
    elif page == "🤖 Model Info":
        model_info_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()