import streamlit as st
import cv2
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
from datetime import datetime

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Advanced Deepfake Detector", page_icon="🎥")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stVideo {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    .header-container {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(120deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .analysis-header {
        background-color: #eef2f7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource()
def load_model():
    processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    return pipeline("image-classification", model=model, feature_extractor=processor)

pipe = load_model()

def calculate_advanced_metrics(real_scores, fake_scores, real_count, fake_count):
    total_frames = len(real_scores)
    metrics = {
        'confidence_volatility': np.std(real_scores + fake_scores),
        'avg_real_confidence': np.mean(real_scores),
        'avg_fake_confidence': np.mean(fake_scores),
        'detection_stability': 1 - (np.std(real_scores) / np.mean(real_scores)) if np.mean(real_scores) != 0 else 0,
        'frame_consistency': abs(real_count - fake_count) / total_frames if total_frames > 0 else 0
    }
    return metrics

def process_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    real_scores = []
    fake_scores = []
    frame_indices = []
    real_count, fake_count = 0, 0
    threshold = 0.6

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frame_classifications = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = pipe(pil_image)
            
            if result:
                real_score = next((x["score"] for x in result if x["label"] == "Real"), 0)
                fake_score = next((x["score"] for x in result if x["label"] == "Fake"), 0)

                real_scores.append(real_score)
                fake_scores.append(fake_score)
                frame_indices.append(frame_count)

                label = "Real" if real_score > fake_score and real_score >= threshold else "Fake"
                score = max(real_score, fake_score)
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)

                # Create overlay for better visualization
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (300, 90), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, f"{label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {score:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                frame_classifications.append({
                    'frame': frame_count,
                    'label': label,
                    'confidence': score,
                    'timestamp': frame_count/fps if fps > 0 else 0
                })

                frames.append(cv2.resize(frame, (150, 100)))

                if label == "Real":
                    real_count += 1
                else:
                    fake_count += 1

        frame_count += 1

    cap.release()
    video_metadata = {
        'fps': fps,
        'duration': duration,
        'total_frames': total_frames
    }
    return frames, real_scores, fake_scores, frame_indices, real_count, fake_count, frame_classifications, video_metadata

# Streamlit UI
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.title("🎥 Advanced Deepfake Video Detection")
st.markdown("Professional deep learning-based video analysis for deepfake detection")
st.markdown('</div>', unsafe_allow_html=True)

uploaded_video = st.file_uploader("Upload a video for analysis", type=["mp4", "avi", "mov","png","jpg","jpeg"])

if uploaded_video:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="analysis-header">', unsafe_allow_html=True)
        st.subheader("📼 Input Video")
        st.markdown('</div>', unsafe_allow_html=True)
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.video(video_path)

    if st.button("🔍 Analyze Video", use_container_width=True):
        with st.spinner("🔄 Processing video... This may take a few minutes depending on the video length."):
            frames, real_scores, fake_scores, frame_indices, real_count, fake_count, frame_classifications, video_metadata = process_video(video_path, frame_skip=30)
            advanced_metrics = calculate_advanced_metrics(real_scores, fake_scores, real_count, fake_count)

        # Sample Frames Analysis
        st.markdown("### 🖼️ Frame Analysis")
        frame_tabs = st.tabs(["Sample Frames", "Frame Timeline", "Confidence Distribution"])
        
        with frame_tabs[0]:
            st.image(frames[:5], channels="BGR", caption=[f"Frame {i}" for i in range(1, 6)])
            
        with frame_tabs[1]:
            # Timeline visualization
            timeline_data = pd.DataFrame(frame_classifications)
            if not timeline_data.empty:
                fig = px.scatter(timeline_data, x='timestamp', y='confidence',
                               color='label', title='Detection Confidence Timeline',
                               labels={'timestamp': 'Time (seconds)', 'confidence': 'Confidence Score'})
                st.plotly_chart(fig, use_container_width=True)
                
        with frame_tabs[2]:
            # Confidence distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.kdeplot(data=real_scores, label='Real', color='green')
            sns.kdeplot(data=fake_scores, label='Fake', color='red')
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Density')
            st.pyplot(fig)

        # Results Dashboard
        st.markdown("### 📊 Analysis Dashboard")
        
        # Key Metrics
        metric_cols = st.columns(4)
        total_frames = len(real_scores)
        
        if total_frames > 0:
            real_percentage = (real_count / total_frames) * 100
            fake_percentage = (fake_count / total_frames) * 100
            
            with metric_cols[0]:
                st.metric("Real Frames", f"{real_percentage:.1f}%", f"{real_count} frames")
            with metric_cols[1]:
                st.metric("Fake Frames", f"{fake_percentage:.1f}%", f"{fake_count} frames")
            with metric_cols[2]:
                st.metric("Model Confidence", f"{advanced_metrics['avg_real_confidence']*100:.1f}%")
            with metric_cols[3]:
                st.metric("Detection Stability", f"{advanced_metrics['detection_stability']*100:.1f}%")

        # Advanced Analytics
        st.markdown("### 🔬 Advanced Analytics")
        advanced_cols = st.columns(2)
        
        with advanced_cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Technical Metrics")
            st.markdown(f"- **Video Duration:** {video_metadata['duration']:.2f} seconds")
            st.markdown(f"- **Frame Rate:** {video_metadata['fps']:.1f} FPS")
            st.markdown(f"- **Confidence Volatility:** {advanced_metrics['confidence_volatility']:.3f}")
            st.markdown(f"- **Frame Consistency:** {advanced_metrics['frame_consistency']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with advanced_cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Detection Summary")
            st.markdown(f"- **Average Real Confidence:** {advanced_metrics['avg_real_confidence']:.3f}")
            st.markdown(f"- **Average Fake Confidence:** {advanced_metrics['avg_fake_confidence']:.3f}")
            st.markdown(f"- **Total Frames Analyzed:** {total_frames}")
            st.markdown(f"- **Analysis Coverage:** {(total_frames/video_metadata['total_frames']*100):.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Final Verdict
        st.markdown("### 🎯 Final Verdict")
        final_result = "FAKE" if fake_count > (real_count * 1.2) else "REAL"
        verdict_color = "#ff4b4b" if final_result == "FAKE" else "#00cc66"
        
        st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: {verdict_color}; 
                 color: white; border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2>Video Classification: {final_result}</h2>
                <p>Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a video file to begin the analysis")
    
# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Using advanced deep learning models for video analysis and deepfake detection.</p>
    </div>
""", unsafe_allow_html=True)