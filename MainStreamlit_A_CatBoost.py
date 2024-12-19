import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Grain Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model_from_path():
    try:
        model = load_model(BestModel_googlenet_CatBoost.h5)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_data, class_names):
    try:
        # Convert to array and add batch dimension
        input_image_array = tf.keras.utils.img_to_array(image_data)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Make prediction
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(result)
        confidence = np.max(result) * 100

        return class_names[class_idx], confidence, result

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def create_plotly_chart(class_names, scores):
    # Create horizontal bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=[score * 100 for score in scores],
        y=class_names,
        orientation='h',
        text=[f'{score * 100:.1f}%' for score in scores],
        textposition='auto',
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5
            )
        )
    ))

    # Update layout
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability (%)',
        yaxis_title='Grain Type',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100])  # Set x-axis range from 0 to 100
    )

    return fig

def main():
    st.title("Multiple Grain Type Classification")
    st.write("Upload up to 10 images to classify the types of grain")
    
    # Load model and class names
    model = load_model_from_path()
    class_names = ['Beras', 'Gandum', 'Sorgum']
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return

    # Single file uploader that accepts multiple files
    uploaded_files = st.file_uploader(
        "Choose images (up to 10 files)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="multipleFiles"
    )

    # Check number of files
    if len(uploaded_files) > 10:
        st.warning("Maximum 10 files allowed. Only the first 10 will be processed.")
        uploaded_files = uploaded_files[:10]

    # Display number of uploaded files
    if uploaded_files:
        st.write(f"Number of files uploaded: {len(uploaded_files)}")

        if st.button('Predict All Images'):
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"---\n### Image {idx + 1}")
                
                col1, col2 = st.columns([1, 2])
                
                try:
                    # Ensure the file pointer is at the start
                    uploaded_file.seek(0)
                    
                    # Load image using PIL
                    image = Image.open(uploaded_file)
                    
                    # Create a copy of the image for prediction
                    img_copy = image.copy()
                    img_copy = img_copy.resize((180, 180))
                    
                    # Display original image
                    with col1:
                        st.image(image, caption=f'Image {idx + 1}: {uploaded_file.name}', use_column_width=True)
                    
                    # Get prediction
                    predicted_class, confidence, scores = predict_image(model, img_copy, class_names)
                    
                    with col2:
                        if predicted_class and confidence:
                            # Display results
                            st.success(f"Prediction: {predicted_class}")
                            st.info(f"Confidence: {confidence:.2f}%")
                            
                            # Display raw scores
                            st.write("Detailed predictions:")
                            for name, score in zip(class_names, scores):
                                st.write(f"{name}: {score*100:.2f}%")
                            
                            # Create and display Plotly chart
                            fig = create_plotly_chart(class_names, scores)
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing image {idx + 1}: {str(e)}")
                finally:
                    # Reset file pointer
                    uploaded_file.seek(0)

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.write("""
       aplikasi ini untuk klasifikasi :
        - Beras (Rice)
        - Gandum (Wheat)
        - Sorgum (Sorghum)
        
       
        """)

if __name__ == '__main__':
    main()
