import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from PIL import Image
import io

st.set_page_config(
    page_title="Grain Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_from_path():
    try:
        model = load_model(r'BestModel_googlenet_CatBoost.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_data, class_names):
    try:
        input_image_array = tf.keras.utils.img_to_array(image_data)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(result)
        confidence = np.max(result) * 100

        return class_names[class_idx], confidence, result

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def create_plotly_chart(class_names, scores):
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

    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability (%)',
        yaxis_title='Grain Type',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100]) 
    )

    return fig

def main():
    st.title("Multiple Grain Type Classification")
    st.write("Upload up to 10 images to classify the types of grain")
    
    model = load_model_from_path()
    class_names = ['Beras', 'Gandum', 'Sorgum']
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return

    uploaded_files = st.file_uploader(
        "Choose images (up to 10 files)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="multipleFiles"
    )

    if len(uploaded_files) > 10:
        st.warning("Maximum 10 files allowed. Only the first 10 will be processed.")
        uploaded_files = uploaded_files[:10]

    if uploaded_files:
        st.write(f"Number of files uploaded: {len(uploaded_files)}")

        if st.button('Predict All Images'):
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"---\n### Image {idx + 1}")
                
                col1, col2 = st.columns([1, 2])
                
                try:
                    uploaded_file.seek(0)
                    
                    image = Image.open(uploaded_file)
                    
                    img_copy = image.copy()
                    img_copy = img_copy.resize((180, 180))
                    
                    with col1:
                        st.image(image, caption=f'Image {idx + 1}: {uploaded_file.name}', use_container_width=True)
                    
                    predicted_class, confidence, scores = predict_image(model, img_copy, class_names)
                    
                    with col2:
                        if predicted_class and confidence:
                            st.success(f"Prediction: {predicted_class}")
                            st.info(f"Confidence: {confidence:.2f}%")
                            
                            st.write("Detailed predictions:")
                            for name, score in zip(class_names, scores):
                                st.write(f"{name}: {score*100:.2f}%")
                            
                            fig = create_plotly_chart(class_names, scores)
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing image {idx + 1}: {str(e)}")
                finally:
                    uploaded_file.seek(0)

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
