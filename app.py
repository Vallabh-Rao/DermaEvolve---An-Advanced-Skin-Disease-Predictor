from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
from PIL import Image
import openai
import time
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from tensorflow import lite
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
import pandas as pd
import seaborn as sns
import sqlite3
from datetime import datetime
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

dataset_dir = "Dataset_Sample/Sample_Dataset_DE"
image_size = (64, 64)
batch_size = 32
num_classes = len(os.listdir(dataset_dir))

def predict_image_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data, axis=1)[0]
    
    class_labels = list(val_generator.class_indices.keys())
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, predicted_class_index

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

st.set_page_config(layout="wide", page_title="DermaEvolve - An Advanced Skin Disease Predictor", page_icon="😷")

import os
if not os.path.exists("images"):
    os.makedirs("images")

def init_db():
    conn = sqlite3.connect("dermaevolve.db")
    cursor = conn.cursor()
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT,
            age INTEGER,
            location TEXT
        )
    """)
    # Create activity table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            image_path TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
init_db()
# Register a new user
def register(name, username, email, password, age, location):
    try:
        conn = sqlite3.connect("dermaevolve.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, username, email, password, age, location)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, username, email, password, age, location))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Login user
def login(username, password):
    conn = sqlite3.connect("dermaevolve.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM users WHERE username=? AND password=?
    """, (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Save prediction activity
def save_activity(username, image_path, prediction):
    conn = sqlite3.connect("dermaevolve.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO user_activity (username, image_path, prediction, timestamp)
        VALUES (?, ?, ?, ?)
    """, (username, image_path, prediction, timestamp))
    conn.commit()
    conn.close()

# Display user activity
def display_saved_activities(username):
    st.title(f"📊 Monitor Your Activity, {st.session_state.global_username}")

    conn = sqlite3.connect("dermaevolve.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, image_path, prediction, timestamp FROM user_activity WHERE username=?
    """, (username,))
    activities = cursor.fetchall()
    conn.close()

    if activities:
        for activity in activities:
            activity_id, image_path, prediction, timestamp = activity
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(image_path, caption=f"Prediction: {prediction}\nTime: {timestamp}", use_container_width =True)
            with col2:
                if st.button("Delete", key=f"delete_{activity_id}"):
                    delete_activity(activity_id)
                    st.success("Activity deleted successfully!")
                    st.session_state["rerun"] = not st.session_state.get("rerun", False)

    else:
        st.info("No activity found.")

def delete_activity(activity_id):
    conn = sqlite3.connect("dermaevolve.db")
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM user_activity WHERE id=?
    """, (activity_id,))
    conn.commit()
    conn.close()

if "global_username" not in st.session_state:
    st.session_state.global_username = None
    
def close_sidebar_on_select():
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = True

    if st.session_state.sidebar_open:
        st.session_state.sidebar_open = False

close_sidebar_on_select()

with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options = ["Home", "Predict A Disease", "Your Activity", "Analytics", "About Us", "Terms And Conditions"],
        icons = ["house-door", "person", "camera", "bar-chart", "info-circle", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

if page == "Home":
    
    st.markdown(
        """
        <style>
                .card {
                    width: 100%;
                    background-color: #ffffff;
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                    transition: transform 0.3s ease-in-out;
                    animation: wiggle 2s infinite ease-in-out; /* Wiggle effect */
                }

                .card:hover {
                    transform: translateY(-15px) scale(1.05); /* Lift and scale on hover */
                    animation: bounce 0.6s; /* Quick bounce on hover */
                }

                .card-header {
                    background-color: aquamarine;
                    padding: 20px;
                    text-align: center;
                    font-size: 2em;
                    font-weight: bold;
                    color: black;
                }

                .card-body {
                    padding: 20px;
                    font-size: 1.5em;
                    color: #333;
                    text-align: center;
                    font-weight: bold;
                }

                .card-footer {
                    padding: 15px;
                    background-color: #f1f1f1;
                    text-align: center;
                    border-radius: 0 0 15px 15px;
                }

                /* Keyframes for cartoonish wiggle */
                @keyframes wiggle {
                    0%, 100% {
                        transform: rotate(-3deg);
                    }
                    50% {
                        transform: rotate(3deg);
                    }
                }

                /* Keyframes for bounce effect */
                @keyframes bounce {
                    0%, 100% {
                        transform: translateY(9);
                    }
                    50% {
                        transform: translateY(-70px);
                    }
                }
            </style>

        <div class="card">
            <div class="card-header">
            DermaEvolve
            </div>
            <div class="card-body">
            Your Skin, Our Expertise..! <br>
            Precise Predictions At Your Fingertips.
            </div>
        </div>

        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <style>
        /* Container styling */
        .header-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 0 1px 15px rgba(255, 94, 77, 0.8);
            text-align: justify;
        }

        .subheader-title {
            font-size: 1.5em;
            color: #ffffff;
            font-weight: 300;
            text-shadow: 0 0 10px rgba(0, 136, 255, 0.5);
            text-align: justify;
        }

        /* Justified text styling */
        .intro-text, {
            font-size: 1.2em;
            color: #ffffff;
            line-height: 1.6;
            margin-top: 20px;
            text-align: justify;
        }

        /* Model Information Box Styling */
        .model-info {
            background-color: #6affc3;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(111, 111, 111, 0.1);
        }

        .model-info h4 {
            font-size: 1.3em;
            color: #001f3d;
            margin-bottom: 15px;
        }

        /* Highlight text */
        .highlight-text {
            color: #fb0000;
            font-weight: 600;
        }

        /* List styling */
        ul {
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="intro-text" style="text-align: justify;">DermaEvolve leverages cutting-edge machine learning models to accurately predict and diagnose various skin diseases using just an image. This innovative tool is designed to assist healthcare professionals and individuals alike in identifying skin conditions early, aiding in better treatment decisions and outcomes.</p>', unsafe_allow_html=True)

    st.markdown('<p class="intro-text" style="text-align: justify;">The heart of DermaEvolve is built on advanced deep learning models that have been trained on large-scale dermatological datasets. Here’s a closer look at the models that power the predictions:</p>', unsafe_allow_html=True)

    
    slider_value = st.slider("", min_value=1, max_value=4, value=1, step=1)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    if slider_value == 1:
        with col1:
            st.markdown(
                """
                <div style="background-color: rgb(237, 237, 36); width: 20em; height: auto; color:black; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>Customized CNN Model</h3>
                    <p style="text-align: justify;">This model is specifically tailored for skin disease prediction, leveraging a customized CNN architecture.</p>
                    <p><strong>Accuracy: 82.70%</strong></p>
                </div>
                """, unsafe_allow_html=True)
    elif slider_value == 2:
        with col2:
            st.markdown(
                """
                <div style="background-color: rgb(160, 237, 36); width: 20em; height: auto; color:black; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>MobileNet</h3>
                    <p style="text-align: justify;">MobileNet is a lightweight neural network model faster and efficient for complex predictions, making it the most flexible fine-tuned model.</p>
                    <p><strong>Accuracy: 66.00%</strong></p>
                </div>
                """, unsafe_allow_html=True)
    elif slider_value == 3:
        with col3:
            st.markdown(
                """
                <div style="background-color: rgb(54, 252, 225); width: 20em; height: auto; color:black; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>DenseNet-169</h3>
                    <p style="text-align: justify;">DenseNet-169 is a deep convolutional network that improves performance by connecting each layer to every other layer in a feed-forward fashion.</p>
                    <p><strong>Accuracy: 75.71%</strong></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        with col4:
            st.markdown(
                """
                <div style="background-color: rgb(245, 104, 104); width: 20em; height: auto; color:black; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>NASNet</h3>
                    <p style="text-align: justify;">NASNet is a neural architecture search-based model designed for optimal performance on image classification tasks like skin disease detection.</p>
                    <p><strong>Accuracy: 62.76%</strong></p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="intro-text" style="text-align: justify;">Our models are trained using high-quality, labeled dermatological datasets, and we leverage various techniques such as SMOTE-based augmentation to address data imbalances, improving model robustness and accuracy.</p>', unsafe_allow_html=True)

    
    st.subheader("Key Features of DermaEvolve")
    st.markdown("""
        <ul class="intro-text" style="text-align: justify;">
            <li><b>Real-time Skin Disease Prediction:</b> Using deep learning models for fast and accurate diagnosis.</li>
            <li><b>Mobile-Friendly:</b> With models optimized for mobile use, DermaEvolve can run seamlessly on Android devices.</li>
            <li><b>Wide Range of Diseases:</b> Capable of predicting a variety of skin conditions, including melanoma, basal cell carcinoma, and more.</li>
            <li><b>AI-Driven Image Description:</b> Provides insightful descriptions of skin lesions along with disease predictions.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.subheader("How It Works ?")
    
    st.markdown('<p class="intro-text" style="text-align: justify;">DermaEvolve allows users to upload an image of a skin lesion or mole, and within seconds, it returns a prediction of the skin condition. The system analyzes features like color, texture, shape, and size to determine the disease, providing a reliable diagnosis.</p>', unsafe_allow_html=True)

    footer = """
    <style>
    footer {
        content: "📘 For educational purposes only";
        visibility: visible;
    }
    
    footer:after {
        
        visibility: visible;
        display: block;
        position: relative;
        color: gray;
        font-size: 14px;
        text-align: center;
        padding: 10px;
    }
    </style>
    """
    
    st.markdown(footer, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user = login(login_username, login_password)
            if user:
                st.session_state.global_username = login_username
                st.success(f"Welcome, {login_username}! Active Session Initiated. Exit App to LogOut Automatically.")
                st.session_state["rerun"] = not st.session_state.get("rerun", False)
            else:
                st.error("Invalid username or password.")

    # Register Tab
    with tab2:
        st.subheader("Register")
        name = st.text_input("Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        age = st.number_input("Age", min_value=0, step=1)
        location = st.text_input("Location")
        if st.button("Register"):
            if register(name, username, email, password, age, location):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists. Please choose a different one.")

        
elif page == "Predict A Disease":
    if st.session_state.global_username:
        st.title("Predict A Disease")
        st.warning(" Image Upload Instructions")
        st.markdown("""
        - **Image should be clear and focused** to ensure accurate classification.
        - Before selecting a model, Please refer to the ***ANALYTICS*** section for more info about precision and accuracy of all our models.
        - **Preferred format**: .jpg, .jpeg, .png (max 200MB).
        - Ensure the skin lesion or area is well-lit and captured without obstructions.
        - If using the camera to capture, make sure the image is in focus and taken from an appropriate distance.
        - Try capturing the image only with skin lesion, eliminating the background or unwanted details, considering the sensitiveness of the models.
        """)
        
        
        
        model_paths = {
            "MobileNet": "Android_Compatible_models/MobileNet.tflite",
            "DenseNet169": "Android_Compatible_models/DenseNet_169.tflite",
            "Custom CNN": "Android_Compatible_models/CNN_Customized.tflite",
            "ResNet50": "Android_Compatible_models/ResNet_50.tflite",
            "NasNet": "Android_Compatible_models/NasNet.tflite"
        }

        st.markdown(
            """
            <style>
                .title {
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 24px;
                    text-align: center;
                }
                .subtitle {
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 18px;
                    text-align: center;
                }
                .warning {
                    color: #ff0000;
                    font-size: 16px;
                    font-weight: normal;
                }
            </style>
            """, unsafe_allow_html=True
        )

        st.markdown('<p class="title">Choose image input method</p>', unsafe_allow_html=True)

        def load_model(model_path):
            interpreter = lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        
        
        def predict_image(interpreter, image):
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            image = image.resize((64, 64))
            image_array = np.array(image).astype(np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], image_array)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            return np.argmax(output_data, axis=1)[0]

        image_option = st.radio("", ("Upload Image 🖼️", "Capture Image 📷"))
        image = None

        if image_option == "Upload Image 🖼️":
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
        elif image_option == "Capture Image 📷":
            uploaded_image = st.camera_input("Capture Image")
            if uploaded_image is not None:
                image = Image.open(uploaded_image)

        if image is not None:
            image = image.resize((64, 64))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            image_path = f"images/uploaded_images_{timestamp}.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            selected_model = st.selectbox("Select Model", ["MobileNet", "DenseNet169", "Custom CNN", "ResNet50", "NasNet"])
            model_path = model_paths[selected_model]

            start_time = time.time()
            
            interpreter = load_tflite_model(model_path)
            predicted_class_label, predicted_class_index = predict_image_tflite(interpreter, img_array)
            
            end_time = time.time()
            
            st.markdown(f'<p class="subtitle">Predicted Class: <strong style="color: yellow;">{predicted_class_label}</strong></p>', unsafe_allow_html=True)
            
            disease_info = {
                "Actinic Keratosis": {
                    "name": "Actinic Keratosis",
                    "symptoms": "Rough, scaly patches on the skin, often in areas exposed to the sun.",
                    "treatment": "Cryotherapy, topical treatments (e.g., fluorouracil), or photodynamic therapy.",
                },
                "Basal Cell Carcinoma": {
                    "name": "Basal Cell Carcinoma",
                    "symptoms": "Shiny or pearly bumps, open sores that don’t heal, or reddish patches.",
                    "treatment": "Surgical removal, Mohs surgery, or radiation therapy.",
                },
                "Blue Naevus": {
                    "name": "Blue Naevus",
                    "symptoms": "Benign blue or bluish-black moles often found on the face, hands, or feet.",
                    "treatment": "Typically no treatment is required unless changes are noted.",
                },
                "Dermatofibroma": {
                    "name": "Dermatofibroma",
                    "symptoms": "Firm, small nodules on the skin, usually painless.",
                    "treatment": "Surgical removal if bothersome.",
                },
                "Elastosis Perforans Serpiginosa": {
                    "name": "Elastosis Perforans Serpiginosa",
                    "symptoms": "Raised, ring-shaped lesions often found on the neck or arms.",
                    "treatment": "Topical retinoids or corticosteroids; cryotherapy in severe cases.",
                },
                "Lentigo Maligna": {
                    "name": "Lentigo Maligna",
                    "symptoms": "Flat, dark patches that grow slowly, typically on sun-exposed skin.",
                    "treatment": "Surgical excision, Mohs surgery, or laser therapy.",
                },
                "Melanocytic Nevus": {
                    "name": "Melanocytic Nevus",
                    "symptoms": "Common moles that are usually brown or skin-colored.",
                    "treatment": "Generally no treatment unless atypical or changing.",
                },
                "Melanoma": {
                    "name": "Melanoma",
                    "symptoms": "Asymmetrical moles with irregular borders and color variations.",
                    "treatment": "Surgical removal, immunotherapy, chemotherapy, or radiation therapy.",
                },
                "Nevus Sebaceus": {
                    "name": "Nevus Sebaceus",
                    "symptoms": "Yellowish patches on the scalp or face, often present at birth.",
                    "treatment": "Surgical removal if changes occur or for cosmetic reasons.",
                },
                "Pigmented Benign Keratosis": {
                    "name": "Pigmented Benign Keratosis",
                    "symptoms": "Dark, waxy, or warty growths on the skin.",
                    "treatment": "Cryotherapy or laser removal for cosmetic reasons.",
                },
                "Seborrheic Keratosis": {
                    "name": "Seborrheic Keratosis",
                    "symptoms": "Brown or black growths with a waxy or stuck-on appearance.",
                    "treatment": "Cryotherapy, curettage, or laser treatment if necessary.",
                },
                "Squamous Cell Carcinoma": {
                    "name": "Squamous Cell Carcinoma",
                    "symptoms": "Firm, red nodules or scaly lesions, often on sun-exposed areas.",
                    "treatment": "Surgical excision, Mohs surgery, or radiation therapy.",
                },
                "Vascular Lesion": {
                    "name": "Vascular Lesion",
                    "symptoms": "Red or purple spots due to abnormal blood vessels.",
                    "treatment": "Laser therapy or sclerotherapy for cosmetic purposes.",
                },
            }

            def render_disease_info(disease_data):
                return f"""
                <div style="border: 2px solid #4CAF50; padding: 20px; margin: 20px; border-radius: 10px; background-color: #f9f9f9;">
                    <h2 style="color: #4CAF50;">{disease_data['name']}</h2>
                    <p><strong>Symptoms:</strong> {disease_data['symptoms']}</p>
                    <p><strong>Treatment:</strong> {disease_data['treatment']}</p>
                </div>
                """

            if predicted_class in disease_info:
                disease_html = render_disease_info(disease_info[predicted_class])
                html(disease_html, height=300)
                time_taken = end_time - start_time
                st.write(f"Time taken for prediction: {time_taken:.4f} seconds")       
        else:
            st.warning("Please upload or capture an image to proceed.")

        footer = """
        <style>
        footer {
            visibility: hidden;
        }
        
        footer:after {
            content: "📘 For educational purposes only";
            visibility: visible;
            display: block;
            position: relative;
            color: gray;
            font-size: 14px;
            text-align: center;
            padding: 10px;
        }
        </style>
        """
        
        st.markdown(footer, unsafe_allow_html=True)
        try:
            save_activity(st.session_state.global_username, image_path, predicted_class)
            st.success("Prediction saved successfully!")
        except:
            pass
        
    else:
        st.error("Please log in to predict a disease.")

if page == "Your Activity":
    if st.session_state.global_username:
        display_saved_activities(st.session_state.global_username)
    else:
        st.error("Please log in to view your activity.")
        
elif page == "Analytics":
    
    model_data = {
        "Model": ["Custom CNN", "DenseNet 169", "MobileNet", "NASNet", "ResNet 50"],
        "Accuracy": [0.87, 0.80, 0.66, 0.68, 0.36],
        "Macro Avg": [0.87, 0.80, 0.65, 0.67, 0.36],
        "Weighted Avg": [0.87, 0.80, 0.65, 0.67, 0.36],
    }
    
    df = pd.DataFrame(model_data)
    
    # Streamlit Interface
    st.title("Analytics: Model Performance Comparison")
    
    st.title("Model Accuracy Data")
    st.dataframe(df)
    
    # Accuracy Comparison
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots()
    bars = ax.bar(df["Model"], df["Accuracy"], color=["#4CAF50", "#2196F3", "#FFC107", "#FF5722", "#9C27B0"])
    ax.set_title("Model Accuracy Comparison", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", 
                ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)
    
    # Macro Avg vs Weighted Avg Comparison
    st.subheader("Macro Avg vs Weighted Avg Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(df["Model"]))
    
    bars1 = ax.bar(x, df["Macro Avg"], bar_width, label="Macro Avg", color="#3F51B5")
    bars2 = ax.bar([p + bar_width for p in x], df["Weighted Avg"], bar_width, label="Weighted Avg", color="#FF9800")
    
    ax.set_title("Macro Avg vs Weighted Avg for Models", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", 
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", 
                ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)

    st.markdown(
        """
        ### Insights
        - **Custom CNN** outperformed all other models with an accuracy of **87%**, as well as the highest macro and weighted averages.
        - **DenseNet 169** showed strong performance, achieving **80% accuracy**, but fell short of the Custom CNN.
        - **MobileNet** and **NASNet** had moderate accuracy scores (**66%** and **68%**, respectively).
        - **ResNet 50** performed poorly with an accuracy of only **36%**, indicating it may not be suitable for this dataset.
        """
    )
    
    data = {
        "Disease": [
            "Actinic_Keratosis", "Basal_Cell_Carcinoma", "Blue_Naevus", "Dermatofibroma", 
            "Elastosis_Perforans_Serpiginosa", "Lentigo_Maligna", "Melanocytic_Nevus", 
            "Melanoma", "Nevus_Sebaceus", "Pigmented_Benign_Keratosis", "Seborrheic_Keratosis", 
            "Squamous_Cell_Carcinoma", "Vascular_Lesion"
        ],
        "Original": [329, 514, 100, 122, 73, 98, 7078, 1567, 72, 1099, 80, 197, 142],
        "Augmented": [1539, 2285, 486, 591, 353, 473, 9716, 5417, 353, 4189, 393, 942, 686],
        "SMOTE": [9716] * 13
    }

    df_1 = pd.DataFrame(data)

    st.title("Analytics For Data Distribution - How Class Imbalance Was Handles ?")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.3
    x = range(len(df_1["Disease"]))

    ax.bar(x, df_1["Original"], width, label="Original", color="blue")
    ax.bar([i + width for i in x], df_1["Augmented"], width, label="Augmented", color="orange")
    ax.bar([i + 2 * width for i in x], df_1["SMOTE"], width, label="SMOTE", color="green")

    ax.set_title("Dataset Distribution Comparison")
    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels(df_1["Disease"], rotation=45, ha="right")
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Number of Images")
    ax.legend()

    st.pyplot(fig)

    st.title("Rare Disease Analysis")
    rare_diseases = {
        "Elastosis_Perforans_Serpiginosa": 353,
        "Nevus_Sebaceus": 353,
        "Blue_Naevus": 486,
        "Lentigo_Maligna": 473
    }

    fig, ax = plt.subplots()
    ax.pie(rare_diseases.values(), labels=rare_diseases.keys(), autopct='%1.1f%%', colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
    ax.set_title("Rare Disease Distribution After Augmentation")
    st.pyplot(fig)


    st.title("Dataset Summary")
    summary_data = {
        "Stage": ["Original", "After Augmentation", "After SMOTE"],
        "Total Images": [sum(df_1["Original"]), sum(df_1["Augmented"]), 126308]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)
    
    data = {
        "Disease": [
            "Actinic_Keratosis", "Basal_Cell_Carcinoma", "Blue_Naevus", "Dermatofibroma", 
            "Elastosis_Perforans_Serpiginosa", "Lentigo_Maligna", "Melanocytic_Nevus", 
            "Melanoma", "Nevus_Sebaceus", "Pigmented_Benign_Keratosis", "Seborrheic_Keratosis", 
            "Squamous_Cell_Carcinoma", "Vascular_Lesion"
        ],
        "Original": [329, 514, 100, 122, 73, 98, 7078, 1567, 72, 1099, 80, 197, 142],
        "Augmented": [1539, 2285, 486, 591, 353, 473, 9716, 5417, 353, 4189, 393, 942, 686],
        "SMOTE": [9716] * 13
    }

    df = pd.DataFrame(data)
    df.set_index("Disease", inplace=True)

    st.title("Class Distribution Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt="d", cmap="coolwarm", cbar=True, linewidths=0.5, ax=ax)
    ax.set_title("Class Distribution Heatmap (Raw Values)", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    st.pyplot(fig)

    footer = """
    <style>
    footer {
        visibility: hidden;
    }
    
    footer:after {
        content: "📘 For educational purposes only";
        visibility: visible;
        display: block;
        position: relative;
        color: gray;
        font-size: 14px;
        text-align: center;
        padding: 10px;
    }
    </style>
    """
    
    st.markdown(footer, unsafe_allow_html=True)
    
elif page == "About Us":
    
    st.title("About Us... The DEVELOPERS..! 🐍")
    
    


    st.markdown("""
        <style>
            .team-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
                
            }
            .card {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 300px;
                text-align: justify;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .card:hover {
                transform: translateY(-15px);
                
                box-shadow: 0 6px 12px rgba(255, 0, 0, 0.8);
            }
            .card-header {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
            }
            .card-content {
                font-size: 14px;
                line-height: 1.6;
                color: #555;
            }
            .card-content a {
                color: #0078d7;
                text-decoration: none;
                position: relative;
            }
            .card-content a::after {
                content: '';
                position: absolute;
                width: 100%;
                height: 2px;
                background: linear-gradient(90deg, red, green, blue);
                bottom: -2px;
                left: 0;
                transform: scaleX(0);
                transform-origin: right;
                transition: transform 0.3s ease-in-out;
            }
            .card-content a:hover::after {
                transform: scaleX(1);
                transform-origin: left;
                
            }
        </style>
    """, unsafe_allow_html=True)

    # About Us Section
    st.markdown("""
        <div class="team-container">
            <!-- Member 1 -->
            <div class="card">
                <div class="card-header">Lokesh Bhaskar</div>
                <div class="card-content">
                    <strong>Role:</strong> Machine Learning Enthusiast<br>
                    <strong>Email:</strong> <a href="mailto:lokesh.bhaskarnr@gmail.com">lokesh.bhaskarnr@gmail.com</a><br>
                    <strong>Phone:</strong> +91 73376 49759<br>
                    <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/lokesh-bhaskar-4113ab2a4/" target="_blank">Lokesh Bhaskar</a><br>
                    Specializes in AI/ML, data science, and predictive modeling, with expertise in image classification, GANs, and handling imbalanced datasets. Successfully managed and led real-time projects for developing end-to-end solutions, including real-time applications like handwritten digit recognition and skin disease prediction. Experienced in Android and web app development, as well as front-end design and database integration, ensuring seamless and efficient project execution.
                </div>
            </div>
            <!-- Member 2 -->
            <div class="card">
                <div class="card-header">Lakshmanan Basavaraj</div>
                <div class="card-content">
                    <strong>Role:</strong> Data Science Enthusiast<br>
                    <strong>Email:</strong> <a href="mailto:lakshmananbasavarajm@gmail.com">lakshmananbasavarajm@gmail.com</a><br>
                    <strong>Phone:</strong> +91 78296 09448<br>
                    <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/lakshmanan-b-236831233/" target="_blank">Lakshmanan B</a><br>
                    A Computer Science student passionate about using Python to solve real-world problems and specializing in data science and machine learning. Certified in Python Programming, Web Programming and Machine Learning. I’m looking for opportunities where I can grow and contribute to a team that values innovation and practical solutions.
                </div>
            </div>
            <!-- Member 3 -->
            <div class="card">
                <div class="card-header">L Vallabha Rao</div>
                <div class="card-content">
                    <strong>Role:</strong> Machine Learning Engineer<br>
                    <strong>Email:</strong> <a href="mailto:vallabhaarao@gmail.com">vallabhaarao@gmail.com</a><br>
                    <strong>Phone:</strong> +91 90713 03897<br>
                    <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/vallabha-rao-892193226/" target="_blank">Vallabha Rao</a><br>
                    A Computer Science student with a solid foundation in machine learning, web development, and cloud computing technologies. Led projects such as sentiment analysis and medical image classification, utilizing advanced neural network techniques like CNNs and GANs. Proficient in backend languages and databases. Hands-on experience with Google Cloud services and generative AI platforms like TensorFlow and Vertex AI.
                </div>
            </div>
            <!-- Member 4 -->
            <div class="card">
                <div class="card-header">Nikhil R Naik</div>
                <div class="card-content">
                    <strong>Role:</strong> Frontend Developer And Data Science Enthusiast<br>
                    <strong>Email:</strong> <a href="mailto:nikhillyrein456@gmail.com">nikhillyrein456@gmail.com</a><br>
                    <strong>Phone:</strong> +91 72043 70179<br>
                    <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/ananya-sharma" target="_blank">Nikhil R Naik</a><br>
                    As a Computer Science student, a passion is held for exploring data and deriving insights using Python. A strong foundation has been built in data science and machine learning principles, with hands-on experience gained in applying these skills to solve real-world problems. A desire exists to contribute to teams that value innovation and practical solutions while continuously seeking opportunities to expand knowledge in emerging technologies.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f4f4f9; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="font-size: 24px; color: #333333; font-weight: bold;">Interested in Our Dataset or Collaboration?</h2>
            <p style="font-size: 18px; color: #555555;">If you are interested in gaining access to the dataset, wish to collaborate on exciting projects, or have any inquiries, we would love to hear from you. Feel free to reach out to us for further discussions!</p>
            <p style="font-size: 18px; color: #333333;"><strong>Email:</strong> <a href="mailto:lokesh.bhaskarnr@gmail.com" style="color: #0066cc; font-weight: bold; text-decoration: none;">lokesh.bhaskarnr@gmail.com</a></p>
            <p style="font-size: 18px; color: #333333;"><a href="mailto:vallabhaarao@gmail.com" style="color: #0066cc; font-weight: bold; text-decoration: none;">vallabhaarao@gmail.com</a></p>
            <p style="font-size: 18px; color: #555555;">You can also explore our models and more on GitHub:</p>
            <p><a href="https://github.com/LokeshBhaskarNR/DermaEvolve---An-Advanced-Skin-Disease-Predictor.git" target="_blank" style="color: #0066cc; font-size: 18px; text-decoration: none; font-weight: bold; border-bottom: 2px solid #0066cc; transition: all 0.3s ease-in-out;">GitHub Repository</a></p>
            <p style="font-size: 16px; color: #777777; margin-top: 10px;">We look forward to connecting with you!</p>
        </div>
    """, unsafe_allow_html=True)

    footer = """
    <style>
    footer {
        visibility: hidden;
    }
    
    footer:after {
        content: "📘 For educational purposes only";
        visibility: visible;
        display: block;
        position: relative;
        color: gray;
        font-size: 14px;
        text-align: center;
        padding: 10px;
    }
    </style>
    """
    
    st.markdown(footer, unsafe_allow_html=True)

    
elif page == "Terms And Conditions":
    
    st.title("Terms And Conditions 📜")

    custom_css = """
    <style>

    .welcome-text {
        font-weight: bold;
        font-size: 1.5em;
        margin: 20px;
        color: black;
    }

    .highlight {
        font-weight: bold;
        color: rgb(255, 255, 0);
    }

    .privacy {
        color: rgb(255, 255, 0);;
        text-align: justify;
        font-weight: bold;
    }

    .accuracy {
        color: rgb(255, 255, 0);;
        text-align: justify;
        font-weight: bold;
    }

    .user-content {
        color: rgb(255, 255, 0);;
        text-align: justify;
        font-weight: bold;
    }

    .updates {
        color: rgb(255, 255, 0);;
        text-align: justify;
        font-weight: bold;
    }

    .support {
        color: rgb(255, 255, 0);;
        text-align: justify;
        font-weight: bold;
    }

    .footer {
        font-weight: bold;
        color: #000000;
        font-size: larger; font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    .footer strong {
        display: block;
        margin-top: 5px;
        font-size: large; font-weight: bold;
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="border-radius: 10px; background-color: white; padding: 20px;">
        <p class="welcome-text"> By using our app DermaEvolve - Skin Disease Predictor, you agree to the following terms:</p>

        <div style="border-radius: 10px; background-color: rgb(39, 39, 39); padding:10px">
            <p style="text-align: justify;"><span class="highlight privacy">🔒 Privacy:</span> Your privacy is important to us. All images and predictions are processed locally and are not shared with any third party.</p>
            <p style="text-align: justify;"><span class="highlight accuracy">⚠️ Accuracy:</span> Our app is designed for informational purposes only and is not a substitute for professional medical advice. Consult a dermatologist for any skin concerns.</p>
            <p style="text-align: justify;"><span class="highlight user-content">📷 User Content:</span> You are solely responsible for the images you upload. Avoid uploading personal or sensitive information.</p>
            <p style="text-align: justify;"><span class="highlight updates">📈 Updates:</span> We reserve the right to modify or discontinue any feature of the app without prior notice.</p>
            <p style="text-align: justify;"><span class="highlight support">🛠️ Support:</span> For any issues or queries, please contact our support team.</p>
        </div>

        <p class="footer" style="color: black;">
            Thank you for using our app!<br>
            <strong>Happy Diagnosing!</strong>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer = """
    <style>
    footer {
        visibility: hidden;
    }
    
    footer:after {
        content: "📘 For educational purposes only";
        visibility: visible;
        display: block;
        position: relative;
        color: gray;
        font-size: 14px;
        text-align: center;
        padding: 10px;
    }
    </style>
    """
    
    st.markdown(footer, unsafe_allow_html=True)
