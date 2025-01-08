import os
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pickle

# Constants for Google Drive and Model
FOLDER_ID = '1pJ90jYSw87YAJcIUmKgJoxLRvqfsVqFH'
CLIENT_ID = '422923931863-2ce1taege1b76j9i7m3nbnbulgje8fau.apps.googleusercontent.com'
CLIENT_SECRET = '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'
REDIRECT_URIS = ['http://localhost:8501', 'http://localhost:8502']
DOWNLOAD_DIR = 'tempDownloadFile'
LAST_RUN_FILE = os.path.join(os.path.dirname(__file__), 'Last_run.txt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_plant_model.h5')
SENSOR_DATA_PATH = os.path.join(os.path.dirname(__file__), 'sensor_data.xlsx')

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Load the model for prediction
model = load_model(MODEL_PATH)

# Class labels for disease prediction
class_labels = {
    0: 'Alstonia Scholaris diseased (P2a)',
    1: 'Alstonia Scholaris healthy (P2b)',
    2: 'Arjun diseased (P1a)',
    3: 'Arjun healthy (P1b)',
    4: 'Bael diseased (P4b)',
    5: 'Basil healthy (P8)',
    6: 'Chinar diseased (P11b)',
    7: 'Chinar healthy (P11a)',
    8: 'Gauva diseased (P3b)',
    9: 'Gauva healthy (P3a)',
    10: 'Jamun diseased (P5b)',
    11: 'Jamun healthy (P5a)',
    12: 'Jatropha diseased (P6b)',
    13: 'Jatropha healthy (P6a)',
    14: 'Lemon diseased (P10b)',
    15: 'Lemon healthy (P10a)',
    16: 'Mango diseased (P0b)',
    17: 'Mango healthy (P0a)',
    18: 'Pomegranate diseased (P9b)',
    19: 'Pomegranate healthy (P9a)',
    20: 'Pongamia Pinnata diseased (P7b)',
    21: 'Pongamia Pinnata healthy (P7a)'
}

# Function to generate fake sensor data
def generate_fake_data(start_time, end_time, interval_minutes=30):
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    # Generate random sensor data
    data = {
        "date_time": timestamps,
        "pH": np.round(np.random.uniform(5.5, 8.5, len(timestamps)), 2),
        "EC": np.random.randint(1500, 2000, len(timestamps)),
        "PPM": np.random.randint(750, 950, len(timestamps)),
        "Temp": np.round(np.random.uniform(70.0, 80.0, len(timestamps)), 1),
        "Humidity": np.random.randint(40, 70, len(timestamps)),
    }
    
    return pd.DataFrame(data)

# Main function to update the sensor data file
def update_sensor_data(file_path):
    # Get current time and start of the day
    current_time = datetime.now()
    start_of_day = current_time.replace(hour=0, minute=1, second=0, microsecond=0)
    
    # Load existing data if file exists, otherwise create an empty DataFrame
    if os.path.exists(file_path):
        existing_data = pd.read_excel(file_path)
        last_recorded_time = pd.to_datetime(existing_data["date_time"]).max()
    else:
        existing_data = pd.DataFrame()
        last_recorded_time = start_of_day - timedelta(minutes=30)  # start generating from midnight
    
    # Determine the start time for new data
    new_data_start_time = last_recorded_time + timedelta(minutes=30)
    
    # Generate data up to the current time
    new_data = generate_fake_data(new_data_start_time, current_time)
    
    # Append new data to the existing data and save to Excel
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.to_excel(file_path, index=False)

    return updated_data

# Function to visualize sensor data
def visualize_sensor_data(sensor_data):
    # Plotting the sensor data
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=100)

    # pH vs Time
    axs[0].plot(sensor_data["date_time"], sensor_data["pH"], label="pH", color="green")
    axs[0].set_title("pH over Time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("pH")

    # EC vs Time
    axs[1].plot(sensor_data["date_time"], sensor_data["EC"], label="EC", color="blue")
    axs[1].set_title("EC over Time")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("EC")

    # Temp vs Time
    axs[2].plot(sensor_data["date_time"], sensor_data["Temp"], label="Temp", color="red")
    axs[2].set_title("Temperature over Time")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Temperature (°F)")

    # Improve layout and show
    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

# Authentication function for individual user credentials
def authenticate_with_google(user_email):
    try:
        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "redirect_uris": REDIRECT_URIS,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
                }
            },
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )

        # Set a unique token file for each user based on their email
        token_file = os.path.join(DOWNLOAD_DIR, f'{user_email}_token.pickle')

        if os.path.exists(token_file):
            # Load credentials from the file if they exist
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
            return creds

        # If no credentials, run OAuth flow and save token to a unique file
        creds = flow.run_local_server(port=8502)

        # Save the credentials for future use
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

        return creds
    except Exception as e:
        st.error(f"Error during authentication: {str(e)}")
        return None

# File handling functions
def get_last_run_time():
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, 'r') as f:
            return datetime.fromisoformat(f.read().strip())
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
#Function to update run time in text file
def update_last_run_time():
    with open(LAST_RUN_FILE, 'w') as f:
        f.write(datetime.utcnow().isoformat())

#Function to download image from drive
def download_image(file_id, file_name, drive_service):
    try:
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        request = drive_service.files().get_media(fileId=file_id)
        with open(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return file_path
    except Exception as e:
        st.error(f"Failed to download image: {str(e)}")
        return None

#Function to process the image from specified image path
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

#Function to predict the image from specified image path
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels.get(predicted_class, "Unknown Class")

# Streamlit app interface for page selection
page = st.sidebar.radio("Select a Page", ["Google Drive Downloader", "Leaf Disease Prediction", "Sensor Data Visualization"])

if page == "Sensor Data Visualization":
    st.title("Sensor Data Visualization")
    # Update and load sensor data
    sensor_data = update_sensor_data(SENSOR_DATA_PATH)
    # Visualize the sensor data
    visualize_sensor_data(sensor_data)

elif page == "Google Drive Downloader":
    st.title("Google Drive Image Downloader and Classifier")
    
    user_email = st.text_input("Enter your email for authentication")

    if st.button("Authenticate and Download Images") and user_email:
        credentials = authenticate_with_google(user_email)
        
        if credentials:
            st.success("Authenticated successfully.")
            
            try:
                drive_service = build('drive', 'v3', credentials=credentials)
                last_run_time = get_last_run_time()
                last_run_iso = last_run_time.isoformat() + "Z"

                st.write(f"Filtering files created after: {last_run_iso}")

                query = f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and createdTime >= '{last_run_iso}'"
                results = drive_service.files().list(q=query, pageSize=10).execute()
                files = results.get('files', [])
                
                if not files:
                    st.write("No new images found since last run.")
                else:
                    for file in files:
                        file_name = file.get('name')
                        file_id = file.get('id')
                        file_path = download_image(file_id, file_name, drive_service)

                        if file_path:
                            st.image(file_path, caption=file_name, use_column_width=True)
                            prediction = predict_image(file_path)
                            st.write(f"Predicted Class for {file_name}: {prediction}")

                    update_last_run_time()

            except HttpError as error:
                st.error(f"An error occurred while accessing Google Drive: {error}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

elif page == "Leaf Disease Prediction":
    st.title("Leaf Disease Prediction from Uploaded Image or Path Input")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    image_path = st.text_input("Or, enter an image file path on your system")

    if uploaded_image is not None:
        with open(os.path.join("temp_image.jpg"), "wb") as f:
            f.write(uploaded_image.getbuffer())
        image_path = "temp_image.jpg"

    if image_path:
        try:
            prediction = predict_image(image_path)
            st.image(image_path, caption="Uploaded Image", use_column_width=True)
            st.write(f"Predicted Class: {prediction}")
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
