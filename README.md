# Vishwani-MachineLearning
My machine learning projects
Steps to Implement Leaf Classification in a Streamlit Application
# 1.	Download Training Data
o	Dataset: Download the plant leaves dataset from Kaggle:
https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification
This dataset contains images of 22 leaf categories, including healthy and diseased leaves.

# 2.	Train and Save the Model
o	Use the provided Python file OptimizedModel_For_ImageClassification.py to train the model with the Kaggle dataset.
o	Once trained, save the model for further use, ensuring it is compatible with the Streamlit app. Save the model in a format like .pkl or .h5.
o	Alternative Download Link:If you want to save time and computation power, download the trained model directly from this Google Drive link:
https://drive.google.com/file/d/1OKSLnffIk3TR2ZB2BIm8di9D7CLlLHNh/view

# 3.Access Google Drive Files Using GCP

Enable the Google Drive API from the Google Cloud Console.
Use the google-auth and google-api-python-client libraries to authenticate and interact with your Drive.
Authenticate with a service account or OAuth2 credentials and specify the Drive folder to list and download files.

# 4.	Setup the Streamlit Application
o	App Name: DevNetHackathonV2App.py.
o	Functionality:
	Connect to a Google Drive folder containing leaf data for classification.
	Use the trained model to classify new images into one of the 22 categories.
	The app should maintain a record of the last execution time to identify and process new images during subsequent runs.

# 5.	Implement Google Drive Integration
o	Use the Google Drive API to access and download images from a specified folder.
o	Store the timestamp of the last program run locally (e.g., in a .txt or .json file) and use it to filter new files in the Drive folder during the next execution.

# 6.	Model Integration in Streamlit App
o	Load the saved trained model in DevNetHackathonV2App.py.
o	Preprocess the uploaded leaf images (resize, normalize, etc.).
o	Use the model to predict whether the leaves are healthy or diseased.
o	Display the classification results in the Streamlit app.

# 7.	Run and Monitor the App
o	Deploy and run the Streamlit application locally or on a cloud service.
o	Ensure that the app is capable of dynamically updating its results based on new files added to the Google Drive folder.

