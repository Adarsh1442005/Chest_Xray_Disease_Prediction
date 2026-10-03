## Author and Builder of the Project
   Adarsh Pandey  
   B.Tech Student | Backend Developer | AI & Digital Forensics Enthusiast

## **Note** As of Now I have not uploaded the dataset for the model in the github if anybody wants to train the same model he/ she can request to me about the dataset on my email-id ap4866017@gmail.com.. I have dataset which is already neat and clean you people not have to do the data_preprocessing Steps........
   



# Chest X-Ray Disease Prediction 

A deep learning project that uses a Convolutional Neural Network (CNN) to classify chest X-ray images into multiple disease categories.  
The project includes a **PyTorch model**, a **Flask backend**, and a simple **HTML frontend** for user interaction.

---

## 📂 Project Structure
chest_xray_5_diseases/
|___train/ **Train folder and test folder you will not find in the fork repo you have to request ot me I wil provide to you**
|___test/
├── Backend/
│   └── backend.py          # Flask app for serving predictions
│   └── Backend_predict.py  # Prediction route logic
│
├── Front/
│   └── index.html          # User interface
│
├── cnn_model.py            # CNN model architecture
├── predict.py              # Helper function for single image prediction
├── dataset_loader.py       # Data loading utilities
├── cnn_model.pth           # Trained model weights  **Once your model get trained this file 
                              auto create you not have to create it explicitly you will have 
                              to run just train.py file**
|__train.py                               
├── uploads/                # Uploaded images (auto-created)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation


---

## 🚀 Features
- CNN model trained on chest X-ray dataset.
- Flask backend for serving predictions.
- Frontend form to upload X-ray images.
- JSON response with predicted disease class.
- Evaluation metrics: accuracy, classification report, confusion matrix.

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adarsh1442005/Chest_Xray_Disease_Prediction.git
   cd Chest_Xray_Disease_Prediction

2. **Install the dependecies**
pip install -r requirements.txt

3. **Ensure Uploads folder Exist**

4. **Running the Project**
    python Backend/backend.py
    Open the frontend

Go to http://127.0.0.1:5000/ in your browser.

Upload a chest X-ray image.

Get prediction results. 

5. **Model Evaluation**
    python evaluate.py
    This prints:

   Classification Report (precision, recall, F1-score per class).

     Confusion Matrix (visualizing misclassifications).

6. **Requirements**

   Python 3.8+
   PyTorch
    Flask
   scikit-learn
   torchvision

**Notes**: Ensure cnn_model.pth is present (trained weights).
           Modify cnn_model.py if you want to experiment with different architectures.
           Use predict.py for single-image predictions outside Flask.




