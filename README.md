# Handwritten Digit Recognition

##  Project Description

This project is a **Handwritten Digit Recognition System** that uses a **Convolutional Neural Network (CNN)** to identify handwritten digits from 0 to 9.
The model is trained on digit images and predicts the digit from an uploaded image.

##  Technologies Used

* Python
* TensorFlow / Keras
* Flask
* HTML
* NumPy
* Pillow

##  Project Structure

Handwritten-Digit-Recognition
│
├── backend
│   ├── app.py
│   ├── train_model.py
│   └── digit_cnn_model.h5
│
├── frontend
│   └── index.html
│
├── requirements.txt
└── README.md

##  How to Run the Project

1. Clone the repository

git clone https://github.com/divy686/Handwritten-Digit-Recognition.git

2. Install dependencies

pip install -r requirements.txt

3. Run the backend

python backend/app.py

4. Open the frontend

Open **index.html** in your browser.

##  Features

* Recognizes handwritten digits (0–9)
* CNN based deep learning model
* Simple web interface for digit prediction

##  Model

The model is trained using a **Convolutional Neural Network (CNN)** which learns patterns from digit images to accurately classify digits.

##  Future Improvements

* Improve model accuracy
* Add drawing canvas for digit input
* Deploy the project online
