# Brain Tumor MRI Image Classification

This project uses deep learning to classify brain MRI images into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

We trained a Convolutional Neural Network (CNN) using the Brain MRI dataset and deployed the model using "Streamlit", allowing users to upload an MRI image and receive a real-time prediction.


## Dataset

- Source: [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Structure: 
BrainTumorDataset/
├── Training/
│ ├── Glioma/
│ ├── Meningioma/
│ ├── No Tumor/
│ └── Pituitary/
└── Testing/


## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib / Seaborn
- Streamlit


## Features

- Cleaned and preprocessed MRI scan data
- Trained CNN achieving high validation accuracy
- Interactive web app using Streamlit
- Predicts tumor type from uploaded images in real time


## How to Run the App

1. Clone this repository:
 ```bash
 git clone https://github.com/your-username/brain-tumor-mri-classifier.git
 cd brain-tumor-mri-classifier

