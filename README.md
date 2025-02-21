# DermaEvolve: An Advanced Skin Disease Predictor

**DermaEvolve** is an AI-powered tool designed to predict various skin diseases using machine learning models. This project utilizes deep learning techniques, particularly convolutional neural networks (CNNs), to classify dermatological diseases based on images of skin lesions. The tool is built with TensorFlow and Streamlit, providing an easy-to-use interface for users to upload images and receive predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How to Contribute](#how-to-contribute)
- [License](#license)
- [Contact](#contact)

## Overview

DermaEvolve is aimed at providing a fast and reliable way to diagnose common and rare dermatological diseases using deep learning models. It leverages a variety of pre-trained models and custom-built architectures for accurate predictions, including models such as **ResNet50**, **EfficientNet**, and **MobileNet**. The app offers a simple, user-friendly interface through **Streamlit**, making it accessible to both technical and non-technical users. 

The main goal of this project is to aid in the early detection of skin diseases, improving outcomes by offering rapid predictions based on images.

## Features

- **User-Friendly Interface**: Built with **Streamlit** for easy interaction.
- **Image Upload and Prediction**: Upload an image of a skin lesion and get an immediate prediction.
- **Multiple Model Support**: Choose between different models like **ResNet50**, **EfficientNet**, and **MobileNet** for predictions.
- **Real-Time Predictions**: Fast, real-time predictions with the ability to view the results instantly after uploading an image.
- **Accuracy**: High accuracy achieved by fine-tuning pre-trained models on a dermatology-specific dataset.

## Technologies Used

- **TensorFlow**: For model building, training, and deployment.
- **Streamlit**: For creating the interactive web application interface.
- **OpenCV**: For image preprocessing and transformations before feeding them into models.
- **Pillow**: For image manipulation and transformations.
- **NumPy**: For handling numerical operations and array manipulations.
- **Matplotlib**: For plotting evaluation results and confusion matrices.

## Installation

### Prerequisites

- **Python 3.x**: Ensure you have Python 3.x installed. You can check this by running:
  ```bash
  python --version
  ```
- **Git**: Install Git to clone the repository. If you don’t have Git installed, download it from [here](https://git-scm.com/).

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourgithubusername/dermaevolve.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd dermaevolve
   ```

3. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows, use: venv\Scripts\activate
   ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the application**:
   Open a browser and navigate to `http://localhost:8501` to use the app.

## Usage

1. **Upload an image**: Choose an image of a skin lesion that you want to classify. The image should be in a supported format (e.g., PNG, JPEG).
2. **Model Selection**: Once the image is uploaded, you can select which model to use (ResNet50, EfficientNet, or MobileNet).
3. **Prediction**: Click the "Predict" button to receive the disease classification along with the confidence score.
4. **Result Display**: The application will display the predicted disease and confidence score.

## Dataset

The dataset used for training the model is derived from publicly available sources, which contain labeled images of various skin lesions. Rare Diseaes have been loaded from [Dermnet NZ](https://dermnetnz.org/). The dataset includes the following categories:

- **Basal Cell Carcinoma**
- **Squamous Cell Carcinoma**
- **Melanoma**
- **Actinic Keratosis**
- **Pigmented Benign Keratosis**
- **Seborrheic Keratosis**
- **Vascular Lesion**
- **Blue Naevus**
- **Elastosis Perforans Serpiginosa**
- **Melanocytic Nevus**
- **Dermatofibroma**
- **Lentigo Maligna**
- **Nevus Sebaceus**

The images in the dataset are preprocessed (resized to a 64 x 64 size and normalized) before being fed into the models for training.

## Model Architecture

The models used in **DermaEvolve** for skin disease classification include:

1. **MobileNet**: A lightweight architecture specifically designed for mobile and embedded applications, ensuring fast inference times and efficient computation, making it ideal for deployment in mobile apps. MobileNet is known for its speed and small size, allowing for fast, real-time predictions.

2. **NASNet**: Neural Architecture Search (NAS) is a technique used to automatically search for the best architecture for a given problem. **NASNet** is one such architecture that is optimized for high performance in image classification tasks, achieving impressive results on benchmark datasets.

3. **Custom CNN**: A custom convolutional neural network (CNN) architecture developed specifically for skin disease classification. This model combines various layers, including convolutional layers, pooling layers, and fully connected layers, to capture complex patterns in dermatological images and classify them accurately.

4. **ResNet50**: A deep residual network architecture known for its ability to train very deep networks. The model uses **skip connections** (or residual connections) to help prevent vanishing gradients, making it easier to train deeper networks. This architecture is used for feature extraction in the skin disease classification task.

5. **DenseNet169**: DenseNet (Densely Connected Convolutional Networks) is a CNN architecture where each layer is connected to every other layer in a feed-forward fashion. This approach promotes feature reuse and helps in learning more complex representations. **DenseNet169** is the variant used in this project, known for its efficiency and high performance in image classification tasks.

These models were trained using a dermatology-specific dataset, with some models being fine-tuned using transfer learning from pre-trained weights (e.g., ImageNet) to improve their performance on the skin disease classification task.

## How to Contribute

We welcome contributions! To contribute to this project, please follow these steps:

1. **Fork the repository** to your GitHub account.
2. **Clone the repository** to your local machine.
   ```bash
   git clone https://github.com/yourgithubusername/dermaevolve.git
   ```
3. **Create a new feature branch**: 
   ```bash
   git checkout -b feature-name
   ```
4. **Make changes** and add your improvements.
5. **Commit your changes**:
   ```bash
   git commit -am 'Add new feature or fix issue'
   ```
6. **Push your changes** to the branch:
   ```bash
   git push origin feature-name
   ```
7. **Create a pull request** with a description of the changes you have made.

We recommend making small, focused pull requests that address specific features or fixes. This makes it easier for others to review your contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries, questions, or collaborations, feel free to reach out:

- **Email**: [vallabhaarao@gmail.com](mailto:vallabhaarao@gmail.com)
- **GitHub Repository**: [DermaEvolve - An-Advanced Skin Disease Predictor]([https://github.com/yourgithubusername/dermaevolve](https://github.com/LokeshBhaskarNR/DermaEvolve---An-Advanced-Skin-Disease-Predictor.git))

If you're interested in the dataset, want to collaborate, or discuss the project, don't hesitate to contact us!

---
