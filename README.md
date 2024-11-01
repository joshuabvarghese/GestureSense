# **Gesture Sense**  
*Real-time Sign Language Recognition using CNN and OpenCV*

This project demonstrates a real-time **Sign Language Recognition (SLR)** system using **Convolutional Neural Networks (CNN)** and **OpenCV**. The system identifies **American Sign Language (ASL)** alphabets (excluding 'J' and 'Z') by recognizing hand gestures captured through a webcam.

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Real-time Recognition with OpenCV](#real-time-recognition-with-opencv)
6. [Installation](#installation)
7. [Usage](#usage)

## **Introduction**
The objective of this project is to create a deep learning model capable of recognizing hand gestures representing ASL alphabets. The project uses a **CNN** for feature extraction from grayscale images and **OpenCV** for real-time recognition through webcam input.

## **Dataset**
The dataset used is a subset of the **MNIST (Modified National Institute of Standards and Technology)** dataset, which includes images of 24 ASL alphabets (excluding 'J' and 'Z'). Each image is **28x28 pixels** in grayscale.

### **Dataset link:**  
[Download MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)

## **Model Architecture**
The model is based on a **Convolutional Neural Network (CNN)** with the following architecture:
- **Input Layer:** 28x28 grayscale image
- **Conv2D Layer 1:** 8 filters of size (3x3), ReLU activation
- **MaxPooling Layer 1:** 2x2 pool size
- **Conv2D Layer 2:** 16 filters of size (3x3), ReLU activation
- **MaxPooling Layer 2:** 4x4 pool size
- **Dropout Layer:** Dropout rate of 0.5 to prevent overfitting
- **Dense Layer:** 128 units, ReLU activation
- **Flatten Layer**
- **Output Layer:** 26 units with softmax activation for classification into 26 classes (A-Z, except J and Z)

## **Model Summary:**
```python
classifier = Sequential()
classifier.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28,28,1)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size=(4,4)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(26, activation='softmax'))
```

## **Training**

The model is trained using Keras with the following parameters:

Optimizer: SGD
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 50 (can be reduced for faster training)
Batch Size: 100


```python
classifier.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=50, batch_size=100)
```
After training, the model achieves an accuracy of approximately 94%.

## **Real-time Recognition with OpenCV**

Using OpenCV, the system captures real-time webcam input, processes it into a 28x28 grayscale image, and predicts the corresponding ASL alphabet. The process includes:

Capturing webcam input.
Converting the captured image to grayscale.
Resizing and preprocessing the image to match the input size of the trained model.
Using the model for prediction.


## **OpenCV Capture and Prediction:**

```python
def main():
    cam_capture = cv2.VideoCapture(0)
    _, image_frame = cam_capture.read()
    image_grayscale = cv2.cvtColor(crop_image(image_frame, 300,300,300,300), cv2.COLOR_BGR2GRAY)
    im_resized = cv2.resize(image_grayscale, (28,28))
    im_resized = np.expand_dims(np.resize(im_resized, (28, 28, 1)), axis=0)
    prediction, pred_class = keras_predict(classifier, im_resized)
```

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/username/signspotter.git
cd signspotter
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run on Google Colab:
To load the dataset and run the training, you can also use Google Colab for easy access to GPU and faster training.

## **Usage**

Train the model using the provided training code or download the pre-trained model.

Run the OpenCV script to start the webcam capture for real-time recognition.
Show ASL alphabets (A-Z, except J and Z) using your hand and the model will predict the letter on screen.
```bash
python real_time_recognition.py
```











