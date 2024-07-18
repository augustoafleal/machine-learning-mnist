import streamlit as st

st.title("MNIST Digit Recognition with Machine Learning 🖐️🔢")
st.write(
    """
Digit recognition project using CNN (Convolutional Neural Networks) and KNN (K-Nearest Neighbors) models.

The models are used to recognize handwritten digits using the MNIST dataset.
"""
)

st.header("MNIST Dataset 📊")
st.write(
    """
The MNIST dataset is widely used in the field of image recognition, containing 60,000 training images and 10,000 test images of digits from 0 to 9.
"""
)

st.header("Models Used 🧠")
st.subheader("Convolutional Neural Networks (CNN) 🌐")
st.write(
    """
CNNs are neural networks designed to process data with a grid-like structure, such as images. Typically, three convolutional layers are used to extract features from the images:
- Convolutional Layers: use filters to identify patterns in the images.
- Pooling Layers: reduce dimensionality while preserving important features.
- Fully Connected Layers: perform the final classification.
"""
)
st.subheader("K-Nearest Neighbors (KNN) 🤝")
st.write(
    """
KNN is a supervised learning algorithm that classifies new samples based on the majority vote of their k-nearest neighbors (7 neighbors provided the best results in the tests and was the value used). A distance metric is used to identify the nearest neighbors (we used Euclidean distance in the model), and classification is based on the most common class among the k-nearest neighbors.
"""
)

st.header("Results 📈")
st.write(
    """
CNN: Achieved an accuracy of 98.97%

KNN: Despite being simpler, the model also achieved a good result with 96.94% accuracy.
"""
)

st.header("References 📚")
st.write(
    """
- MNIST database: [MNIST database](http://yann.lecun.com/exdb/mnist/)
- CNN: [Convolutional Neural Networks Explained](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)
- KNN: [Machine Learning Basics with the K-Nearest Neighbors Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
- Project Repository: [GitHub Repository](https://github.com)
"""
)
