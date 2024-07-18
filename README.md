# MNIST Digit Recognition with Machine Learning
Digit recognition project using CNN (Convolutional Neural Networks) and KNN (K-Nearest Neighbors) models.

The models are used to recognize handwritten digits using the MNIST dataset, and it can be viewed on [Streamlit app](https://aafl-machine-learning-mnist.streamlit.app/)

## MNIST Dataset
The MNIST dataset is widely used in the field of image recognition, containing 60,000 training images and 10,000 test images of digits from 0 to 9.

## Models Used

### Convolutional Neural Networks (CNN)
CNNs are neural networks designed to process data with a grid-like structure, such as images. Typically, three convolutional layers are used to extract features from the images:
- Convolutional Layers: use filters to identify patterns in the images.
- Pooling Layers: reduce dimensionality while preserving important features.
- Fully Connected Layers: perform the final classification.

### K-Nearest Neighbors (KNN)
KNN is a supervised learning algorithm that classifies new samples based on the majority vote of their k-nearest neighbors (7 neighbors provided the best results in the tests and was the value used). A distance metric is used to identify the nearest neighbors (we used Euclidean distance in the model), and classification is based on the most common class among the k-nearest neighbors.

## Results
CNN: Achieved an accuracy of 98.97%

KNN: Despite being simpler, the model also achieved a good result with 96.94% accuracy.

![Presentation](assets/presentation.gif)

## References
MNIST database: [MNIST database](http://yann.lecun.com/exdb/mnist/)

CNN: [Convolutional Neural Networks Explained](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)

KNN: [Machine Learning Basics with the K-Nearest Neighbors Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
