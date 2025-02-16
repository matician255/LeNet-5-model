#### **Problem Statement**
The goal of this project is to implement the **LeNet-5** architecture, a classic convolutional neural network (CNN), for handwritten digit recognition using the **MNIST dataset**. This project demonstrates my ability to design, train, and evaluate a deep learning model for computer vision tasks.

---

#### **Dataset Description**
- **MNIST Dataset**: A widely used dataset in computer vision, consisting of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels.
  - **Training Set**: 60,000 images.
  - **Test Set**: 10,000 images.
- The dataset is preprocessed by normalizing pixel values to the range [0, 1] and reshaping the images to include a channel dimension (28x28x1).

---

#### **Model Architecture**
The **LeNet-5** architecture consists of the following layers:
1. **Convolutional Layer 1**: 6 filters, 5x5 kernel, tanh activation.
2. **Average Pooling Layer 1**: 2x2 pool size.
3. **Convolutional Layer 2**: 16 filters, 5x5 kernel, tanh activation.
4. **Average Pooling Layer 2**: 2x2 pool size.
5. **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
6. **Fully Connected Layer 1**: 120 units, tanh activation.
7. **Fully Connected Layer 2**: 84 units, tanh activation.
8. **Output Layer**: 10 units (one for each class), softmax activation.

---

#### **Training Process**
- **Optimizer**: Adam.
- **Loss Function**: Categorical cross-entropy.
- **Batch Size**: 64.
- **Epochs**: 10.
- **Metrics**: Accuracy.

---

#### **Results and Insights**
- **Test Accuracy**: ~98.5% (may vary slightly depending on training).
- **Training and Validation Curves**:
  - The model achieves high accuracy on both the training and validation sets, indicating good generalization.
  - The loss curves show consistent convergence, with no signs of overfitting.
- **Key Insights**:
  - LeNet-5 is a simple yet powerful architecture for handwritten digit recognition.
  - The use of convolutional layers allows the model to learn spatial hierarchies of features, making it robust to variations in digit appearance.

---

### **3. How to Run the Code**
1. Install the required libraries:
   ```bash
   pip install tensorflow matplotlib
   ```
2. Download the code and run it:
   ```bash
   python lenet5_tensorflow.py
   ```
3. The script will:
   - Load and preprocess the MNIST dataset.
   - Build and train the LeNet-5 model.
   - Plot the training and validation accuracy/loss curves.
   - Evaluate the model on the test set and print the accuracy.
