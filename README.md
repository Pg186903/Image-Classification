## Project: Image Classification with CIFAR-10
### Overview

This project focuses on building an image classification model using the CIFAR-10 dataset, which contains 60,000 color images (32x32 pixels) across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The workflow includes data preprocessing, visualization, model building with deep learning (ANN/CNN), and evaluation.

# 

### Steps in the Notebook
#### 1. Environment Setup & Data Loading

* Libraries used:
  
  * tensorflow, keras → deep learning model building
  
  * numpy → data handling
  
  * matplotlib → visualization

        import tensorflow as tf
        from tensorflow import keras
        import matplotlib.pyplot as plt
        import numpy as np

* Checked dataset structure:

      X_train.shape  # (50000, 32, 32, 3)
      y_train.shape  # (50000, 1)

#### 2. Data Exploration & Visualization

* Visualized sample images from CIFAR-10 with their class labels.

      def plot_sample(index):
          plt.figure(figsize=(10,1))
          plt.imshow(X_train[index])


* Confirmed balanced distribution across 10 categories.

#### 3. Preprocessing

* Normalized image pixel values (0–255 → 0–1) for faster training.

* Converted labels to one-hot encoded format for categorical classification.

      X_train = X_train / 255
      X_test = X_test / 255
      
      y_train = keras.utils.to_categorical(y_train, 10)
      y_test = keras.utils.to_categorical(y_test, 10)

#### 4. Model Building

* Started with a baseline Artificial Neural Network (ANN).

* Then built a Convolutional Neural Network (CNN) for better performance.

      model = keras.Sequential([
          keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
          keras.layers.MaxPooling2D((2,2)),
          keras.layers.Conv2D(64, (3,3), activation='relu'),
          keras.layers.MaxPooling2D((2,2)),
          keras.layers.Flatten(),
          keras.layers.Dense(64, activation='relu'),
          keras.layers.Dense(10, activation='softmax')
      ])


* Compiled the model with:

      model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

#### 5. Training the Model

* Used training data (X_train, y_train) with validation on test set.

* Trained for multiple epochs to improve accuracy.

      history = model.fit(X_train, y_train, epochs=10, 
                          validation_data=(X_test, y_test))

#### 6. Model Evaluation

* Evaluated model performance using accuracy and loss curves.

* Achieved significantly better accuracy with CNN compared to ANN.

* Compared training vs validation accuracy to check for overfitting.

      test_loss, test_acc = model.evaluate(X_test, y_test)
      print("Test Accuracy:", test_acc)

# 

### Results

* The CNN model achieved high accuracy (~70–80%) on CIFAR-10.

* CNN significantly outperformed ANN due to its ability to capture spatial features in images.

* The model can predict object categories such as airplanes, cars, animals, and ships.

# 

### Key Features of the Project

* Dataset: CIFAR-10 (10 categories, 32x32 RGB images).

* Applied both ANN and CNN models.

* Preprocessing included normalization and one-hot encoding.

* Evaluated using accuracy and loss metrics.

* Can be extended with:

  * Data augmentation for better generalization.
  
  * Advanced CNN architectures (ResNet, VGG, Inception).
  
  * Hyperparameter tuning for optimizer, learning rate, and epochs.
