#!/usr/bin/env python
# coding: utf-8

# ### PORTFOLIO OF WORK
# ## Module Title: Understanding AI (771763_B25_T2)
# 
# ### COMPONENT TWO (Emergency Vehicle Identification)
# 
# ### Student Name: Stephen Ojochogwu Stephen
# ### Student Number: 202552762

# ### Project Overview
# This project focuses on developing a Convolutional Neural Network (CNN) to classify images of vehicles as either **emergency** or **non-emergency**. The goal is to build a robust model suitable for real-world deployment by experimenting with different architectures, regularisation techniques, and hyperparameters.
# 
# The dataset is sourced from Kaggle and contains labelled images used to train and evaluate the model.
# 

# In[8]:


# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ### Loading Dataset
# The dataset labels are loaded from a CSV file, which maps each image to its corresponding class:
# - **1 = Emergency vehicle**
# - **0 = Non-emergency vehicle**

# In[9]:


#  Load labels from train CSV (Ground truth)
labels_df = pd.read_csv("train.csv")

# Verify labels
print(labels_df["emergency_or_not"].value_counts())
# 1 = Emergency, 0 = Non‑Emergency


# ### Train-Validation Split
# The dataset is split into training and validation sets. This ensures that both classes (emergency and non-emergency) are proportionally represented in each subset.
# 
# Training set: 80%
# Validation set: 20%
# 

# In[10]:


# Train / Validation split
train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df["emergency_or_not"],
    random_state=42 )


# ### Image Preprocessing
# All images are resized to **128 × 128 pixels** to ensure a consistent input size for the CNN.
# The images are then converted into NumPy arrays for model input.

# In[11]:


#  Image loader function
def load_images(df, base_dir, img_size=(128, 128)):
    X, y = [], []

    for _, row in df.iterrows():
        # images are in Emergency_Vehicles/train/
        img_path = os.path.join(base_dir, "train", row["image_names"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0

        X.append(img)
        y.append(row["emergency_or_not"])

    return np.array(X), np.array(y)


# In[12]:


#  Load datasets
IMAGE_DIR = "Emergency_Vehicles"

x_train, y_train = load_images(train_df, IMAGE_DIR)
x_val, y_val = load_images(val_df, IMAGE_DIR)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


# ### Data Visualisation
# A sample image from the training set is displayed to verify correct label assignment and proper preprocessing.

# In[13]:


# Visual Verification
plt.imshow(x_train[1])
plt.title(f"Label = {y_train[1]} (1=Emergency, 0=Non‑Emergency)")
plt.axis("off")
plt.show()


# ### Baseline CNN Architecture
# The baseline model is composed of three convolutional blocks, beginning with 32-filter Conv2D layer, each followed by max pooling. The model is trained for **20 epochs** using the default Adam optimiser (0.001)
# 
# Key components:
# - Conv2D layers with ReLU activation
# - MaxPooling layers for dimensionality reduction
# - Fully connected Dense layer for classification
# - Softmax output layer for binary classification

# In[14]:


# Baseline CNN Model
# Creating a Sequential class instance of the model with 3 convulation blocks.
# Including model parameters such as Pooling layer, flatten layer, dense layer and output dense layer. 
baseline_cnn = Sequential([ Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Flatten(), Dense(64, activation="relu"),
                     Dense(2, activation="softmax") ])

# Here i compile the CNN model using the default Adam optimiser learning rate.
# loss function: sparse_categorical_crossentropy, perfomance metric: accuracy
baseline_cnn.compile( optimizer=Adam(), loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] )

baseline_cnn.summary()

# Model Training (20 Epochs)
history_1 = baseline_cnn.fit( x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val) )


# ### Model Evaluation
# The model is evaluated using the following metrics:
# - Accuracy: Overall correctness of predictions
# - Precision: Proportion of predicted emergency vehicles that are correct
# - Recall: Ability to correctly identify emergency vehicles

# In[15]:


# Model Evaluation 

# Performance Metrics (accuracy, precision, recall)
y_pred_1 = np.argmax(baseline_cnn.predict(x_val), axis=1)

accuracy = accuracy_score(y_val, y_pred_1)
precision = precision_score(y_val, y_pred_1)
recall = recall_score(y_val, y_pred_1)

# Confusion Matrix & Loss Curves
cm = confusion_matrix(y_val, y_pred_1)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non‑Emergency", "Emergency"],
            yticklabels=["Non‑Emergency", "Emergency"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history_1.history["loss"], label="Training Loss")
plt.plot(history_1.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baeline CNN Training and Validation Loss")
plt.show()

# Model Evaluation 
results_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Baseline CNN": [accuracy, precision, recall]
})
print("Baseline CNN Performance Table:")
results_table


# The baseline CNN achieved an accuracy of 71.2%, with a high recall of 87.6% but relatively low precision of 60.6%. This indicates that the model is effective at identifying emergency vehicles but tends to misclassify a significant number of non-emergency vehicles as emergency vehicles.
# 
# The confusion matrix further confirms this behaviour, showing a high number of false positives (78), alongside a relatively small number of false negatives (17). This suggests that the model is biased toward predicting the emergency class, prioritising recall over precision.
# 
# Analysis of the training and validation loss curves reveals clear signs of overfitting. While training loss decreases steadily, validation loss increases sharply after several epochs, indicating that the model fails to generalise well to unseen data. Overall, although the model performs reasonably well in detecting emergency vehicles, its high false positive rate and overfitting limit its effectiveness for real-world deployment. This highlights the need for improved regularisation and hyperparameter tuning.

# ## Hyperparameter Tuning
# To improve model performance in a scientifically rigorous manner, a structured hyperparameter tuning strategy was adopted. Rather than modifying multiple parameters simultaneously, a controlled-variable approach was used, where one key factor is changed at a time while keeping all others constant. This ensures that any observed performance differences can be directly attributed to the specific modification.
# ### Experimental Tuning Strategy
# The tuning process followed this order:
# 
# 1. Baseline Model
# Establish a benchmark performance for comparison.
# 
# 2. Regularisation (Dropout)
# Introduce dropout to reduce overfitting.
# 
# 3. Learning Rate Tuning
# Adjust optimisation dynamics to improve convergence.
# 
# 4. Batch Size Tuning
# Analyse impact on training stability and generalisation.
# 
# 5. Architecture Depth
# Increase model capacity once optimisation is stable.
# 
# ### Model 1: Baseline CNN
# The baseline model was implemented using three convolutional blocks and trained using the default Adam optimiser (learning rate = 0.001) to establish a reference point for all subsequent experiments and identify initial issues such as overfitting or class imbalance
# ### Model 2: Custom CNN A (Add Dropout)
# A dropout layer (rate = 0.5) was added after the dense layer to reduce overfitting observed in the baseline model and improve generalisation to unseen data. Dropout is a widely used regularisation technique that prevents co-adaptation of neurons and reduces overfitting in deep networks (Srivastava et al., 2014).
# 
# ### Model 3: Custom CNN B (Tune Learning Rate)
# The learning rate was reduced from 0.001 to 0.0001. This is to improve convergence stability and prevent overshooting during optimisation. The learning rate is one of the most critical hyperparameters influencing model convergence and performance (Goodfellow, Bengio and Courville, 2016).
# 
# ### Model 4: Custom CNN C (Tune Batch Size)
# The batch size was increased from 32 to 64 to evaluate its impact on training stability and analyse its effect on model generalisation. Prior Literature suggests that mini-batch sizes in the range of 32-128 are commonly effective for deep learning optimization. Smaller batches introduce noise that can improve generalisation, while larger batches can lead to faster but potentially less generalisable convergence (Keskar et al., 2017). A batch size of 64 was selected as it represents a moderate increase from baseline, allowing smoother gradient updates.
# 
# ### Model 5: Custom CNN D (Increase Architecture Depth)
# An additional convolutional block (Conv2D + MaxPooling) was introduced to increase model depth. To improve feature extraction capability and enhance model capacity for complex patterns. Deeper convolutional networks are capable of learning hierarchical representations of visual features, leading to improved performance on image classification tasks (Krizhevsky, Sutskever and Hinton, 2012).

# ### Custom CNN A. Add Dropout

# In[16]:


# Custom CNN A. Adding a 50% Dropout rate as regularization to test whether overfitting reduces.

custom_cnn_a = Sequential([ Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Flatten(), Dense(64, activation="relu"), Dropout(0.5),
                     Dense(2, activation="softmax") ])

custom_cnn_a.compile( optimizer=Adam(), loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] )

custom_cnn_a.summary()

#Training
history_2 = custom_cnn_a.fit( x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val) )

# Model Evaluation 
y_pred_2 = np.argmax(custom_cnn_a.predict(x_val), axis=1)

accuracy = accuracy_score(y_val, y_pred_2)
precision = precision_score(y_val, y_pred_2)
recall = recall_score(y_val, y_pred_2)

# Confusion Matrix & Loss Curves
cm = confusion_matrix(y_val, y_pred_2)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non‑Emergency", "Emergency"],
            yticklabels=["Non‑Emergency", "Emergency"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history_2.history["loss"], label="Training Loss")
plt.plot(history_2.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Custom CNN A: Training and Validation Loss")
plt.show()

# Model Evaluation 
results_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Custom CNN A": [accuracy, precision, recall]
})
print("Custom CNN A Performance Table:")
results_table


# The introduction of dropout (rate = 0.5) significantly improved the performance and generalisation capability of the model compared to the baseline CNN.
# 
# The model achieved an accuracy of **84.85%**, demonstrating a substantial improvement in overall classification performance. Precision increased to **85.95%**, indicating that the model is highly reliable when predicting emergency vehicles, with a strong reduction in false positives. Recall reached **75.91%**, showing that the model still identifies most emergency vehicles, although some are missed.
# 
# Analysis of the confusion matrix provides deeper insight into model behaviour. The model correctly classified **176 non-emergency vehicles (true negatives)** and **104 emergency vehicles (true positives)**. Importantly, false positives were reduced to **17**, confirming that the model is less likely to incorrectly label non-emergency vehicles as emergency vehicles. However, **33 false negatives** indicate that some emergency vehicles are not detected, reflecting a trade-off between precision and recall. Compared to the baseline model, this represents a shift toward a more conservative and balanced prediction strategy.
# 
# The loss curves reveal that training loss steadily decreases, indicating effective learning. However, validation loss initially decreases and then begins to increase after approximately 6–8 epochs. This suggests that, although dropout reduces overfitting, it does not completely eliminate it. The widening gap between training and validation loss confirms that the model still begins to memorise the training data at later stages.

# ### Custom CNN B. Tune Learning Rate

# In[17]:


# Custom CNN B
custom_cnn_b = Sequential([ Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Flatten(), Dense(64, activation="relu"), Dropout(0.5),
                     Dense(2, activation="softmax") ])

# Tuning learning rate to 0.0001 to improve convergence stability
custom_cnn_b.compile( optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] )

custom_cnn_b.summary()

# Training
history_3 = custom_cnn_b.fit( x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val) )

# Performance Metrics
y_pred_3 = np.argmax(custom_cnn_b.predict(x_val), axis=1)

accuracy = accuracy_score(y_val, y_pred_3)
precision = precision_score(y_val, y_pred_3)
recall = recall_score(y_val, y_pred_3)

# Confusion Matrix & Loss Curves
cm = confusion_matrix(y_val, y_pred_3)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non‑Emergency", "Emergency"],
            yticklabels=["Non‑Emergency", "Emergency"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history_3.history["loss"], label="Training Loss")
plt.plot(history_3.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Custom CNN B: Training and Validation Loss")
plt.show()

# Model Evaluation 
results_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Custom CNN B": [accuracy, precision, recall]
})
print("Custom CNN B Performance Table:")
results_table


# ### Custom CNN B 
# 
# Custom CNN B introduced a reduced learning rate (0.0001) while retaining dropout regularisation, with the aim of improving convergence stability and overall model performance.
# 
# The model achieved an accuracy of **80.30%**, which represents a decline compared to Custom CNN A. Precision remained relatively strong at **81.58%**, indicating that predictions of emergency vehicles are still fairly reliable. However, recall dropped to **67.88%**, showing that the model is less effective at identifying emergency vehicles and misses a larger proportion of true cases.
# 
# The confusion matrix provides further insight into this behaviour. The model correctly classified **172 non-emergency vehicles (true negatives)** and **93 emergency vehicles (true positives)**. However, **21 false positives** indicate a slight increase in incorrect emergency predictions compared to Custom CNN A. More notably, **44 false negatives** highlight a substantial drop in the model’s ability to detect emergency vehicles. This confirms that the lower recall is driven by an increased number of missed emergency cases.
# 
# Analysis of the loss curves indicates improved training stability compared to the previous model. Both training and validation loss decrease smoothly, and the gap between them remains relatively small throughout the training process. Unlike Custom CNN A, there is no sharp divergence between training and validation loss, suggesting that the reduced learning rate helps control overfitting and leads to more stable convergence.
# 
# Overall, Custom CNN B demonstrates improved optimisation behaviour and reduced overfitting due to the lower learning rate. However, this comes at the cost of reduced performance, particularly in recall.

# ### Custom CNN C. Tune Batch Size

# In[18]:


# Custom CNN C
custom_cnn_c = Sequential([ Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Conv2D(256, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Flatten(), Dense(64, activation="relu"), Dropout(0.5),
                     Dense(2, activation="softmax") ])

custom_cnn_c.compile( optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] )

custom_cnn_c.summary()

# Training. Tune Batch Size to 64 to investigate smoother gradient updates and test generalization to unseen data.
history_4 = custom_cnn_c.fit( x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val) )

# Performance Metrics
y_pred_4 = np.argmax(custom_cnn_c.predict(x_val), axis=1)

accuracy = accuracy_score(y_val, y_pred_4)
precision = precision_score(y_val, y_pred_4)
recall = recall_score(y_val, y_pred_4)

# Confusion Matrix & Loss Curves
cm = confusion_matrix(y_val, y_pred_4)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non‑Emergency", "Emergency"],
            yticklabels=["Non‑Emergency", "Emergency"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history_4.history["loss"], label="Training Loss")
plt.plot(history_4.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Custom CNN C: Training and Validation Loss")
plt.show()

# Model Evaluation 
results_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Custom CNN C": [accuracy, precision, recall]
})
print("Custom CNN C Performance Table:")
results_table


# ### Custom CNN C
# 
# Custom CNN C introduced a larger batch size (64) along with increased model capacity by adding an additional convolutional layer, while retaining the reduced learning rate and dropout regularisation. The aim was to evaluate the impact of batch size on training stability and generalisation performance.
# 
# The model achieved an accuracy of **78.79%**, representing a further decline compared to Custom CNN B. Precision increased slightly to **84.54%**, indicating that predictions of emergency vehicles remain highly reliable, with a low rate of false positives. However, recall dropped significantly to **59.85%**, highlighting a substantial decrease in the model’s ability to correctly identify emergency vehicles.
# 
# The confusion matrix provides deeper insight into this performance. The model correctly classified **178 non-emergency vehicles (true negatives)** and **82 emergency vehicles (true positives)**. False positives were reduced to **15**, the lowest among all models so far, demonstrating very strong precision and conservative prediction behaviour. However, **55 false negatives** represent a significant increase, indicating that many emergency vehicles are missed. This confirms that the model has become overly conservative, prioritising precision at the expense of recall.
# 
# Analysis of the loss curves shows relatively stable training and validation behaviour. Both training and validation loss decrease consistently over time, and the gap between them remains moderate, suggesting improved control over overfitting compared to earlier models. The absence of sharp divergence indicates that the combination of a lower learning rate and larger batch size contributes to smoother optimisation.

# ### Custom CNN D. Increase Architecture Depth

# In[19]:


# Custom CNN D. 4th Convulation Block introduced to increase model depth and feature extraction capacity

custom_cnn_d = Sequential([ Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                     Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2), 
                       Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Conv2D(256, (3,3), padding="same", activation="relu", input_shape=(128,128,3)),
                     MaxPooling2D(2,2),
                     Flatten(), Dense(64, activation="relu"), Dropout(0.5),
                     Dense(2, activation="softmax") ])

custom_cnn_d.compile( optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] )

custom_cnn_d.summary()

# Training
history_5 = custom_cnn_d.fit( x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val) )

# Performance Metrics
y_pred_5 = np.argmax(custom_cnn_d.predict(x_val), axis=1)

accuracy = accuracy_score(y_val, y_pred_5)
precision = precision_score(y_val, y_pred_5)
recall = recall_score(y_val, y_pred_5)

# Confusion Matrix & Loss Curves
cm = confusion_matrix(y_val, y_pred_5)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non‑Emergency", "Emergency"],
            yticklabels=["Non‑Emergency", "Emergency"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history_5.history["loss"], label="Training Loss")
plt.plot(history_5.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Custom CNN D: Training and Validation Loss")
plt.show()

# Model Evaluation 
results_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Custom CNN D": [accuracy, precision, recall]
})
print("Custom CNN D Performance Table:")
results_table


# ### Custom CNN D 
# 
# Custom CNN D increased the architectural depth of the model by adding an additional convolutional block, while maintaining dropout regularisation, a reduced learning rate, and a larger batch size. The aim was to enhance feature extraction and improve classification performance by increasing model capacity.
# 
# The model achieved an accuracy of **82.42%**, representing an improvement over Custom CNN B and C, though slightly lower than Custom CNN A. Precision and recall are both **78.83%**, indicating a well-balanced model that performs equally well in correctly identifying emergency vehicles and avoiding misclassification. This balance suggests that the model has achieved a more effective trade-off between false positives and false negatives.
# 
# The confusion matrix supports this observation. The model correctly classified **164 non-emergency vehicles (true negatives)** and **108 emergency vehicles (true positives)**. There are **29 false positives** and **29 false negatives**, showing a symmetrical distribution of errors. Compared to previous models, Custom CNN D does not overly prioritise either precision or recall, making it a more balanced classifier overall. Although false positives are higher than in Custom CNN C, the reduction in false negatives represents a significant improvement in detecting emergency vehicles.
# 
# Analysis of the loss curves shows that both training and validation loss decrease steadily, indicating effective learning. The gap between training and validation loss remains moderate, although there is a noticeable fluctuation in validation loss at later epochs, suggesting some instability and mild overfitting. However, this behaviour is less severe than in the baseline model, indicating that regularisation and learning rate tuning are still effective in controlling overfitting to some extent.
# 
# Custom CNN D demonstrates a strong balance between precision and recall, making it a more reliable model for practical applications. While it does not achieve the highest accuracy, its balanced performance and improved detection of emergency vehicles make it a strong candidate compared to other models. This suggests that increasing model depth can enhance feature representation, but must be carefully combined with proper regularisation to maintain generalisation performance.

# In[20]:


# Show Custom CNN D model confidence for random images
import random

idx = random.randint(0, len(x_val)-1)

probs = custom_cnn_d.predict(x_val[idx:idx+1])[0]
pred = np.argmax(probs)

plt.imshow(x_val[idx])
plt.title(
    f"True: {y_val[idx]} | Pred: {pred}\n"
    f"P(Non‑Emerg)={probs[0]:.2f}, P(Emerg)={probs[1]:.2f}"
)
plt.axis("off")
plt.show()


# In[21]:


# Show model confidence for random images
import random

idx = random.randint(0, len(x_val)-1)

probs = custom_cnn_d.predict(x_val[idx:idx+1])[0]
pred = np.argmax(probs)

plt.imshow(x_val[idx])
plt.title(
    f"True: {y_val[idx]} | Pred: {pred}\n"
    f"P(Non‑Emerg)={probs[0]:.2f}, P(Emerg)={probs[1]:.2f}"
)
plt.axis("off")
plt.show()


# ### References
# 
# Bergstra, J. and Bengio, Y. (2012) ‘Random Search for Hyper-Parameter Optimization’, Journal of Machine Learning Research, 13, pp. 281–305.
# 
# 
# Goodfellow, I., Bengio, Y. and Courville, A. (2016) Deep Learning. MIT Press.
# 
# 
# Keskar, N.S. et al. (2017) ‘On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima’, International Conference on Learning Representations (ICLR).
# 
# 
# Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2012) ‘ImageNet classification with deep convolutional neural networks’, Advances in Neural Information Processing Systems (NeurIPS).
# 
# 
# Srivastava, N. et al. (2014) ‘Dropout: A Simple Way to Prevent Neural Networks from Overfitting’, Journal of Machine Learning Research, 15, pp. 1929–1958.

# In[ ]:





# In[ ]:




