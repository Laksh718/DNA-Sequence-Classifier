#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feed-Forward Neural Network for DNA Sequence Classification

This script implements a Feed-Forward Neural Network for DNA sequence classification
based on k-mer features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    """Load the preprocessed data."""
    X_train = np.load('../outputs/X_train.npy')
    X_test = np.load('../outputs/X_test.npy')
    y_train = np.load('../outputs/y_train.npy')
    y_test = np.load('../outputs/y_test.npy')
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_train_cat, y_test, y_test_cat

def build_model(input_shape, num_classes):
    """Build a feed-forward neural network model."""
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_nn(X_train, X_test, y_train_cat, y_test, y_test_cat):
    """Train and evaluate the neural network model."""
    # Create model
    model = build_model(X_train.shape[1], y_train_cat.shape[1])
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    os.makedirs('../outputs/nn_models', exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath='../outputs/nn_models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../plots/nn_training_history.png')
    plt.close()
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Generate and print classification report
    class_names = {
        0: 'GPCRs',
        1: 'Tyrosine kinase',
        2: 'Protein phosphatases',
        3: 'PTPs',
        4: 'AARSs',
        5: 'Ion channels',
        6: 'Transcription Factor'
    }
    target_names = [class_names[i] for i in range(7)]
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Save the classification report to a file
    with open('../outputs/nn_classification_report.txt', 'w') as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Neural Network Classifier')
    plt.tight_layout()
    plt.savefig('../plots/nn_confusion_matrix.png')
    plt.close()
    
    return model

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directories
    os.makedirs('../outputs', exist_ok=True)
    os.makedirs('../plots', exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_train_cat, y_test, y_test_cat = load_data()
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Loaded testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Train and evaluate the model
    print("\nTraining Feed-Forward Neural Network...")
    nn_model = train_and_evaluate_nn(X_train, X_test, y_train_cat, y_test, y_test_cat)
    
    # Save the model architecture and weights
    nn_model.save('../outputs/nn_model')
    print("\nNeural Network model saved to '../outputs/nn_model'")
    
    print("\nNeural Network classification completed.")

if __name__ == "__main__":
    main() 