#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-Nearest Neighbors Classification for DNA Sequences

This script implements a KNN classifier for DNA sequence classification
based on k-mer features.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def main():
    start_time = time.time()
    
    # Load training and testing data
    X_train = np.load('outputs/X_train.npy')
    X_test = np.load('outputs/X_test.npy')
    y_train = np.load('outputs/y_train.npy')
    y_test = np.load('outputs/y_test.npy')
    
    print(f"\nLoaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Loaded testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Load protein class mapping
    with open('outputs/protein_classes.pkl', 'rb') as f:
        protein_classes = pickle.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train KNN classifier
    print("\nTraining KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}\n")
    
    # Generate classification report
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [protein_classes.get(cls, f"Class {cls}") for cls in unique_classes]
    
    print("Detailed Classification Report:")
    report = classification_report(y_test, y_pred, 
                                 labels=unique_classes,
                                 target_names=target_names, 
                                 zero_division=0)
    print(report)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, 
                       index=[protein_classes.get(cls, f"Class {cls}") for cls in unique_classes],
                       columns=[protein_classes.get(cls, f"Class {cls}") for cls in unique_classes])
    
    print("Confusion Matrix:")
    print(cm_df)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('KNN Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/knn_confusion_matrix.png')
    
    # Identify important features (using K-nearest distances indirectly)
    if len(X_test) > 0:
        # Get the nearest neighbors for each test sample
        distances, indices = knn.kneighbors(X_test)
        
        # Load k-mer features
        with open('outputs/kmer_features.pkl', 'rb') as f:
            kmer_features = pickle.load(f)
        
        # Create a feature importance plot based on the features that contribute most to correct predictions
        feature_importance = np.zeros(X_train.shape[1])
        correct_predictions = y_test == y_pred
        
        if sum(correct_predictions) > 0:
            # Only analyze correctly predicted samples
            correct_samples = X_test[correct_predictions]
            correct_indices = indices[correct_predictions]
            
            # For each correct prediction, find which features were most similar to its nearest neighbor
            for i, sample in enumerate(correct_samples):
                neighbor_idx = correct_indices[i, 0]  # Get the index of the nearest neighbor
                neighbor = X_train[neighbor_idx]
                
                # Calculate feature-wise similarity (smaller difference means more important)
                feature_diff = np.abs(sample - neighbor)
                feature_importance += (1 / (feature_diff + 1e-10))  # Avoid division by zero
            
            # Normalize feature importance
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
                
                # Create and save feature importance plot
                top_n = 20  # Show top 20 features
                indices = np.argsort(feature_importance)[::-1][:top_n]
                
                plt.figure(figsize=(10, 8))
                plt.title('Top 20 K-mer Features for KNN Classification')
                plt.bar(range(top_n), feature_importance[indices])
                plt.xticks(range(top_n), [kmer_features[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('plots/knn_feature_importance.png')
    
    # Save the model
    with open('outputs/knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    
    # Save protein class mapping again (in case it was modified)
    with open('outputs/protein_classes.pkl', 'wb') as f:
        pickle.dump(protein_classes, f)
    
    print("\nKNN model saved to 'outputs/knn_model.pkl'")
    print("Protein class mapping saved to 'outputs/protein_classes.pkl'")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 