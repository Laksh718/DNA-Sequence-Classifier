#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN Classification for DNA Sequences

This script implements a K-Nearest Neighbors classifier for DNA sequence classification
based on k-mer features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
import joblib

def load_data():
    """Load the preprocessed data."""
    X_train = np.load('../outputs/X_train.npy')
    X_test = np.load('../outputs/X_test.npy')
    y_train = np.load('../outputs/y_train.npy')
    y_test = np.load('../outputs/y_test.npy')
    
    with open('../outputs/kmer_features.pkl', 'rb') as f:
        kmer_features = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, kmer_features

def find_optimal_k(X_train, y_train):
    """Find the optimal k value for KNN classifier using cross-validation."""
    # For small datasets, limit k to at most n_samples/2
    max_k = min(3, len(X_train) // 2)
    param_grid = {'n_neighbors': range(1, max_k + 1)}
    knn = KNeighborsClassifier()
    
    # Use simple cross-validation for very small datasets
    if len(np.unique(y_train)) < 3:
        cv = 2
    else:
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        
    grid_search = GridSearchCV(
        knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}\n")
    
    return grid_search.best_params_['n_neighbors']

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k):
    """Train and evaluate KNN classifier."""
    print(f"Training KNN classifier with k={k}...")
    
    # Train the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}\n")
    
    # Get unique classes from both train and test sets
    unique_classes = sorted(list(set(np.unique(y_train)) | set(np.unique(y_test))))
    target_names = [f"Class {i}" for i in unique_classes]
    
    print("Classification Report:")
    try:
        report = classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names)
        print(report)
    except Exception as e:
        print("Could not generate classification report due to limited test data")
        print(f"Predicted classes: {np.unique(y_pred)}")
        print(f"Actual classes: {np.unique(y_test)}")
    
    return knn

def main():
    # Create output directories
    os.makedirs('../outputs', exist_ok=True)
    os.makedirs('../plots', exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, kmer_features = load_data()
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Loaded testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Find optimal k
    print("\nFinding optimal k for KNN classifier...")
    optimal_k = find_optimal_k(X_train, y_train)
    
    # Train and evaluate the model
    print(f"\nTraining KNN classifier with k={optimal_k}...")
    knn_model = train_and_evaluate_knn(X_train, X_test, y_train, y_test, optimal_k)
    
    # Save the model
    joblib.dump(knn_model, '../outputs/knn_model.pkl')
    print("\nKNN model saved to '../outputs/knn_model.pkl'")
    
    print("\nKNN classification completed.")

if __name__ == "__main__":
    main() 