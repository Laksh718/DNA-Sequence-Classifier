#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest Classification for DNA Sequences

This script implements a Random Forest classifier for DNA sequence classification
based on k-mer features.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
import joblib

def load_data():
    """Load the preprocessed data."""
    X_train = np.load(os.path.join('outputs', 'X_train.npy'))
    X_test = np.load(os.path.join('outputs', 'X_test.npy'))
    y_train = np.load(os.path.join('outputs', 'y_train.npy'))
    y_test = np.load(os.path.join('outputs', 'y_test.npy'))
    
    with open(os.path.join('outputs', 'kmer_features.pkl'), 'rb') as f:
        kmer_features = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, kmer_features

def optimize_hyperparameters(X_train, y_train):
    """Find the optimal hyperparameters for Random Forest classifier using cross-validation."""
    # For small datasets, use a simpler parameter grid
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Use simple cross-validation for very small datasets
    if len(np.unique(y_train)) < 3:
        cv = 2
    else:
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}\n")
    
    return grid_search.best_params_

def train_and_evaluate_rf(X_train, X_test, y_train, y_test, best_params, kmer_features):
    """Train and evaluate the Random Forest classifier."""
    # Define protein class names
    protein_classes = {
        0: "GPCRs",
        1: "Tyrosine kinase",
        2: "Protein phosphatases",
        3: "PTPs",
        4: "AARSs",
        5: "Ion channels",
        6: "Transcription Factor"
    }
    
    # Train the model
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(**best_params, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Get unique classes from both test and predicted labels
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=[protein_classes[i] for i in unique_classes],
        output_dict=True
    )
    print(classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=[protein_classes[i] for i in unique_classes]
    ))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[protein_classes[i] for i in unique_classes],
        columns=[protein_classes[i] for i in unique_classes]
    )
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances for Random Forest Classifier')
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), [kmer_features[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'rf_feature_importance.png'))
    plt.close()
    
    # Save top features to CSV
    top_features = pd.DataFrame({
        'K-mer': [kmer_features[i] for i in indices[:20]],
        'Importance': importances[indices[:20]]
    })
    top_features.to_csv(os.path.join('outputs', 'rf_top_features.csv'), index=False)
    
    # Save the protein class mapping
    joblib.dump(protein_classes, os.path.join('outputs', 'protein_classes.pkl'))
    print("Protein class mapping saved to 'outputs/protein_classes.pkl'")
    
    return rf

def main():
    start_time = time.time()
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, kmer_features = load_data()
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Loaded testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Optimize hyperparameters
    print("\nOptimizing hyperparameters for Random Forest classifier...")
    best_params = optimize_hyperparameters(X_train, y_train)
    
    # Train and evaluate the model
    print("\nTraining Random Forest classifier with optimal hyperparameters...")
    rf_model = train_and_evaluate_rf(X_train, X_test, y_train, y_test, best_params, kmer_features)
    
    # Save the model
    joblib.dump(rf_model, os.path.join('outputs', 'rf_model.pkl'))
    print("\nRandom Forest model saved to 'outputs/rf_model.pkl'")
    
    print("\nRandom Forest classification completed.")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 