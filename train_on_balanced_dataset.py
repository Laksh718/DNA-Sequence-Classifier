import os
import pandas as pd
import numpy as np
import pickle
import joblib
from Bio import SeqIO
from io import StringIO
from collections import defaultdict
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Make sure the outputs directory exists
os.makedirs('outputs', exist_ok=True)

def generate_kmers(k):
    """Generate all possible k-mers of length k."""
    nucleotides = ['A', 'C', 'G', 'T']
    return [''.join(i) for i in itertools.product(nucleotides, repeat=k)]

def count_kmers(sequence, k):
    """Count occurrences of all possible k-mers in a sequence."""
    kmer_counts = defaultdict(int)
    all_kmers = generate_kmers(k)
    for kmer in all_kmers:
        kmer_counts[kmer] = 0
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(nucleotide in 'ACGT' for nucleotide in kmer):
            kmer_counts[kmer] += 1
    
    return kmer_counts

def create_kmer_features(sequences, k):
    """Create a feature matrix of k-mer frequencies for all sequences."""
    all_kmers = generate_kmers(k)
    X = np.zeros((len(sequences), len(all_kmers)))
    
    for i, seq in enumerate(sequences):
        kmer_counts = count_kmers(seq, k)
        for j, kmer in enumerate(all_kmers):
            X[i, j] = kmer_counts[kmer]
    
    return X, all_kmers

def parse_fasta_with_class(fasta_path):
    """Parse a FASTA file containing DNA sequences with class labels."""
    sequences = []
    classes = []
    headers = []
    
    with open(fasta_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            header = record.id
            sequence = str(record.seq)
            
            # Extract class from the header (format: sequence_name|protein_type|class_number)
            try:
                class_label = int(header.split('|')[-1])
                classes.append(class_label)
                sequences.append(sequence)
                headers.append(header)
            except (ValueError, IndexError):
                print(f"Error parsing class label from header: {header}")
    
    return sequences, classes, headers

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save confusion matrix."""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join('outputs', f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()

def plot_feature_importance(feature_importance, feature_names, model_name):
    """Plot and save feature importance."""
    # Get top 20 features
    indices = np.argsort(feature_importance)[-20:]
    top_features = [feature_names[i] for i in indices]
    top_importance = feature_importance[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importance, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.title(f'Top 20 Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join('outputs', f'{model_name.lower().replace(" ", "_")}_feature_importance.png'))
    plt.close()

def main():
    print("Starting DNA sequence classification training...")
    start_time = time.time()
    
    # Set k-mer length
    k = 3
    print(f"Using k-mer length of {k}...")
    
    # Load and parse the FASTA file
    fasta_path = 'datasets/protein_classification/balanced_dataset.fasta'
    sequences, classes, headers = parse_fasta_with_class(fasta_path)
    
    print(f"Loaded {len(sequences)} sequences with {len(set(classes))} different classes")
    
    # Generate k-mer features
    print(f"Generating {k}-mer features...")
    X, kmer_features = create_kmer_features(sequences, k)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    with open(os.path.join('outputs', 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save a sample of the features for inspection
    feature_df = pd.DataFrame(X, columns=kmer_features)
    feature_df['Class'] = classes
    feature_df.head(10).to_csv(os.path.join('outputs', 'kmer_features_sample.csv'), index=False)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, classes, test_size=0.2, random_state=42, stratify=classes
    )
    
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    
    # Train KNN model
    print("Training KNN model...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_knn = grid_search.best_estimator_
    print(f"Best KNN parameters: {grid_search.best_params_}")
    
    # Evaluate KNN
    y_pred_knn = best_knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Accuracy: {accuracy_knn:.4f}")
    
    # Define class names for the reports
    class_names = {
        0: "GPCRs",
        1: "Tyrosine kinase",
        2: "Protein phosphatases",
        3: "PTPs",
        4: "AARSs",
        5: "Ion channels",
        6: "Transcription Factor"
    }
    
    # Create classification report
    target_names = [class_names[i] for i in sorted(set(classes))]
    print("\nClassification Report (KNN):")
    print(classification_report(y_test, y_pred_knn, target_names=target_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_knn, target_names, "KNN")
    
    # Save the KNN model
    joblib.dump(best_knn, os.path.join('outputs', 'knn_model.pkl'))
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)
    
    best_rf = grid_search_rf.best_estimator_
    print(f"Best Random Forest parameters: {grid_search_rf.best_params_}")
    
    # Evaluate Random Forest
    y_pred_rf = best_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
    
    # Create classification report
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=target_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_rf, target_names, "Random Forest")
    
    # Plot feature importance for Random Forest
    plot_feature_importance(best_rf.feature_importances_, kmer_features, "Random Forest")
    
    # Save the Random Forest model
    joblib.dump(best_rf, os.path.join('outputs', 'rf_model.pkl'))
    
    # Training complete
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Models saved to 'outputs' directory")

if __name__ == "__main__":
    main() 