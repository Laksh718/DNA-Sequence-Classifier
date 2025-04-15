#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate k-mer Dataset from DNA Sequences

This script processes DNA sequences to create feature vectors based on k-mer frequencies.
K-mers are subsequences of length k from a DNA sequence, and their frequencies can be used
as features for machine learning models.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from Bio import SeqIO
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_kmers(sequence, k=3):
    """Generate all k-mers of length k from a sequence."""
    kmers = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1
    return kmers

def create_feature_matrix(sequences, class_labels, k=3):
    """Create a feature matrix of k-mer frequencies for all sequences."""
    # Generate all possible k-mers
    nucleotides = ['A', 'T', 'G', 'C']
    all_possible_kmers = []
    
    def generate_all_kmers(prefix, k):
        if k == 0:
            all_possible_kmers.append(prefix)
            return
        for nucleotide in nucleotides:
            generate_all_kmers(prefix + nucleotide, k - 1)
    
    generate_all_kmers('', k)
    
    # Create feature matrix
    X = np.zeros((len(sequences), len(all_possible_kmers)))
    y = np.array(class_labels)
    
    # Fill feature matrix with k-mer frequencies
    for i, seq in enumerate(sequences):
        kmers = generate_kmers(seq, k)
        for j, kmer in enumerate(all_possible_kmers):
            if kmer in kmers:
                X[i, j] = kmers[kmer] / (len(seq) - k + 1)  # Normalize by sequence length
    
    return X, y, all_possible_kmers

def main():
    start_time = time.time()
    
    # Load sequences from FASTA file
    sequences = []
    class_labels = []
    protein_classes = {}
    
    for record in SeqIO.parse('data/sequences.fasta', 'fasta'):
        description_parts = record.description.split('|')
        if len(description_parts) >= 3:
            protein_class = description_parts[1]
            class_label = int(description_parts[2])
            protein_classes[class_label] = protein_class
            sequences.append(str(record.seq))
            class_labels.append(class_label)
    
    print(f"Loaded {len(sequences)} sequences with {len(set(class_labels))} different classes")
    
    # Generate k-mer features
    k = 3
    print(f"Generating {k}-mer features...")
    X, y, kmers = create_feature_matrix(sequences, class_labels, k)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of {k}-mer features: {len(kmers)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    with open('outputs/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save data
    np.save('outputs/X_train.npy', X_train_scaled)
    np.save('outputs/X_test.npy', X_test_scaled)
    np.save('outputs/y_train.npy', y_train)
    np.save('outputs/y_test.npy', y_test)
    
    # Save protein class mapping
    with open('outputs/protein_classes.pkl', 'wb') as f:
        pickle.dump(protein_classes, f)
    
    # Save feature names
    with open('outputs/kmer_features.pkl', 'wb') as f:
        pickle.dump(kmers, f)
    
    # Save a sample of the feature matrix as CSV for inspection
    df_sample = pd.DataFrame(X[:5], columns=kmers)
    df_sample['class'] = y[:5]
    df_sample.to_csv('outputs/kmer_features_sample.csv', index=False)
    
    print("\nData processing completed. Files saved to 'outputs/' directory.")
    print(f"Sample of k-mer features saved to 'outputs/kmer_features_sample.csv'")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 