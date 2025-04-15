#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis for DNA Sequences

This script explores the DNA sequence data to gain insights into the dataset properties.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def parse_fasta_with_class(file_path):
    """Parse a FASTA file containing DNA sequences with class labels."""
    sequences = []
    classes = []
    
    with open(file_path, 'r') as file:
        seq = ""
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if seq:  # If we've already read a sequence, store it
                    sequences.append(seq)
                    seq = ""
                # Extract class label from header
                class_label = int(line.split('|')[-1])
                classes.append(class_label)
            else:
                seq += line
        # Don't forget the last sequence
        if seq:
            sequences.append(seq)
    
    return sequences, classes

def main():
    # Path to the dataset
    file_path = 'datasets/human.txt'  # Updated path
    
    # Parse the FASTA file
    sequences, classes = parse_fasta_with_class(file_path)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Sequence': sequences,
        'Class': classes
    })
    
    # Display basic info
    print("Dataset Shape:", data.shape)
    print("\nClass Distribution:")
    print(data['Class'].value_counts().sort_index())
    
    # Map class labels to gene family names
    class_names = {
        0: 'G protein-coupled receptors (GPCRs)',
        1: 'Tyrosine kinase',
        2: 'Protein tyrosine phosphatases',
        3: 'Protein tyrosine phosphatases (PTPs)',
        4: 'Aminoacyl-tRNA synthetases (AARSs)',
        5: 'Ion channels',
        6: 'Transcription Factor'
    }
    
    # Calculate sequence lengths
    data['Length'] = data['Sequence'].apply(len)
    
    # Summary statistics for sequence length
    print("\nSequence Length Statistics:")
    print(data['Length'].describe())
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot sequence length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Length'], bins=30, kde=True)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Frequency')
    plt.savefig('plots/sequence_length_distribution.png')
    plt.close()
    
    # Nucleotide composition analysis
    def nucleotide_composition(sequence):
        """Calculate the composition of A, T, G, C in a sequence."""
        length = len(sequence)
        return {
            'A': sequence.count('A') / length,
            'T': sequence.count('T') / length,
            'G': sequence.count('G') / length,
            'C': sequence.count('C') / length
        }
    
    # Calculate nucleotide composition for each sequence
    compositions = data['Sequence'].apply(nucleotide_composition)
    
    # Extract compositions into DataFrame
    comp_df = pd.DataFrame(compositions.tolist())
    data = pd.concat([data, comp_df], axis=1)
    
    # Calculate GC content
    data['GC_Content'] = data['G'] + data['C']
    
    # Plot GC content distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data['GC_Content'], bins=30, kde=True)
    plt.title('Distribution of GC Content')
    plt.xlabel('GC Content')
    plt.ylabel('Frequency')
    plt.savefig('plots/gc_content_distribution.png')
    plt.close()
    
    # GC content by class
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='GC_Content', data=data)
    plt.title('GC Content by Gene Class')
    plt.xlabel('Gene Class')
    plt.ylabel('GC Content')
    plt.xticks(ticks=range(len(class_names)), labels=[f"{i}: {class_names[i]}" for i in range(len(class_names))], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/gc_content_by_class.png')
    plt.close()
    
    print("\nEDA completed. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    main() 