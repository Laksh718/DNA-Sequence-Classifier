import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

def create_feature_matrix(sequences, k=3):
    """Create a feature matrix of k-mer frequencies for sequences."""
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
    
    # Fill feature matrix with k-mer frequencies
    for i, seq in enumerate(sequences):
        kmers = generate_kmers(seq, k)
        for j, kmer in enumerate(all_possible_kmers):
            if kmer in kmers:
                X[i, j] = kmers[kmer] / (len(seq) - k + 1)  # Normalize by sequence length
    
    return X, all_possible_kmers

def get_protein_description(protein_class):
    """Return a human-readable description of each protein class."""
    descriptions = {
        "GPCRs": "G Protein-Coupled Receptors - Cell membrane proteins important for communication between cells. Essential for vision, smell, and hormone responses.",
        "Tyrosine kinase": "Tyrosine Kinases - Enzymes that add phosphate groups to proteins. Critical for cell growth, division, and communication.",
        "Protein phosphatases": "Protein Phosphatases - Enzymes that remove phosphate groups from proteins. Help regulate many cell processes.",
        "PTPs": "Protein Tyrosine Phosphatases - Specialized enzymes that remove phosphates from specific amino acids. Important for immune system function.",
        "AARSs": "Aminoacyl-tRNA Synthetases - Essential enzymes that help build proteins by connecting amino acids to RNA molecules.",
        "Ion channels": "Ion Channels - Proteins that create pores in cell membranes, allowing ions to pass through. Critical for nerve signals and muscle contraction.",
        "Transcription Factor": "Transcription Factors - Proteins that control which genes are turned on or off. Help determine cell type and function."
    }
    
    if protein_class in descriptions:
        return descriptions[protein_class]
    return "Unknown protein class"

def main():
    parser = argparse.ArgumentParser(description='Predict protein classes from DNA sequences')
    parser.add_argument('--input', type=str, required=True, help='Input FASTA file with DNA sequences')
    parser.add_argument('--model', type=str, default='knn', choices=['knn', 'rf'], help='Model to use (knn or rf)')
    parser.add_argument('--kmer', type=int, default=3, help='K-mer length (default: 3)')
    parser.add_argument('--output', type=str, help='Output file for results (default: predictions.csv)')
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Check if the model exists
    model_file = f"outputs/{args.model}_model.pkl"
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found")
        sys.exit(1)
    
    # Check if the scaler exists
    scaler_file = "outputs/feature_scaler.pkl"
    if not os.path.exists(scaler_file):
        print(f"Error: Scaler file '{scaler_file}' not found")
        sys.exit(1)
    
    # Load the protein class mapping
    class_file = "outputs/protein_classes.pkl"
    if not os.path.exists(class_file):
        print(f"Error: Class mapping file '{class_file}' not found")
        sys.exit(1)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Load sequences from FASTA file
    sequences = []
    sequence_ids = []
    true_classes = []
    
    print("\n=== DNA SEQUENCE CLASSIFICATION ===")
    print("This tool analyzes DNA sequences to predict which type of protein they encode.\n")
    
    for record in SeqIO.parse(args.input, 'fasta'):
        sequence_ids.append(record.id)
        sequences.append(str(record.seq))
        
        # Try to extract true class from header if available
        try:
            class_label = int(record.id.split('|')[-1])
            true_classes.append(class_label)
        except (ValueError, IndexError):
            true_classes.append(None)
    
    if len(sequences) == 0:
        print("Error: No sequences found in the input file")
        sys.exit(1)
    
    print(f"✓ Loaded {len(sequences)} DNA sequences from {args.input}")
    print(f"✓ Using {args.kmer}-mer patterns to analyze sequences")
    print(f"✓ Selected classification model: {args.model.upper()}")
    
    # Generate k-mer features
    print("\nAnalyzing DNA patterns...")
    X, kmers = create_feature_matrix(sequences, args.kmer)
    
    # Load the scaler
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Load the model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load the protein class mapping
    with open(class_file, 'rb') as f:
        protein_classes = pickle.load(f)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_scaled)
    
    # For KNN, get prediction probabilities if available
    probabilities = None
    confidences = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Sequence_ID': sequence_ids,
        'Predicted_Class_ID': y_pred,
        'Predicted_Class': [protein_classes.get(cls, f"Unknown Class {cls}") for cls in y_pred]
    })
    
    # Add confidence scores if available
    if confidences is not None:
        results['Confidence'] = confidences
    
    # Add true classes if available
    if all(cls is not None for cls in true_classes):
        results['True_Class_ID'] = true_classes
        results['True_Class'] = [protein_classes.get(cls, f"Unknown Class {cls}") for cls in true_classes]
        results['Correct'] = results['Predicted_Class_ID'] == results['True_Class_ID']
    
    # Display results in a human-readable format
    print("\n=== PREDICTION RESULTS ===\n")
    
    # Summary statistics
    if 'True_Class' in results.columns:
        accuracy = results['Correct'].mean() * 100
        print(f"Overall accuracy: {accuracy:.1f}% ({sum(results['Correct'])} correct out of {len(results)} sequences)\n")
    
    # Class distribution
    print("Predicted Class Distribution:")
    class_counts = results['Predicted_Class'].value_counts()
    for cls, count in class_counts.items():
        print(f"• {cls}: {count} sequences ({count/len(sequences)*100:.1f}%)")
    
    # Detailed results for each sequence
    print("\nDetailed Results for Each Sequence:")
    for i, row in results.iterrows():
        print(f"\nSequence {i+1}: {row['Sequence_ID']}")
        print(f"  Predicted: {row['Predicted_Class']}")
        if 'True_Class' in results.columns:
            print(f"  Actual: {row['True_Class']}")
            print(f"  Correct: {'✓' if row['Correct'] else '✗'}")
        
        if 'Confidence' in results.columns:
            confidence = row['Confidence'] * 100
            print(f"  Confidence: {confidence:.1f}%")
        
        # Add protein description
        description = get_protein_description(row['Predicted_Class'])
        print(f"  What this means: {description}")
    
    # Generate a visualization of results
    if 'True_Class' in results.columns:
        # Create confusion matrix
        true_classes_list = results['True_Class'].unique()
        pred_classes_list = results['Predicted_Class'].unique()
        class_labels = sorted(list(set(true_classes_list) | set(pred_classes_list)))
        
        # Create plot
        plt.figure(figsize=(10, 8))
        conf_matrix = pd.crosstab(results['True_Class'], results['Predicted_Class'])
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=True, yticklabels=True)
        plt.title('Confusion Matrix: True vs Predicted Classes')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        # Save plot
        confusion_matrix_file = 'outputs/confusion_matrix.png'
        plt.savefig(confusion_matrix_file)
        print(f"\nConfusion matrix saved to {confusion_matrix_file}")
    
    # Save results to CSV
    output_file = args.output if args.output else f"outputs/predictions_{args.model}.csv"
    results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    # Include explanations for different confidence levels
    if 'Confidence' in results.columns:
        print("\n=== UNDERSTANDING CONFIDENCE SCORES ===")
        print("• 90-100%: Very high confidence - prediction is very likely correct")
        print("• 70-90%: High confidence - prediction is likely correct")
        print("• 50-70%: Moderate confidence - prediction may be correct")
        print("• Below 50%: Low confidence - prediction is uncertain")
    
    # Provide a conclusion and next steps
    print("\n=== SUMMARY ===")
    print("This analysis identified the most likely protein type encoded by each DNA sequence.")
    print("The classification is based on recognizing specific patterns in the DNA that are")
    print("characteristic of different protein families.")
    
    print("\n=== NEXT STEPS ===")
    print("1. For sequences with high confidence predictions, you can consider these")
    print("   classifications reliable for further research or applications.")
    print("2. For low confidence predictions, consider using longer DNA sequences")
    print("   or alternative classification methods for validation.")
    print("3. Use these predictions to guide your understanding of the biological")
    print("   function of these sequences in your research.")

if __name__ == "__main__":
    main() 