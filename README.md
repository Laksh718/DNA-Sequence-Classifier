# DNA Sequence Classification Model

## Project Overview

This project focuses on classifying DNA sequences into 7 protein classes using k-mer feature extraction and machine learning models. The system analyzes nucleotide patterns to identify functional protein categories without requiring full protein translation.

## Protein Classes

The model identifies 7 distinct protein classes based on DNA sequence patterns:

1. **GPCRs (G Protein-Coupled Receptors)**

   - Cell membrane receptors involved in signal transduction
   - Critical for sensory perception, hormone response, and neurotransmission
   - Examples: Rhodopsin, Adrenergic receptors, Opioid receptors

2. **Tyrosine Kinases**

   - Enzymes that transfer phosphate groups to tyrosine residues
   - Key roles in cellular growth, differentiation, and metabolism
   - Examples: Insulin receptor, Epidermal Growth Factor Receptor (EGFR)

3. **Protein Phosphatases**

   - Enzymes that remove phosphate groups from proteins
   - Regulate cell signaling by counterbalancing kinase activity
   - Examples: PP1, PP2A, PTEN

4. **PTPs (Protein Tyrosine Phosphatases)**

   - Specialized phosphatases that specifically remove phosphates from tyrosine residues
   - Important in immune response and cell cycle regulation
   - Examples: PTP1B, CD45, SHP-2

5. **AARSs (Aminoacyl-tRNA Synthetases)**

   - Essential enzymes that attach amino acids to their corresponding tRNAs
   - Critical for accurate protein synthesis
   - Examples: Alanyl-tRNA synthetase, Histidyl-tRNA synthetase

6. **Ion Channels**

   - Membrane proteins that allow ions to pass through cellular membranes
   - Essential for nerve impulse transmission and muscle contraction
   - Examples: Sodium channels, Potassium channels, Calcium channels

7. **Transcription Factors**
   - Proteins that regulate gene expression by binding to DNA
   - Control cell development, differentiation, and response to environmental stimuli
   - Examples: TATA-binding protein, p53, NF-κB

## Applications

This DNA classification model has several important applications:

- **Drug Discovery**: Identifying protein classes from DNA sequences can accelerate target identification for drug development.
- **Genetic Diagnostics**: Detecting functional elements in genetic sequences can help identify disease-related mutations.
- **Genomic Analysis**: Rapidly categorizing gene functions in newly sequenced genomes.
- **Protein Engineering**: Understanding DNA patterns that correspond to specific protein functions can guide protein design.
- **Evolutionary Biology**: Studying the conservation of functional DNA patterns across species.

## How It Works

1. **K-mer Feature Extraction**: The model breaks DNA sequences into k-mers (subsequences of length k) and counts their frequencies.

2. **Feature Scaling**: K-mer frequencies are normalized to account for sequence length variations.

3. **Classification Models**:

   - **K-Nearest Neighbors (KNN)**: Classifies sequences based on similarity to known examples.
   - **Random Forest**: Uses decision trees to identify discriminative k-mer patterns.

4. **Evaluation Metrics**:
   - **Accuracy**: Overall percentage of correctly classified sequences.
   - **Precision**: Proportion of predicted positives that are actually positive.
   - **Recall**: Proportion of actual positives that were correctly identified.
   - **F1-Score**: Harmonic mean of precision and recall.
   - **Confusion Matrix**: Visualization of true vs. predicted classes.

## Interpreting Results

When using the model, look for:

1. **Distinct Starting Patterns**: Each protein class has characteristic DNA motifs:

   - GPCRs typically start with "ATG"
   - Tyrosine kinases often begin with "GCT"
   - Protein phosphatases commonly start with "CG"
   - PTPs frequently start with "TACG"
   - AARSs typically begin with "GAT"
   - Ion channels often start with "CTAC"
   - Transcription factors frequently begin with "ACGT"

2. **Feature Importance**: The most discriminative k-mers for classification.

3. **Confidence Scores**: Probability estimates for each class prediction.

## Getting Started

```bash
# Run the full pipeline from raw sequences to model evaluation
python run_pipeline.py

# For prediction on new sequences
python predict.py --input your_sequences.fasta --model knn --kmer 3
```

## License

MIT License
