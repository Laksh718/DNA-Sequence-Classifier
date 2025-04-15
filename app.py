import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import joblib
from Bio import SeqIO
from io import StringIO
from collections import defaultdict, Counter
import itertools
from sklearn.preprocessing import StandardScaler
import time

# Try to import TensorFlow, but don't fail if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="DNA Sequence Classifier",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global text settings for maximum readability - Dark mode */
    body {
        font-family: Arial, sans-serif;
        font-size: 18px;
        line-height: 1.6;
        color: #ffffff;
        background-color: #121212;
    }
    
    .main {
        padding: 2rem;
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Make titles stand out */
    .stTitle {
        font-size: 3.5rem !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Team info section */
    .team-info {
        padding: 1.5rem;
        background-color: #1e1e1e;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.5);
        color: #ffffff;
        border: 2px solid #4c8bf5;
    }
    
    .team-info h2 {
        color: #4c8bf5;
        margin-bottom: 1rem;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .team-info ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .team-info li {
        margin: 0.5rem 0;
        font-size: 1.2rem;
        color: #ffffff;
    }
    
    .team-info p {
        margin: 0.5rem 0;
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    /* Feature cards */
    .feature-card {
        padding: 1.5rem;
        background-color: #1e1e1e;
        border-radius: 5px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.5);
        margin: 1rem 0;
        border-left: 6px solid #4c8bf5;
    }
    
    .feature-card h3 {
        color: #4c8bf5;
        margin-bottom: 0.8rem;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .feature-card p {
        color: #ffffff;
        line-height: 1.6;
        font-size: 1.1rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 2px solid #4c8bf5;
        color: #ffffff;
    }
    
    .info-box h2 {
        color: #4c8bf5;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 2px solid #4c8bf5;
        color: #ffffff;
    }
    
    .prediction-box h4 {
        color: #4c8bf5;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .prediction-box p {
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4c8bf5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 12px 24px;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        background-color: #3a7be0;
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    
    /* Improve text readability */
    p, li {
        color: #ffffff;
        line-height: 1.6;
        font-size: 18px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #4c8bf5;
        font-weight: 700;
    }
    
    /* Make main headers more prominent */
    h1 {
        font-size: 2.5rem;
    }
    
    h2 {
        font-size: 2rem;
    }
    
    h3 {
        font-size: 1.75rem;
    }
    
    h4 {
        font-size: 1.5rem;
    }
    
    /* Fix for dataframes */
    .dataframe {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    .dataframe th {
        background-color: #4c8bf5;
        color: white;
        text-align: left;
        padding: 12px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .dataframe td {
        border: 1px solid #333333;
        padding: 12px;
        color: #ffffff;
        font-size: 1rem;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #2a2a2a;
    }
    
    /* Force high contrast text everywhere */
    div.stMarkdown {
        color: #ffffff !important;
    }
    
    /* Fix all markdown element colors */
    .stMarkdown p, .stMarkdown li {
        color: #ffffff !important;
        font-size: 18px;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #4c8bf5 !important;
        font-weight: 700;
    }
    
    /* Other Streamlit components */
    .st-bx {
        color: #ffffff;
        background-color: #1e1e1e;
    }
    
    /* Labels for inputs */
    .stSelectbox label, .stSlider label, .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #121212;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4c8bf5;
        border-radius: 4px 4px 0 0;
    }
    
    /* Sidebar */
    .st-emotion-cache-fblp2m {
        color: #ffffff !important;
        font-size: 1.1rem;
    }
    
    .st-emotion-cache-16txtl3 {
        font-weight: 600;
        font-size: 1.1rem;
        color: #ffffff !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        font-size: 1rem;
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    /* Metrics */
    .stMetric label {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .stMetric value {
        color: #4c8bf5 !important;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Alerts and messages */
    .stAlert {
        padding: 16px;
        border-radius: 8px;
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    .stAlert p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
    }
    
    /* File uploader */
    .stFileUploader {
        padding: 15px;
        border-radius: 8px;
        border: 2px dashed #4c8bf5;
        background-color: #1e1e1e;
    }
    
    /* Fix any Streamlit-specific components */
    .st-emotion-cache-16idsys p {
        color: #ffffff !important;
    }
    
    .st-emotion-cache-183lzff {
        color: #ffffff;
    }
    
    /* Change the widget backgrounds */
    .stTextInput > div > div, .stSelectbox > div > div {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    /* Fix for expanders */
    .streamlit-expanderHeader {
        background-color: #1e1e1e;
        color: #ffffff !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Make sure checkbox labels are visible */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Override the default background of the app */
    section[data-testid="stSidebar"] {
        background-color: #121212;
        color: #ffffff;
    }
    
    section.main {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Plots - ensure they have dark backgrounds */
    .stPlot {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Container styles */
    .stContainer, .stColumn, .stColumns {
        color: #ffffff !important;
        background-color: #121212;
    }
    
    /* Fix any dropdowns */
    div[data-baseweb="select"] {
        background-color: #2a2a2a;
    }
    
    div[data-baseweb="popover"] {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    div[data-baseweb="menu"] {
        background-color: #2a2a2a;
    }
    
    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }
    
    /* Fix for code syntax highlighting */
    .language-python, .language-bash {
        color: #ffffff;
        background-color: #2a2a2a;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Team Information
st.title("🧬 DNA Sequence Classifier")

# Team Information
st.markdown("""
<div class="team-info">
<h2>🎓 Team Information - Group 5</h2>
<p><strong>Project By:</strong></p>
<ul>
    <li>👨‍💻 Laksh Baweja</li>
    <li>👨‍💻 Krish</li>
    <li>👨‍💻 Yadhi</li>
    <li>👨‍💻 Priyanshu</li>
    <li>👨‍💻 Naman</li>
    <li>👨‍💻 Deepansh</li>
</ul>
<p><em>Department of Computer Science and Engineering</em></p>
</div>
""", unsafe_allow_html=True)

# Protein Class Explanations
st.markdown("""
<div class="info-box">
<h2>🧪 Understanding Protein Classes</h2>
<p>This tool classifies DNA sequences into 7 protein types. Here's what each type does:</p>

<div class="feature-card">
    <h3>G Protein-Coupled Receptors (GPCRs)</h3>
    <p>Cell membrane proteins that act like antennas for cells. They detect signals from outside the cell (like hormones, light, scents) and tell the cell how to respond.</p>
    <p><strong>Real-world relevance:</strong> Over 30% of all modern medicines target GPCRs, including drugs for allergies, pain, and high blood pressure.</p>
</div>

<div class="feature-card">
    <h3>Tyrosine Kinases</h3>
    <p>Enzymes that add phosphate groups (like "on" switches) to proteins in the cell. They help control cell growth, division, and many other important functions.</p>
    <p><strong>Real-world relevance:</strong> Many cancer treatments target tyrosine kinases because they can be overactive in cancer cells.</p>
</div>

<div class="feature-card">
    <h3>Protein Phosphatases</h3>
    <p>Enzymes that remove phosphate groups (like "off" switches) from proteins. They work in balance with kinases to control cellular processes.</p>
    <p><strong>Real-world relevance:</strong> Mutations in these proteins are linked to several diseases including diabetes and cancer.</p>
</div>

<div class="feature-card">
    <h3>Protein Tyrosine Phosphatases (PTPs)</h3>
    <p>A specialized type of phosphatases that specifically remove phosphates from the amino acid tyrosine. They're critically important in immune cell function.</p>
    <p><strong>Real-world relevance:</strong> These proteins play major roles in immune disorders and are targets for autoimmune disease treatments.</p>
</div>

<div class="feature-card">
    <h3>Aminoacyl-tRNA Synthetases (AARSs)</h3>
    <p>Enzymes that help build proteins by attaching the correct amino acids to carrier molecules (tRNAs) during protein synthesis.</p>
    <p><strong>Real-world relevance:</strong> Defects in these proteins can cause a variety of neurological and metabolic disorders.</p>
</div>

<div class="feature-card">
    <h3>Ion Channels</h3>
    <p>Proteins that form pores in cell membranes, allowing specific ions (like sodium or potassium) to pass through. Essential for nerve signals and muscle contraction.</p>
    <p><strong>Real-world relevance:</strong> Many medications for heart conditions, epilepsy, and pain target ion channels.</p>
</div>

<div class="feature-card">
    <h3>Transcription Factors</h3>
    <p>Proteins that control which genes are active or inactive. They act like "on/off switches" for gene expression.</p>
    <p><strong>Real-world relevance:</strong> Many developmental disorders and cancers are linked to problems with transcription factors.</p>
</div>
</div>
""", unsafe_allow_html=True)

def calculate_sequence_stats(sequence):
    """Calculate detailed statistics for a DNA sequence."""
    length = len(sequence)
    gc_content = (sequence.count('G') + sequence.count('C')) / length
    at_content = (sequence.count('A') + sequence.count('T')) / length
    stats = {
        'Length': length,
        'GC Content': f"{gc_content:.2%}",
        'AT Content': f"{at_content:.2%}",
        'A Count': sequence.count('A'),
        'T Count': sequence.count('T'),
        'G Count': sequence.count('G'),
        'C Count': sequence.count('C')
    }
    return stats

def parse_fasta_with_class(fasta_content):
    """Parse a FASTA string containing DNA sequences with class labels."""
    sequences = []
    classes = []
    headers = []
    
    for line in fasta_content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if sequences and sequences[-1]:
                sequences.append("")
            headers.append(line)
            try:
                class_label = int(line.split('|')[-1])
                classes.append(class_label)
            except (ValueError, IndexError):
                st.error(f"Error parsing class label from header: {line}")
                return None, None, None
        elif line:
            if not sequences:
                sequences.append("")
            sequences[-1] += line
    
    sequences = [seq for seq in sequences if seq]
    
    if len(sequences) != len(classes):
        st.error("Number of sequences doesn't match number of class labels")
        return None, None, None
    
    return sequences, classes, headers

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
    
    progress_bar = st.progress(0)
    for i, seq in enumerate(sequences):
        kmer_counts = count_kmers(seq, k)
        for j, kmer in enumerate(all_kmers):
            X[i, j] = kmer_counts[kmer]
        progress_bar.progress((i + 1) / len(sequences))
    
    return X, all_kmers

def plot_sequence_length_distribution(sequences):
    """Plot sequence length distribution."""
    lengths = [len(seq) for seq in sequences]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(lengths, bins=30, kde=True)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Frequency')
    return fig

def plot_gc_content_distribution(sequences):
    """Plot GC content distribution."""
    gc_contents = []
    for seq in sequences:
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        gc_contents.append(gc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(gc_contents, bins=30, kde=True)
    plt.title('Distribution of GC Content')
    plt.xlabel('GC Content')
    plt.ylabel('Frequency')
    return fig

def plot_nucleotide_composition(sequences):
    """Plot nucleotide composition for all sequences."""
    compositions = []
    for seq in sequences:
        total = len(seq)
        comp = {
            'A': seq.count('A') / total,
            'T': seq.count('T') / total,
            'G': seq.count('G') / total,
            'C': seq.count('C') / total
        }
        compositions.append(comp)
    
    df = pd.DataFrame(compositions)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(ax=ax)
    plt.title('Nucleotide Composition Distribution')
    plt.ylabel('Frequency')
    return fig

def plot_gc_content_by_class(sequences, classes):
    """Plot GC content by class."""
    gc_contents = []
    for seq in sequences:
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        gc_contents.append(gc)
    
    data = pd.DataFrame({
        'GC_Content': gc_contents,
        'Class': classes
    })
    
    class_names = {
        0: 'GPCRs',
        1: 'Tyrosine kinase',
        2: 'Protein phosphatases',
        3: 'PTPs',
        4: 'AARSs',
        5: 'Ion channels',
        6: 'Transcription Factor'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Class', y='GC_Content', data=data)
    plt.title('GC Content by Gene Class')
    plt.xlabel('Gene Class')
    plt.ylabel('GC Content')
    plt.xticks(range(len(class_names)), 
               [f"{i}: {class_names[i]}" for i in range(len(class_names))], 
               rotation=45, ha='right')
    plt.tight_layout()
    return fig

# About Section
st.sidebar.markdown("""
### About
This DNA Sequence Classifier application helps analyze and classify DNA sequences using various machine learning models. 
It provides detailed sequence analysis, feature extraction, and prediction capabilities.

### Features
- 🧬 Sequence Analysis
- 📊 Interactive Visualizations
- 🔍 K-mer Feature Extraction
- 🤖 ML Model Predictions
- 📈 Detailed Statistics
""")

# Sidebar Settings
st.sidebar.header("Settings")
k = st.sidebar.slider("K-mer length", min_value=2, max_value=5, value=3, 
                      help="""
                      K-mer length determines how we break down DNA sequences for analysis:
                      
                      - k=2: Looks at pairs of nucleotides (AA, AT, AG, AC, etc.)
                      - k=3: Looks at triplets (AAA, AAT, AAG, etc.)
                      - k=4: Looks at quadruplets (AAAA, AAAT, etc.)
                      - k=5: Looks at quintuplets (AAAAA, AAAAT, etc.)
                      
                      Higher k values:
                      - Capture more complex patterns
                      - Require more computational power
                      - Need more data to be effective
                      
                      Lower k values:
                      - Are simpler to compute
                      - Work well with smaller datasets
                      - May miss complex patterns
                      
                      For this dataset, k=3 is recommended as it provides a good balance between pattern complexity and computational efficiency.
                      """)

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["K-Nearest Neighbors (KNN)", "Random Forest"],
    help="KNN is better for small datasets, while Random Forest works well with larger datasets"
)

# Add detailed explanation for Random Forest
if model_type == "Random Forest":
    st.sidebar.markdown("""
        ### About Random Forest
        
        Random Forest is an advanced machine learning algorithm that works by creating multiple decision trees and combining their predictions.
        
        #### How it Works:
        1. Multiple Trees: Creates many decision trees (typically 100-500)
        2. Random Selection: Each tree uses a random subset of features
        3. Majority Vote: Final prediction is based on majority vote from all trees
        
        #### Advantages:
        - Handles large datasets efficiently
        - Works well with complex patterns in DNA sequences
        - Reduces overfitting through ensemble learning
        - Provides feature importance scores
        
        #### Best Used When:
        - Dataset has more than 100 sequences
        - Sequences have complex patterns
        - You need to understand feature importance
    """)

# Add this after the model selection
st.sidebar.info("""
### Dataset Size Notice
This application is designed to work with DNA sequence datasets. Please note:

1. Small datasets (less than 20 sequences):
   - K-Nearest Neighbors is recommended
   - Use k=3 for k-mer length
   - Expect lower accuracy due to limited data

2. Medium datasets (20-100 sequences):
   - Both KNN and Random Forest work well
   - Try k=3 or k=4 for k-mer length

3. Large datasets (100+ sequences):
   - Random Forest is recommended
   - Try k=4 or k=5 for k-mer length
""")

# File uploader with detailed instructions
st.markdown("""
### 📥 Upload DNA Sequences
Please upload a FASTA file containing DNA sequences. Each sequence should have a header with class information in the format:
```
>sequence_name|additional_info|class_number
ATGC...
```

**What is a FASTA file?**
A FASTA file is a text file that contains DNA sequences. Each sequence starts with a header line (beginning with '>') followed by the sequence itself. The header contains information about the sequence, including its class label.

**Example FASTA file:**
```
>sequence1|human|0
ATGCGATCGATCGATCG
>sequence2|mouse|1
CGATCGATCGATCGATC
```
""")

uploaded_file = st.file_uploader("Upload your FASTA file", type=['txt', 'fasta'])

if uploaded_file is not None:
    with st.spinner('Processing sequences...'):
        fasta_content = uploaded_file.read().decode()
        sequences, classes, headers = parse_fasta_with_class(fasta_content)
        
        if sequences and classes:
            st.success(f"✅ Successfully loaded {len(sequences)} sequences")
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Basic Analysis",
                "🧬 Detailed Analysis",
                "🔍 K-mer Features",
                "🤖 Prediction"
            ])
            
            with tab1:
                st.header("Basic Sequence Analysis")
                
                # Display basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Sequences", len(sequences))
                with col2:
                    st.metric("Number of Classes", len(set(classes)))
                with col3:
                    avg_len = sum(len(seq) for seq in sequences) / len(sequences)
                    st.metric("Average Sequence Length", f"{avg_len:.1f}")
                
                # Display sequence length distribution
                st.subheader("Sequence Length Distribution")
                st.markdown("""
                This plot shows how long the DNA sequences are. The x-axis shows the length of sequences, 
                and the y-axis shows how many sequences have that length. This helps us understand if our 
                sequences are similar in length or vary significantly.
                """)
                st.pyplot(plot_sequence_length_distribution(sequences))
            
            with tab2:
                st.header("Detailed Sequence Analysis")
                
                # Nucleotide composition
                st.subheader("Nucleotide Composition")
                st.markdown("""
                This plot shows the proportion of each nucleotide (A, T, G, C) in the sequences. 
                Each box represents the distribution of that nucleotide across all sequences. 
                This helps us understand if certain nucleotides are more common in our sequences.
                """)
                st.pyplot(plot_nucleotide_composition(sequences))
                
                # GC content analysis
                st.subheader("GC Content Analysis")
                st.markdown("""
                GC content is the percentage of G and C nucleotides in a sequence. This is important because:
                - GC-rich regions are more stable
                - Different organisms have different GC content
                - It can help identify different types of DNA sequences
                """)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_gc_content_distribution(sequences))
                with col2:
                    st.pyplot(plot_gc_content_by_class(sequences, classes))
                
                # Detailed sequence information
                st.subheader("Individual Sequence Details")
                st.markdown("""
                Here you can see detailed statistics for each sequence, including:
                - Length: How many nucleotides are in the sequence
                - GC Content: Percentage of G and C nucleotides
                - AT Content: Percentage of A and T nucleotides
                - Count of each nucleotide type
                """)
                for i, (seq, header) in enumerate(zip(sequences, headers)):
                    with st.expander(f"Sequence {i+1}: {header}"):
                        stats = calculate_sequence_stats(seq)
                        st.json(stats)
            
            with tab3:
                st.header("K-mer Feature Analysis")
                st.markdown("""
                K-mers are short DNA sequences of length k. For example, with k=3, we look at all possible 
                3-letter combinations (like 'ATG', 'CGA', etc.) in the DNA sequence. This helps us:
                - Identify patterns in the DNA sequences
                - Find common subsequences
                - Create features for machine learning
                """)
                
                with st.spinner(f'Generating {k}-mer features...'):
                    X, kmer_features = create_kmer_features(sequences, k)
                    feature_df = pd.DataFrame(X, columns=kmer_features)
                    
                    # Display feature matrix
                    st.subheader(f"{k}-mer Feature Matrix")
                    st.markdown("""
                    This table shows how many times each k-mer appears in each sequence. 
                    Each row is a sequence, and each column is a different k-mer. 
                    The numbers show how many times that k-mer appears in the sequence.
                    """)
                    st.dataframe(feature_df.style.highlight_max(axis=0))
                    
                    # Feature statistics
                    st.subheader("Feature Statistics")
                    st.markdown("""
                    These tables show which k-mers are most and least common across all sequences. 
                    This helps us understand which patterns are important in our DNA sequences.
                    """)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Most Common k-mers:")
                        mean_counts = feature_df.mean()
                        st.dataframe(mean_counts.nlargest(10))
                    
                    with col2:
                        st.write("Least Common k-mers:")
                        st.dataframe(mean_counts.nsmallest(10))
                    
                    # Download feature matrix
                    csv = feature_df.to_csv(index=False)
                    st.download_button(
                        "Download Feature Matrix",
                        csv,
                        "kmer_features.csv",
                        "text/csv",
                        key='download-csv'
                    )
            
            with tab4:
                st.header("Model Prediction")
                st.markdown("""
                This section uses machine learning to predict the class of each DNA sequence. 
                The model learns from the k-mer features we generated to make predictions.
                
                **Note about accuracy:**
                - With small datasets (less than 20 sequences), accuracy may be lower due to limited training data
                - The model needs enough examples of each class to learn effectively
                - Consider gathering more sequence data if possible
                """)
                
                # Define protein class mapping
                protein_classes = {
                    0: "GPCRs",
                    1: "Tyrosine kinase",
                    2: "Protein phosphatases",
                    3: "PTPs",
                    4: "AARSs",
                    5: "Ion channels",
                    6: "Transcription Factor"
                }
                
                if st.button("Run Prediction", help="Click to start prediction"):
                    with st.spinner('Running prediction...'):
                        try:
                            # Generate features
                            X, kmer_features = create_kmer_features(sequences, k)
                            
                            # Load and apply scaler
                            scaler_path = os.path.join('outputs', 'feature_scaler.pkl')
                            if os.path.exists(scaler_path):
                                with open(scaler_path, 'rb') as f:
                                    scaler = pickle.load(f)
                                X_scaled = scaler.transform(X)
                            else:
                                st.warning("No scaler found. Using unscaled features.")
                                X_scaled = X
                            
                            # Load and run appropriate model
                            model_loaded = False
                            model_path = os.path.join('outputs', 'knn_model.pkl') if model_type == "K-Nearest Neighbors" else os.path.join('outputs', 'rf_model.pkl')
                            
                            if model_path and os.path.exists(model_path):
                                try:
                                    model = joblib.load(model_path)
                                    model_loaded = True
                                except Exception as e:
                                    st.error(f"Error loading model: {str(e)}")
                            else:
                                st.error(f"Model file not found for {model_type}")
                                st.info("""
                                Please make sure you have trained models in the outputs directory.
                                You can train the models by running the pipeline:
                                1. Make sure you have the training data in the datasets directory
                                2. Run: `python run_pipeline.py`
                                """)
                            
                            if model_loaded:
                                # Make predictions
                                y_pred = model.predict(X_scaled)
                                
                                # Create results DataFrame
                                results_df = pd.DataFrame({
                                    'Sequence': headers,
                                    'True Class': classes,
                                    'Predicted Class': y_pred,
                                    'Correct': classes == y_pred
                                })
                                
                                # Display results
                                st.subheader("Prediction Results")
                                st.dataframe(results_df.style.apply(lambda x: ['background: #90EE90' if v else 'background: #FFB6C6' 
                                                                            for v in x == x], subset=['Correct']))
                                
                                # Calculate and display metrics
                                accuracy = (y_pred == classes).mean()
                                st.metric("Model Accuracy", f"{accuracy:.2%}")
                                
                                # Add detailed explanation for each predicted class
                                st.subheader("What Each Prediction Means")
                                
                                # Get protein class names from mapping
                                class_names = {v: k for k, v in protein_classes.items()}
                                
                                # Create a function to get protein descriptions
                                def get_protein_description(class_id):
                                    class_name = protein_classes.get(class_id, "Unknown")
                                    descriptions = {
                                        "GPCRs": "G Protein-Coupled Receptors - Cell membrane proteins important for communication between cells. Essential for vision, smell, and hormone responses.",
                                        "Tyrosine kinase": "Tyrosine Kinases - Enzymes that add phosphate groups to proteins. Critical for cell growth, division, and communication.",
                                        "Protein phosphatases": "Protein Phosphatases - Enzymes that remove phosphate groups from proteins. Help regulate many cell processes.",
                                        "PTPs": "Protein Tyrosine Phosphatases - Specialized enzymes that remove phosphates from specific amino acids. Important for immune system function.",
                                        "AARSs": "Aminoacyl-tRNA Synthetases - Essential enzymes that help build proteins by connecting amino acids to RNA molecules.",
                                        "Ion channels": "Ion Channels - Proteins that create pores in cell membranes, allowing ions to pass through. Critical for nerve signals and muscle contraction.",
                                        "Transcription Factor": "Transcription Factors - Proteins that control which genes are turned on or off. Help determine cell type and function."
                                    }
                                    return descriptions.get(class_name, "Unknown protein class")
                                
                                # Show a few example predictions with explanations
                                unique_predictions = results_df['Predicted Class'].unique()
                                for class_id in unique_predictions[:3]:  # Show explanations for up to 3 different predicted classes
                                    protein_name = protein_classes.get(class_id, f"Class {class_id}")
                                    description = get_protein_description(class_id)
                                    
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <h4>Prediction: {protein_name}</h4>
                                        <p>{description}</p>
                                        <p><strong>What this means for your research:</strong> Sequences identified as {protein_name} could be involved in 
                                        {protein_name.lower()}-related functions in the organism. This can guide further experimental work and analysis.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if accuracy < 0.5:
                                    st.warning("""
                                    **Low Accuracy Warning**
                                    The model's accuracy is below 50%. This could be due to:
                                    1. Very small dataset size
                                    2. Complex patterns that need more training data
                                    3. High variability in the sequences
                                    
                                    Suggestions:
                                    - Try using k=3 for k-mer length
                                    - Use KNN for small datasets
                                    - Consider gathering more training data
                                    """)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    "Download Prediction Results",
                                    csv,
                                    "prediction_results.csv",
                                    "text/csv",
                                    key='download-predictions'
                                )
                                
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.info("""
                            Make sure you have trained models in the outputs directory.
                            
                            Common issues:
                            1. Models not trained (run pipeline)
                            2. Wrong file paths (check '../outputs/' directory)
                            3. Dataset too small for selected model
                            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ by Group 5</p>
    <p><small>DNA Sequence Classification Project | 2024</small></p>
</div>
""", unsafe_allow_html=True) 