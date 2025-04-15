#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DNA Sequencing Analysis Pipeline

This script runs the full DNA sequencing analysis pipeline:
1. Exploratory Data Analysis
2. K-mer Feature Generation
3. KNN Classification
4. Random Forest Classification
5. Neural Network Classification
"""

import os
import subprocess
import time

def run_script(script_path, description):
    """Run a Python script and display its output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.run(['python', script_path], capture_output=False, text=True)
    
    if process.returncode != 0:
        print(f"\nError running {script_path}!")
        return False
    
    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
    return True

def main():
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # List of scripts to run in order
    scripts = [
        ('notebooks/1__Exploratory_Data_Analysis_EDA.py', 'Exploratory Data Analysis'),
        ('notebooks/2__Generate_kmer_dataset.py', 'K-mer Feature Generation'),
        ('notebooks/3__KNN_Classification.py', 'KNN Classification'),
        ('notebooks/4__RandomForest.py', 'Random Forest Classification'),
        ('notebooks/5__FeedForward_Neural_Networks.py', 'Neural Network Classification')
    ]
    
    # Run each script
    for script_path, description in scripts:
        success = run_script(script_path, description)
        if not success:
            print(f"Pipeline stopped at: {description}")
            break
    
    print("\nDNA Sequencing Analysis Pipeline completed.")

if __name__ == "__main__":
    main() 