"""
Data processing utilities for FASTA file handling and sequence processing.
"""

import pandas as pd
import re
from Bio import SeqIO
from io import StringIO

def read_fasta_file(file_path):
    """Read FASTA file and return sequences with headers"""
    sequences = []
    headers = []
    
    try:
        with open(file_path, 'r') as file:
            for record in SeqIO.parse(file, 'fasta'):
                headers.append(record.id)
                sequences.append(str(record.seq))
        return sequences, headers
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return [], []

def read_fasta_from_upload(uploaded_file):
    """Read FASTA file from Streamlit upload"""
    sequences = []
    headers = []
    
    try:
        # Convert uploaded file to string
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # Parse FASTA
        for record in SeqIO.parse(stringio, 'fasta'):
            headers.append(record.id)
            sequences.append(str(record.seq))
        
        return sequences, headers
    except Exception as e:
        print(f"Error reading uploaded FASTA file: {e}")
        return [], []

def fasta_to_csv(input_file, output_file):
    """Convert FASTA file to CSV format"""
    sequences, headers = read_fasta_file(input_file)
    
    if not sequences:
        print("No sequences found in FASTA file")
        return False
    
    # Create DataFrame
    df = pd.DataFrame({
        'Header': headers,
        'Sequence': sequences
    })
    
    # Save to CSV
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully converted {len(sequences)} sequences to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return False

def validate_protein_sequence(sequence):
    """Validate if sequence contains only valid amino acid characters"""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.upper().strip()
    
    # Remove any whitespace or newlines
    sequence = re.sub(r'\s+', '', sequence)
    
    # Check if all characters are valid amino acids
    invalid_chars = set(sequence) - valid_aa
    
    if invalid_chars:
        return False, f"Invalid amino acid characters found: {invalid_chars}"
    
    if len(sequence) == 0:
        return False, "Empty sequence"
    
    return True, sequence

def clean_protein_sequence(sequence):
    """Clean protein sequence by removing invalid characters"""
    # Keep only valid amino acid characters
    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    return cleaned

def extract_label_from_header(header):
    """Extract label from FASTA header if present"""
    # Look for label patterns like |0| or |1| in header
    label_match = re.search(r'\|([01])\|', header)
    if label_match:
        return int(label_match.group(1))
    
    # Look for label at the end of header
    if header.endswith('_0') or header.endswith('|0'):
        return 0
    elif header.endswith('_1') or header.endswith('|1'):
        return 1
    
    return None

def create_balanced_sample(input_file, output_file, n_samples=100):
    """Create balanced sample from FASTA file"""
    sequences, headers = read_fasta_file(input_file)
    
    if not sequences:
        print("No sequences found in input file")
        return False
    
    # Extract labels and organize sequences
    labeled_sequences = {'0': [], '1': []}
    
    for header, sequence in zip(headers, sequences):
        label = extract_label_from_header(header)
        if label is not None:
            labeled_sequences[str(label)].append((header, sequence))
    
    # Check if we have both classes
    if not labeled_sequences['0'] or not labeled_sequences['1']:
        print("Error: Need sequences with both labels (0 and 1)")
        return False
    
    # Sample equal amounts from each class
    samples_per_class = n_samples // 2
    
    import random
    random.seed(42)  # For reproducibility
    
    sampled_sequences = []
    
    for label in ['0', '1']:
        available = labeled_sequences[label]
        if len(available) < samples_per_class:
            print(f"Warning: Only {len(available)} sequences available for label {label}, using all")
            sampled = available
        else:
            sampled = random.sample(available, samples_per_class)
        
        sampled_sequences.extend(sampled)
    
    # Write to output file
    try:
        with open(output_file, 'w') as f:
            for header, sequence in sampled_sequences:
                f.write(f">{header}\n{sequence}\n")
        
        print(f"Created balanced sample with {len(sampled_sequences)} sequences")
        return True
    except Exception as e:
        print(f"Error writing sample file: {e}")
        return False

def get_sequence_stats(sequences):
    """Get basic statistics about sequences"""
    if not sequences:
        return {}
    
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        'total_sequences': len(sequences),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_length': sum(lengths) / len(lengths),
        'total_residues': sum(lengths)
    }
    
    return stats
