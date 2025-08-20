"""
Script to convert FASTA files to CSV format.
Usage: python fasta_to_csv.py input.fasta output.csv
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_utils import fasta_to_csv

def main():
    parser = argparse.ArgumentParser(description='Convert FASTA file to CSV format')
    parser.add_argument('input_file', help='Input FASTA file path')
    parser.add_argument('output_file', help='Output CSV file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    success = fasta_to_csv(args.input_file, args.output_file)
    
    if success:
        print("Conversion completed successfully!")
        return 0
    else:
        print("Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
