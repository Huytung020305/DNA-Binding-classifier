"""
Script to create balanced samples from FASTA files.
Usage: python create_sample.py input.fasta output.fasta --samples 100
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_utils import create_balanced_sample

def main():
    parser = argparse.ArgumentParser(description='Create balanced sample from FASTA file')
    parser.add_argument('input_file', help='Input FASTA file path')
    parser.add_argument('output_file', help='Output FASTA file path')
    parser.add_argument('--samples', type=int, default=100, help='Total number of samples (default: 100)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    if args.samples <= 0 or args.samples % 2 != 0:
        print("Error: Number of samples must be a positive even number")
        return 1
    
    success = create_balanced_sample(args.input_file, args.output_file, args.samples)
    
    if success:
        print("Sample creation completed successfully!")
        return 0
    else:
        print("Sample creation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
