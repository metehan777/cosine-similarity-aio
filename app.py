import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
import tkinter as tk
from tkinter import filedialog

def calculate_cosine_similarity(query, text):
    """
    Calculate the cosine similarity between a query and a text using sentence transformers.
    
    Args:
        query (str): The target query string
        text (str): The full text to compare against
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    try:
        # Load multilingual model that supports Turkish
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Create embeddings for both query and text
        query_embedding = model.encode([query])
        text_embedding = model.encode([text])
        
        # Calculate cosine similarity between the query and text embeddings
        similarity = cosine_similarity(query_embedding, text_embedding)[0][0]
        return similarity
    except Exception as e:
        print(f"Error during embedding calculation: {e}")
        return 0.0

def select_text_file():
    """
    Open a file dialog to select a text file
    
    Returns:
        str: The content of the selected file or empty string if canceled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Text File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
    
    return ""

def main():
    parser = argparse.ArgumentParser(description='Calculate cosine similarity between a query and text.')
    parser.add_argument('--query', type=str, help='The target query string')
    parser.add_argument('--text_file', type=str, help='Path to a file containing the text')
    parser.add_argument('--text', type=str, help='The text to compare against (alternative to --text_file)')
    parser.add_argument('--select_file', action='store_true', help='Open file dialog to select a text file')
    
    args = parser.parse_args()
    
    # Get query from arguments or prompt user
    query = args.query
    if not query:
        query = input("Enter the target query: ")
    
    # Get text from file, argument, file dialog, or prompt user
    text = ""
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Error: File '{args.text_file}' not found.")
            sys.exit(1)
    elif args.text:
        text = args.text
    elif args.select_file or not (args.text_file or args.text):
        print("Opening file dialog to select a text file...")
        text = select_text_file()
        if not text:
            print("No file selected or file is empty.")
            
            # Ask if user wants to enter text manually instead
            manual_input = input("Do you want to enter text manually instead? (y/n): ")
            if manual_input.lower() == 'y':
                print("Please paste text below (Ctrl+D or Ctrl+Z to finish):")
                text_lines = []
                try:
                    for line in sys.stdin:
                        text_lines.append(line)
                except KeyboardInterrupt:
                    pass
                text = ''.join(text_lines)
            else:
                sys.exit(0)
    
    if not text.strip():
        print("Error: No text provided.")
        sys.exit(1)
    
    print("Calculating similarity with sentence transformers (this might take a moment)...")
    similarity = calculate_cosine_similarity(query, text)
    print(f"\nCosine Similarity between query and text: {similarity:.4f}")
    
    # Interpret the result
    if similarity > 0.8:
        print("Interpretation: Very high similarity")
    elif similarity > 0.6:
        print("Interpretation: High similarity")
    elif similarity > 0.4:
        print("Interpretation: Moderate similarity")
    elif similarity > 0.2:
        print("Interpretation: Low similarity")
    else:
        print("Interpretation: Very low similarity")

if __name__ == "__main__":
    main()
