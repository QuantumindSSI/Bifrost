#!/usr/bin/env python3
"""
Download and prepare text corpus for PhaseLLM training.

Downloads public domain books from Project Gutenberg for language modeling training.

Usage:
    python scripts/download_text_corpus.py --output-path train_data/text_corpus.txt --num-books 10
"""

import argparse
import re
from pathlib import Path
from typing import List

import requests


def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg header/footer and clean text.
    
    Args:
        text: Raw text from Project Gutenberg
        
    Returns:
        Cleaned text
    """
    # Remove Project Gutenberg header
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    
    # Find start
    start_idx = len(text)
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1 and idx < start_idx:
            start_idx = idx + len(marker)
    
    # Find end
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    
    # Extract content
    content = text[start_idx:end_idx]
    
    # Clean up
    content = re.sub(r'\r\n', '\n', content)  # Normalize line endings
    content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive blank lines
    content = content.strip()
    
    return content


def download_gutenberg_book(book_id: int) -> str:
    """
    Download a book from Project Gutenberg.
    
    Args:
        book_id: Project Gutenberg book ID
        
    Returns:
        Book text
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to download book {book_id}: {e}")
        return ""


def main():
    """Main download entry point."""
    parser = argparse.ArgumentParser(
        description="Download Text Corpus from Project Gutenberg"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="train_data/text_corpus.txt",
        help="Output text file path",
    )
    parser.add_argument(
        "--num-books",
        type=int,
        default=10,
        help="Number of books to download",
    )
    parser.add_argument(
        "--book-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific Project Gutenberg book IDs to download",
    )
    
    args = parser.parse_args()
    
    # Classic public domain books (good for language modeling)
    default_book_ids = [
        11,    # Alice's Adventures in Wonderland
        74,    # The Adventures of Tom Sawyer
        84,    # Frankenstein
        1342,  # Pride and Prejudice
        1661,  # The Adventures of Sherlock Holmes
        2600,  # War and Peace
        2701,  # Moby Dick
        345,   # Dracula
        98,    # A Tale of Two Cities
        844,   # The Divine Comedy
        43,    # The Strange Case of Dr. Jekyll and Mr. Hyde
        174,   # The Adventures of Sherlock Holmes
        46,    # A Christmas Carol
        1232,  # The Prince
        1952,  # The Yellow Wallpaper
    ]
    
    book_ids = args.book_ids if args.book_ids else default_book_ids[:args.num_books]
    
    print("=" * 60)
    print("Downloading Text Corpus from Project Gutenberg")
    print("=" * 60)
    print(f"Output path: {args.output_path}")
    print(f"Books to download: {len(book_ids)}")
    print()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download books
    all_texts = []
    for i, book_id in enumerate(book_ids, 1):
        print(f"[{i}/{len(book_ids)}] Downloading book {book_id}...")
        text = download_gutenberg_book(book_id)
        
        if text:
            cleaned = clean_gutenberg_text(text)
            if len(cleaned) > 1000:  # Only keep substantial texts
                all_texts.append(cleaned)
                print(f"  ✓ Downloaded {len(cleaned)} characters")
            else:
                print(f"  ✗ Text too short after cleaning")
        else:
            print(f"  ✗ Failed to download")
    
    # Combine texts
    print()
    print(f"Successfully downloaded {len(all_texts)} books")
    print(f"Total characters: {sum(len(t) for t in all_texts):,}")
    
    # Write to file (one paragraph per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in all_texts:
            # Split into paragraphs
            paragraphs = re.split(r'\n\n+', text)
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50:  # Only keep substantial paragraphs
                    f.write(para + '\n')
    
    print(f"Saved to: {args.output_path}")
    print()
    print("Training data ready!")


if __name__ == "__main__":
    main()
