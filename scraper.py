#!/usr/bin/env python3
"""
Scrapes, cleans, and prepares text from a URL or local file for LLM training.

This script combines the functionality of a web scraper and a text cleaner
into a single pipeline. It takes a URL or a local HTML file, extracts the
main content using Trafilatura, and then applies a series of state-of-the-art
cleaning steps to produce a clean text file ready for tokenization.

The cleaning pipeline includes:
1.  Encoding error correction (`ftfy`).
2.  Unicode normalization.
3.  Intelligent whitespace normalization.
4.  Selective character cleaning with a whitelist.

The output file is automatically named based on the input, with '_scraped.txt'
appended to the base name.

Usage:
  python scraper.py https://example.com/some/article.html
  python scraper.py /path/to/local_document.html
  python scraper.py https://example.com --verbose
"""
import argparse
import os
import re
import unicodedata
import sys
import subprocess
import importlib
from urllib.parse import urlparse


def _install_and_import(package, import_name=None):
    """
    Tries to import a package, and if it fails, attempts to install it via pip
    and then import it again.
    """
    if import_name is None:
        import_name = package
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"Warning: '{package}' not found. Attempting to install it with pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed '{package}'.")
            return importlib.import_module(import_name)
        except (subprocess.CalledProcessError, ImportError) as e:
            print(f"Error: Failed to install '{package}'. Please install it manually: 'pip install {package}'. Details: {e}")
            sys.exit(1)

# Ensure required packages are installed
requests = _install_and_import("requests")
trafilatura = _install_and_import("trafilatura")
ftfy = _install_and_import("ftfy")

def is_url(s: str) -> bool:
    """Checks if a string is a valid HTTP/HTTPS URL."""
    p = urlparse(s)
    return p.scheme in ("http", "https")


def load_html_from_url(url: str, timeout: int = 10) -> str:
    """Loads HTML content from a URL."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def load_html_from_file(path: str) -> str:
    """Loads HTML content from a local file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_main_text_trafilatura(html: str, url: str = None) -> str:
    """Extracts meaningful main text using Trafilatura."""
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    return downloaded if downloaded else ""


def clean_and_prepare_text(raw_text: str) -> str:
    """
    Applies a series of cleaning and normalization steps to raw text.

    Args:
        raw_text: The input string to clean.

    Returns:
        A cleaned and normalized string, ready for an LLM.
    """
    # 1. Fix encoding errors and mojibake
    text = ftfy.fix_text(raw_text)

    # 2. Normalize to NFC Unicode form
    text = unicodedata.normalize('NFC', text)

    # 3. Intelligent whitespace normalization
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # 4. Selective character cleaning using a whitelist
    allowed_chars = (
        r"A-Za-z0-9\s"  # Alphanumeric and whitespace
        r".,!?'\"()\[\]{}:;-"  # Common punctuation
    )
    text = re.sub(f'[^{allowed_chars}]', '', text)

    # 5. Final whitespace cleanup
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    return text


def generate_output_filename(input_path: str) -> str:
    """Generates an output filename based on the input URL or file path."""
    if is_url(input_path):
        parsed_url = urlparse(input_path)
        # Use the path component, or the netloc if path is empty/root
        path_part = parsed_url.path.strip('/')
        base_name = os.path.basename(path_part) if path_part else parsed_url.netloc
    else:
        base_name = os.path.basename(input_path)

    # Remove extension and append the new suffix
    filename, _ = os.path.splitext(base_name)
    return f"{filename}_scraped.txt"


def main():
    """Main function to parse arguments and run the scrape-and-clean pipeline."""
    parser = argparse.ArgumentParser(
        description="Scrape, clean, and prepare text from a URL or local file for LLM training."
    )
    parser.add_argument("input", help="URL (http/https) or path to a local HTML file.")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP request timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    args = parser.parse_args()

    src = args.input
    output_filename = generate_output_filename(src)

    try:
        # Step 1: Load HTML
        if is_url(src):
            if args.verbose:
                print(f"Loading URL: {src}")
            html = load_html_from_url(src, timeout=args.timeout)
        elif os.path.exists(src) and os.path.isfile(src):
            if args.verbose:
                print(f"Loading local HTML file: {src}")
            html = load_html_from_file(src)
        else:
            raise ValueError("Input must be a valid URL or an existing local file path.")

        # Step 2: Extract raw text
        if args.verbose:
            print("Extracting main content with Trafilatura...")
        raw_text = extract_main_text_trafilatura(html, url=src if is_url(src) else None)

        if not raw_text.strip():
            print("Warning: No meaningful text was extracted from the source.")
            return

        # Step 3: Clean and prepare text
        if args.verbose:
            print("Applying text cleaning and normalization pipeline...")
        cleaned_text = clean_and_prepare_text(raw_text)

        # Step 4: Save to file
        if args.verbose:
            print(f"Saving cleaned text to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f"Successfully processed '{src}' and saved to '{output_filename}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL '{src}': {e}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{src}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()