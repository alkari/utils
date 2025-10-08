#!/usr/bin/env python3
"""
Scrapes, cleans, and prepares text from a URL or local file for LLM training.

This script combines a web scraper and a text cleaner into a single, powerful
pipeline. It can process a URL, a local HTML file, or a plain text file.
It extracts the main content, then applies a series of state-of-the-art
cleaning steps to produce a clean text file ready for tokenization and LLM
fine-tuning.

Core Features:
- HTML/Web Scraping: Uses Trafilatura for robust main content extraction.
- Plain Text Processing: Can skip HTML parsing to clean raw text files.
- Basic Cleaning: Fixes encoding errors, normalizes Unicode, and handles whitespace.
- Advanced NLP Cleaning: Uses spaCy for PII anonymization (e.g., replacing names
  with [PERSON]) and leverages multiprocessing for speed. Can use a GPU if available.
- Perplexity Filtering: Uses a small transformer model (distilgpt2) to score
  text quality and remove nonsensical "gibberish" paragraphs.

Usage:
  # Scrape and clean a web page with advanced features
  python scraper.py https://en.wikipedia.org/wiki/LLM --advanced --verbose

  # Clean a local text file, skipping HTML parsing and filtering gibberish
  python scraper.py my_document.txt --skip-html --advanced --filter-perplexity

Command-Line Options:
  input                     Required. URL, local HTML file, or local text file.
  --timeout                 HTTP request timeout in seconds (default: 10).
  --advanced                Enable advanced cleaning with spaCy (PII removal).
  --processes               Number of CPU cores for advanced cleaning (default: all but one).
  --skip-html               Treat input as a plain text file, skipping HTML parsing.
  --filter-perplexity       Enable perplexity-based filtering to remove gibberish.
  --perplexity-threshold    Threshold for perplexity filter (default: 300.0). Lower is stricter.
  --verbose                 Print detailed progress messages.

The output file is automatically named based on the input, with '_scraped.txt'
appended.
"""
import argparse
import os
import re
import unicodedata
import sys
import multiprocessing
import os
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

# Global variable to hold spaCy model in worker processes
_nlp_model_instance = None

def _init_worker_nlp(model_name):
    """Initializer function for multiprocessing pool workers to load spaCy model."""
    global _nlp_model_instance
    if _nlp_model_instance is None:
        _nlp_model_instance = spaCy.load(model_name)

# Ensure required packages are installed
requests = _install_and_import("requests")
trafilatura = _install_and_import("trafilatura")
ftfy = _install_and_import("ftfy")
spaCy = _install_and_import("spacy")
tqdm = _install_and_import("tqdm")
torch = _install_and_import("torch")
transformers = _install_and_import("transformers")

# Unpack tqdm for direct use
from tqdm import tqdm



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


def _process_single_chunk_with_spacy(chunk_text: str) -> str:
    """
    Worker function for multiprocessing pool to process a single text chunk with spaCy.
    Loads spaCy model if not already loaded in the process.
    """
    global _nlp_model_instance
    if _nlp_model_instance is None:
        # Fallback if initializer didn't run (e.g., in main process for sequential processing)
        _nlp_model_instance = spaCy.load("en_core_web_sm")

    doc = _nlp_model_instance(chunk_text)
    anonymized_tokens = []
    for token in doc:
        if token.ent_type_ == "PERSON" and token.ent_iob_ == "B":
            anonymized_tokens.append("[PERSON]")
            if token.whitespace_:
                anonymized_tokens.append(" ")
        else:
            anonymized_tokens.append(token.text_with_ws)
    return "".join(anonymized_tokens).strip()


def filter_text_by_perplexity(text: str, model, tokenizer, threshold: float) -> str:
    """
    Filters out low-quality paragraphs from text using perplexity scoring.

    Args:
        text: The input text, with paragraphs separated by double newlines.
        model: A pre-trained causal language model (e.g., from transformers).
        tokenizer: The tokenizer for the model.
        threshold: The perplexity score above which a paragraph is discarded.

    Returns:
        The cleaned text with low-quality paragraphs removed.
    """
    # Use the model's configured max length, defaulting to 1024
    max_length = tokenizer.model_max_length
    good_paragraphs = []
    # Process the text paragraph by paragraph
    for paragraph in tqdm(text.split('\n\n'), desc="Filtering by perplexity"):
        paragraph = paragraph.strip()
        if not paragraph or len(paragraph.split()) < 10:  # Skip empty or very short paragraphs
            continue

        try:
            # Tokenize the entire paragraph to get its total length
            token_ids = tokenizer.encode(paragraph, add_special_tokens=False) # Don't add special tokens like <bos>
            total_length = len(token_ids)
            
            perplexities = []
            
            # Process the paragraph in chunks that fit the model's max length
            for i in range(0, total_length, max_length):
                chunk_ids = token_ids[i : i + max_length]
                if not chunk_ids:
                    continue
                
                # Create tensors for the model
                input_ids = torch.tensor([chunk_ids])
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                
                # Calculate perplexity for the chunk and store it
                perplexities.append(torch.exp(outputs.loss).item())

            # If we have valid perplexity scores, calculate the average for the paragraph
            if perplexities:
                avg_perplexity = sum(perplexities) / len(perplexities)
                if avg_perplexity < threshold:
                    good_paragraphs.append(paragraph)
            else:
                # If the paragraph was too short to produce any valid chunks (unlikely with the length check above),
                # we can choose to either keep or discard it. Keeping it is a safe default.
                good_paragraphs.append(paragraph)

        except Exception as e:
            # If tokenization or model fails for any reason, discard the paragraph
            # This can happen with malformed text that the tokenizer can't handle.
            print(f"\nWarning: Could not process a paragraph due to an error: {e}. Skipping it.")
            continue

    return "\n\n".join(good_paragraphs)

def clean_and_prepare_text_advanced(raw_text: str, nlp_model, use_multiprocessing: bool, perplexity_args: dict = None) -> str: # type: ignore
    """
    Applies an advanced cleaning pipeline using spaCy for semantic processing.

    This pipeline includes:
    1. Basic cleaning (encoding, unicode normalization).
    2. Recursive chunking to handle large documents without memory errors.
    3. PII Anonymization (e.g., replacing person names) via multiprocessing.

    Args:
        raw_text: The input string to clean.
        nlp_model: A loaded spaCy NLP model.
        use_multiprocessing: Flag to determine if parallel processing should be used.
        perplexity_args (dict, optional): Arguments for perplexity filtering.

    Returns:
        A cleaned, anonymized, and well-structured string.
    """
    # 1. Perform initial low-level cleaning
    text = ftfy.fix_text(raw_text)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Split text into manageable chunks for spaCy processing.
    max_len = nlp_model.max_length
    all_chunks_to_process = []

    def _collect_chunks_recursively(text_to_process):
        """Recursively splits large text into spaCy-compatible chunks and collects them."""
        if not text_to_process.strip():
            return

        if len(text_to_process) <= max_len:
            all_chunks_to_process.append(text_to_process)
        else:
            # Chunk is too large, split it and process sub-chunks recursively.
            # We prefer to split at paragraph boundaries for semantic coherence.
            split_pos = text_to_process.rfind('\n\n', 0, max_len)
            if split_pos == -1:
                # If no double newline, find the last single newline.
                split_pos = text_to_process.rfind('\n', 0, max_len)
            if split_pos == -1:
                # If no newlines at all, split at the last space.
                split_pos = text_to_process.rfind(' ', 0, max_len)
            if split_pos <= 0: # Ensure split_pos is valid and not at the very beginning
                # As a last resort, hard-split at max_len.
                split_pos = max_len

            _collect_chunks_recursively(text_to_process[:split_pos])
            _collect_chunks_recursively(text_to_process[split_pos:])

    # Start collecting chunks from the entire text
    _collect_chunks_recursively(text)

    # 3. Process chunks in parallel (CPU) or sequentially (GPU/single-core)
    if use_multiprocessing and len(all_chunks_to_process) > 1:
        num_processes = nlp_model._num_processes # Access the custom attribute set in main
        print(f"Processing on {num_processes} CPU cores...")
        with multiprocessing.Pool(processes=num_processes, initializer=_init_worker_nlp, initargs=("en_core_web_sm",)) as pool:
            processed_chunks = list(tqdm(pool.imap(_process_single_chunk_with_spacy, all_chunks_to_process),
                                         total=len(all_chunks_to_process), desc="Processing chunks (parallel)"))
    else:
        processed_chunks = []
        for chunk in tqdm(all_chunks_to_process, desc="Processing chunks (sequential)"):
            processed_chunks.append(_process_single_chunk_with_spacy(chunk))

    # 4. Recombine the spaCy-processed chunks and apply perplexity filtering if enabled
    recombined_text = "\n\n".join(processed_chunks)
    if perplexity_args and perplexity_args.get("model"):
        print(f"Applying perplexity filter with threshold {perplexity_args['threshold']}...")
        return filter_text_by_perplexity(recombined_text, **perplexity_args)
    
    return recombined_text


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
    parser.add_argument("--advanced", action="store_true", help="Use advanced NLP cleaning with spaCy (slower).")
    parser.add_argument("--processes", type=int, default=max(1, os.cpu_count() - 1),
                        help="Number of processor cores to use for advanced cleaning. Default: CPU_COUNT - 1 (min 1).")
    parser.add_argument("--filter-perplexity", action="store_true", help="Enable perplexity-based filtering to remove gibberish text.")
    parser.add_argument("--skip-html", action="store_true", help="Process the input as a plain text file, skipping HTML parsing.")
    parser.add_argument("--perplexity-threshold", type=float, default=300.0, help="Perplexity score threshold for filtering. Lower is stricter. Default: 300.")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    args = parser.parse_args()

    src = args.input
    output_filename = generate_output_filename(src)

    nlp = None
    perplexity_model_args = {}
    is_gpu_active = False
    if args.advanced:
        if args.verbose:
            print("Advanced mode enabled. Checking for GPU...")
        
        # Try to activate GPU. prefer_gpu() returns True if successful.
        is_gpu_active = spaCy.prefer_gpu()
        if is_gpu_active:
            print("GPU detected! spaCy will use the GPU for processing.")
        else:
            print("No compatible GPU detected. spaCy will use the CPU.")

        try:
            if args.verbose: print("Loading spaCy model...")
            nlp = spaCy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Downloading...")
            try:
                spaCy.cli.download("en_core_web_sm")
                nlp = spaCy.load("en_core_web_sm")
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error: Failed to download spaCy model. Please run 'python -m spacy download en_core_web_sm'. Details: {e}")
                sys.exit(1)
        if args.verbose:
            # Store num_processes directly on the nlp object for easy access in the cleaning function
            # This is a bit of a hack but avoids changing the signature of clean_and_prepare_text_advanced
            nlp._num_processes = args.processes
            print("spaCy model loaded.")
        
        if args.filter_perplexity:
            if args.verbose:
                print("Perplexity filter enabled. Loading language model for scoring...")
            model_name = "distilgpt2"
            perplexity_model_args["tokenizer"] = transformers.AutoTokenizer.from_pretrained(model_name)
            perplexity_model_args["model"] = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            perplexity_model_args["threshold"] = args.perplexity_threshold
            if args.verbose: print(f"Language model '{model_name}' loaded.")


    try:
        # Initialize raw_text to be populated by one of the loading methods
        raw_text = ""
        # --- Stage 1: Load Raw Text ---
        if args.skip_html:
            # If --skip-html is used, treat the input as a plain text file.
            if args.verbose:
                print(f"Loading plain text file (skipping HTML parsing): {src}")
            if not (os.path.exists(src) and os.path.isfile(src)):
                 raise FileNotFoundError(f"Input file not found at '{src}'")
            with open(src, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            # Default behavior: treat input as a URL or local HTML file.
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

        # Check if any text was actually extracted before proceeding
        if not raw_text.strip():
            print("Warning: No meaningful text was extracted from the source.")
            return

        # --- Stage 2: Clean and Prepare Text ---
        if args.verbose:
            if args.advanced:
                print("Applying ADVANCED text cleaning and normalization pipeline...")
                # Disable multiprocessing if a GPU is active
                use_multiprocessing = not is_gpu_active and args.processes > 1
                cleaned_text = clean_and_prepare_text_advanced(raw_text, nlp, use_multiprocessing, perplexity_args=perplexity_model_args) # type: ignore
            else:
                print("Applying standard text cleaning and normalization pipeline...")
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