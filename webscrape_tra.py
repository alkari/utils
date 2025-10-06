#!/usr/bin/env python3
"""
Extract meaningful main text from a URL or local HTML file using Trafilatura,
then append or overwrite the text in a specified output file.

Usage examples:
  python extract_append_trafilatura.py https://example.com -o output.txt
  python extract_append_trafilatura.py localpage.html --output out.txt --overwrite
  python extract_append_trafilatura.py https://example.com -o out.txt --verbose
"""
import argparse
import os
from urllib.parse import urlparse
import requests
import trafilatura

def is_url(s: str) -> bool:
    p = urlparse(s)
    return p.scheme in ("http", "https")

def load_html_from_url(url: str, timeout: int = 10) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def load_html_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_main_text_trafilatura(html: str, url: str = None) -> str:
    # Use trafilatura to extract meaningful main text
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    # Trafilatura returns None if extraction fails, convert to empty string
    return downloaded if downloaded else ""

def append_or_write_text(text: str, out_path: str, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "a"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Extract meaningful main text from URL or local HTML file using Trafilatura and write to a text file.")
    parser.add_argument("input", help="URL (http/https) or path to local HTML file")
    parser.add_argument("-o", "--output", required=True, help="Output text filename")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of appending")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    args = parser.parse_args()

    src = args.input
    out = args.output

    try:
        if is_url(src):
            if args.verbose:
                print(f"Loading URL: {src}")
            html = load_html_from_url(src, timeout=args.timeout)
        elif os.path.exists(src) and os.path.isfile(src):
            if args.verbose:
                print(f"Loading local HTML file: {src}")
            html = load_html_from_file(src)
        else:
            raise ValueError("Input must be a valid http/https URL or an existing local HTML file path.")

        if args.verbose:
            print("Extracting main meaningful text with Trafilatura...")
        text = extract_main_text_trafilatura(html, url=src if is_url(src) else None)
        if not text.strip():
            if args.verbose:
                print("No meaningful text extracted.")
        else:
            if args.verbose:
                mode_descr = "overwrite" if args.overwrite else "append"
                print(f"Writing extracted text to {out} ({mode_descr})")
            append_or_write_text(text.strip(), out, overwrite=args.overwrite)
            if args.verbose:
                print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

