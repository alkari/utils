#!/usr/bin/env python3
"""
Extract visible text from a URL or local HTML file and append (or write) to a text file.

Usage examples:
  python extract_append.py https://example.com -o output.txt
  python extract_append.py localpage.html --output out.txt --overwrite
  python extract_append.py https://example.com -o out.txt --verbose
"""
import argparse
import os
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

def is_url(s: str) -> bool:
    # simple check using urlparse and scheme
    p = urlparse(s)
    return p.scheme in ("http", "https")

def load_html_from_url(url: str, timeout: int = 10) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def load_html_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style/meta and hidden elements
    for tag in soup(["script", "style", "noscript", "iframe", "meta", "header", "footer", "form"]):
        tag.decompose()

    # Collect text from common content tags in a reasonable order
    parts = []
    for selector in ["h1","h2","h3","h4","h5","h6","p","li","article","section","div"]:
        for el in soup.find_all(selector):
            text = el.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)

    # Fallback: full-body text if nothing collected
    if not parts:
        body = soup.body
        if body:
            fallback = body.get_text(separator=" ", strip=True)
            if fallback:
                parts.append(fallback)

    # Join with double newlines between blocks for readability
    return "\n\n".join(parts).strip()

def append_or_write_text(text: str, out_path: str, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "a"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Extract text from a URL or local HTML file and append to a text file.")
    parser.add_argument("input", help="URL (http/https) or path to local HTML file")
    parser.add_argument("-o", "--output", required=True, help="Output text filename (will be created if missing)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file instead of appending")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
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
                print(f"Loading local file: {src}")
            html = load_html_from_file(src)
        else:
            raise ValueError("Input must be a valid http/https URL or an existing local HTML file path.")

        if args.verbose:
            print("Extracting text...")
        text = extract_visible_text(html)
        if not text:
            if args.verbose:
                print("No text extracted (page may be empty or heavily scripted).")
        else:
            if args.verbose:
                mode = "overwrite" if args.overwrite else "append"
                print(f"Writing extracted text to {out} ({mode})")
            append_or_write_text(text, out, overwrite=args.overwrite)
            if args.verbose:
                print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

