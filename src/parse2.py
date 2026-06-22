#!/usr/bin/env python3
"""
PDF parser with hybrid approach: text extraction for text PDFs, Vision API for scans.
Optimized with parallel processing and page batching for maximum speed.
"""

import argparse
import base64
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.environ.setdefault("HF_HOME", os.path.join(config.BASE_DIR, ".cache", "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

client = OpenAI(
    api_key="dummy", base_url=config.API_BASE_URL, max_retries=1, timeout=120.0
)

SYSTEM_PROMPT = (
    "You are a document cleaner and OCR expert. Extract and clean text from the provided document. "
    "Remove headers, footers, page numbers, copyrights, TOC, ads, and OCR artifacts. "
    "Preserve main text, headings, lists, tables, and formulas. "
    "Return ONLY cleaned Markdown."
)


def clear_all_caches():
    """Remove all model caches to free disk space."""
    cache_dirs = [
        os.environ.get("HF_HOME"),
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "marker",
    ]
    total_freed = 0
    for cache_dir in cache_dirs:
        if cache_dir and Path(cache_dir).exists():
            size = sum(
                f.stat().st_size for f in Path(cache_dir).rglob("*") if f.is_file()
            )
            shutil.rmtree(cache_dir, ignore_errors=True)
            total_freed += size
            logging.info(f"Cleared: {cache_dir} ({size / 1e9:.2f} GB)")
    logging.info(f"Total freed: {total_freed / 1e9:.2f} GB")


def has_text_layer(pdf_path: Path, sample_pages: int = 3) -> bool:
    """Check if PDF has extractable text layer (not a scan)."""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages[:sample_pages]:
            if page.extract_text().strip():
                return True
        return False
    except Exception:
        return False


def extract_text_fast(pdf_path: Path) -> str:
    """Fast text extraction from PDF with text layer."""
    reader = PdfReader(pdf_path)
    return "\n".join(p.extract_text() for p in reader.pages)


def has_substantial_text(text: str) -> bool:
    """Check if extracted text contains meaningful content."""
    if not text:
        return False
    meaningful = len(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]", text))
    return meaningful > 80


def pdf_to_base64(pdf_path: Path) -> str:
    """Convert PDF file to base64 string."""
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_with_llm_direct_pdf(pdf_b64: str, model: str) -> str:
    """Send PDF directly to LLM (if API supports it)."""
    content = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:application/pdf;base64,{pdf_b64}",
                "detail": "high",
            },
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=8192,
        reasoning_effort="none",
    )
    return resp.choices[0].message.content.strip()


def clean_with_llm_text(text: str, model: str, chunk_size: int = 4000) -> str:
    """Clean text chunks via LLM API (for text PDFs)."""
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    cleaned = []

    for i, chunk in enumerate(tqdm(chunks, desc="  LLM Text", leave=False)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.0,
                max_tokens=8192,
                reasoning_effort="none",
            )
            cleaned.append(resp.choices[0].message.content.strip())
        except Exception as e:
            logging.error(f"  LLM chunk {i + 1} failed: {e}")
            cleaned.append(chunk)
        time.sleep(0.1)

    return "\n".join(cleaned)


def pdf_to_images_b64(pdf_path: Path, dpi: int = 150) -> list:
    """Fallback: Convert PDF pages to base64 encoded JPEG images."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    images_b64 = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images_b64.append(b64)

    doc.close()
    return images_b64


def process_page_batch(batch_data: tuple, model: str) -> str:
    """Process a batch of pages in a single LLM request."""
    batch_num, images_b64_batch = batch_data
    page_range = f"{batch_num * len(images_b64_batch) + 1}-{batch_num * len(images_b64_batch) + len(images_b64_batch)}"

    content = [{"type": "text", "text": f"Pages {page_range}. {SYSTEM_PROMPT}"}]

    # Add all images from batch
    for b64_img in images_b64_batch:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
            }
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=8192,
            reasoning_effort="none",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Batch {batch_num} FAILED: {e}")
        return f"[Error processing batch {batch_num}]"


def clean_with_llm_images_parallel(
    images_b64: list, model: str, pages_per_request: int = 3, workers: int = 3
) -> str:
    """Process PDF pages in parallel with batching."""
    # Split images into batches
    batches = []
    for i in range(0, len(images_b64), pages_per_request):
        batches.append((i // pages_per_request, images_b64[i : i + pages_per_request]))

    logging.info(
        f"  Processing {len(images_b64)} pages in {len(batches)} batches with {workers} workers"
    )

    cleaned_batches = {}

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_page_batch, batch, model): batch[0]
            for batch in batches
        }

        for future in tqdm(
            as_completed(futures),
            total=len(batches),
            desc="  LLM Vision (parallel)",
            leave=False,
        ):
            batch_num = futures[future]
            try:
                result = future.result()
                cleaned_batches[batch_num] = result
            except Exception as e:
                logging.error(f"Batch {batch_num} exception: {e}")
                cleaned_batches[batch_num] = f"[Error processing batch {batch_num}]"

    # Reconstruct document in correct order
    cleaned_pages = [cleaned_batches[i] for i in sorted(cleaned_batches.keys())]
    return "\n\n".join(cleaned_pages)


def get_already_processed(output_dir: Path) -> set:
    """Get set of already processed PDF stems to enable resume."""
    if not output_dir.exists():
        return set()
    return {p.stem.replace("_cleaned", "") for p in output_dir.glob("*_cleaned.md")}


def process_folder(
    input_dir: Path,
    output_dir: Path,
    model: str,
    skip_existing: bool = True,
    dpi: int = 150,
    try_direct_pdf: bool = True,
    pages_per_request: int = 3,
    workers: int = 3,
    use_text_extraction: bool = True,
):
    """Process all PDFs in folder using hybrid approach."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        logging.warning("No PDF files found")
        return

    already_done = get_already_processed(output_dir) if skip_existing else set()
    if already_done:
        logging.info(f"Skipping {len(already_done)} already processed files")

    files_to_process = [p for p in pdf_files if p.stem not in already_done]
    if not files_to_process:
        logging.info("All files are already processed")
        return

    logging.info(f"Files to process: {len(files_to_process)} out of {len(pdf_files)}")
    logging.info(
        f"Optimization: {pages_per_request} pages/request, {workers} parallel workers"
    )

    # Separate text PDFs from scans
    text_pdfs = []
    scan_pdfs = []

    if use_text_extraction:
        logging.info("Checking PDFs for text layer...")
        for pdf_path in tqdm(files_to_process, desc="Checking text layer"):
            if has_text_layer(pdf_path):
                text_pdfs.append(pdf_path)
            else:
                scan_pdfs.append(pdf_path)
        logging.info(f"Text PDFs: {len(text_pdfs)}, Scans: {len(scan_pdfs)}")
    else:
        scan_pdfs = files_to_process

    # Test if API supports direct PDF upload (only for scans)
    use_direct_pdf = False
    if try_direct_pdf and scan_pdfs:
        logging.info("Testing if API supports direct PDF upload...")
        try:
            import fitz

            test_doc = fitz.open()
            test_doc.new_page()
            test_pdf_bytes = test_doc.tobytes()
            test_doc.close()
            test_pdf_b64 = base64.b64encode(test_pdf_bytes).decode("utf-8")

            clean_with_llm_direct_pdf(test_pdf_b64, model)
            use_direct_pdf = True
            logging.info("✓ API supports direct PDF upload - using fast mode for scans")
        except Exception as e:
            logging.info(
                f"✗ API does not support direct PDF ({e}) - falling back to image conversion"
            )
            use_direct_pdf = False

    # Process text PDFs (fast path)
    if text_pdfs:
        logging.info(f"Processing {len(text_pdfs)} text PDFs (fast path)...")
        for pdf_path in tqdm(text_pdfs, desc="Text PDFs"):
            try:
                raw_text = extract_text_fast(pdf_path)
                if not has_substantial_text(raw_text):
                    logging.warning(f"Skipped (empty): {pdf_path.name}")
                    continue

                cleaned_md = clean_with_llm_text(raw_text, model)
                out_path = output_dir / f"{pdf_path.stem}_cleaned.md"
                out_path.write_text(cleaned_md, encoding="utf-8")
            except Exception as e:
                logging.error(f"Error processing {pdf_path.name}: {e}")

    # Process scan PDFs (Vision API path)
    if scan_pdfs:
        logging.info(f"Processing {len(scan_pdfs)} scan PDFs (Vision API)...")
        for pdf_path in tqdm(scan_pdfs, desc="Scan PDFs"):
            try:
                if use_direct_pdf:
                    pdf_b64 = pdf_to_base64(pdf_path)
                    cleaned_md = clean_with_llm_direct_pdf(pdf_b64, model)
                else:
                    images_b64 = pdf_to_images_b64(pdf_path, dpi=dpi)
                    if not images_b64:
                        logging.warning(f"Skipped (no pages): {pdf_path.name}")
                        continue
                    cleaned_md = clean_with_llm_images_parallel(
                        images_b64, model, pages_per_request, workers
                    )

                out_path = output_dir / f"{pdf_path.stem}_cleaned.md"
                out_path.write_text(cleaned_md, encoding="utf-8")

            except Exception as e:
                logging.error(f"Error processing {pdf_path.name}: {e}")

    logging.info(f"Done! Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="PDF parser with hybrid text/vision approach"
    )
    parser.add_argument("-i", "--input", required=True, help="Input folder with PDFs")
    parser.add_argument("-o", "--output", default=None, help="Output folder")
    parser.add_argument("-m", "--model", default="google/gemma-4-31b", help="LLM model")
    parser.add_argument(
        "--no-skip", action="store_true", help="Reprocess already done files"
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="DPI for image conversion fallback"
    )
    parser.add_argument(
        "--no-direct-pdf",
        action="store_true",
        help="Skip direct PDF upload test, use images only",
    )
    parser.add_argument(
        "--pages-per-request",
        type=int,
        default=3,
        help="Number of pages to send in one LLM request (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers for LLM requests (default: 3)",
    )
    parser.add_argument(
        "--no-text-extraction",
        action="store_true",
        help="Disable text extraction, use Vision API for all PDFs",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear all model caches and exit"
    )
    parser.add_argument(
        "--show-cache", action="store_true", help="Show cache sizes and exit"
    )
    args = parser.parse_args()

    os.environ["HTTPX_LOG_LEVEL"] = "WARNING"

    if args.clear_cache:
        clear_all_caches()
        return

    if args.show_cache:
        cache_dirs = {"HuggingFace": os.environ.get("HF_HOME")}
        total = 0
        for name, path in cache_dirs.items():
            if path and Path(path).exists():
                size = sum(
                    f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
                )
                total += size
                print(f"{name:15s}: {path} ({size / 1e9:.2f} GB)")
            else:
                print(f"{name:15s}: does not exist")
        print(f"{'TOTAL':15s}: {total / 1e9:.2f} GB")
        return

    output_dir = (
        Path(args.output) if args.output else Path(config.BASE_DIR) / "cleaned_md"
    )

    process_folder(
        input_dir=Path(args.input),
        output_dir=output_dir,
        model=args.model,
        skip_existing=not args.no_skip,
        dpi=args.dpi,
        try_direct_pdf=not args.no_direct_pdf,
        pages_per_request=args.pages_per_request,
        workers=args.workers,
        use_text_extraction=not args.no_text_extraction,
    )


if __name__ == "__main__":
    main()
