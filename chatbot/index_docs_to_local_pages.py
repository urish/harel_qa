#!/usr/bin/env python3
import os
import sys
import pathlib
from typing import Iterable, List, Tuple, Dict, Optional

import click
import re
import unicodedata

# PDF parsing (Docling)
try:
	from docling_parse.pdf_parser import DoclingPdfParser
	from docling_core.types.doc.page import TextCellUnit
except Exception as e:
	DoclingPdfParser = None  # type: ignore
	TextCellUnit = None  # type: ignore

# HTML/ASPX parsing
from bs4 import BeautifulSoup

# Excel parsing
import pandas as pd

# Text fixing for encoding/mojibake
try:
	from ftfy import fix_text as ftfy_fix_text
except Exception:
	def ftfy_fix_text(text: str) -> str:  # fallback no-op if ftfy not installed
		return text


# Characters to strip (bidi and other direction/format controls)
_BIDI_CHARS = """
\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069
""".strip()
_BIDI_PATTERN = re.compile("[" + _BIDI_CHARS + "]")


def _normalize_whitespace(text: str) -> str:
	# Preserve paragraph breaks, collapse internal whitespace
	paragraphs = [p.strip() for p in re.split(r"\n{2,}", text)]
	collapsed: List[str] = []
	for p in paragraphs:
		p = re.sub(r"[\t\x0b\x0c\r]", " ", p)
		p = re.sub(r"\s{2,}", " ", p).strip()
		if p:
			collapsed.append(p)
	return "\n\n".join(collapsed)


def clean_text(text: str) -> str:
	"""Clean and normalize extracted text.

	- Fix mojibake and decoding issues
	- Normalize Unicode (NFC)
	- Remove bidi/format control chars that can break Hebrew/English rendering
	- Drop common boilerplate lines (very URL-heavy or nav-like)
	- Collapse excessive whitespace while preserving paragraphs
	"""
	if not text:
		return ""

	# Fix common mojibake
	text = ftfy_fix_text(text)

	# Unicode normalize
	text = unicodedata.normalize("NFC", text)

	# Remove bidi and control characters (keep newlines)
	text = _BIDI_PATTERN.sub("", text)
	text = "".join(ch for ch in text if (ch == "\n" or ch == "\t" or (not unicodedata.category(ch).startswith("C"))))

	# Line-level filtering to drop boilerplate
	lines = [ln.strip() for ln in text.splitlines()]
	filtered: List[str] = []
	for ln in lines:
		if not ln:
			continue
		# Skip pure navigation/footer/header like lines
		lower = ln.lower()
		if any(k in lower for k in ("cookies", "privacy policy", "terms of use", "sitemap")) and len(ln) < 120:
			continue
		if re.search(r"^(home|menu|navigation|contact us|all rights reserved)[\W\s]*$", lower):
			continue
		# Skip lines that are mostly URLs or crumbs
		url_tokens = re.findall(r"https?://|www\\.", ln, flags=re.I)
		if len(url_tokens) >= 2 and len(ln) < 160:
			continue
		# Skip social/link clusters
		if sum(1 for m in re.finditer(r"\b(login|signup|facebook|instagram|twitter|linkedin)\b", lower)) >= 2 and len(ln) < 150:
			continue
		filtered.append(ln)

	text = "\n".join(filtered)
	# Normalize whitespace and paragraphs
	text = _normalize_whitespace(text)
	return text


def _format_metadata(source_path: pathlib.Path, kind: str, identifier: str) -> str:
	"""Produce a stable, machine- and human-readable metadata header for citations.

	Example: <<<SOURCE:/abs/path/file.pdf | PAGE:3>>>
	"""
	return f"<<<SOURCE:{str(source_path)} | {kind.upper()}:{identifier}>>>"


def _split_into_sections(text: str, max_chars: int = 2000) -> List[str]:
	"""Split cleaned text into sections capped by max_chars, preserving paragraph boundaries."""
	if not text:
		return []
	paragraphs = text.split("\n\n")
	sections: List[str] = []
	current: List[str] = []
	current_len = 0
	for p in paragraphs:
		p_len = len(p)
		if current and current_len + 2 + p_len > max_chars:
			sections.append("\n\n".join(current))
			current = [p]
			current_len = p_len
		else:
			if current:
				current_len += 2 + p_len
				current.append(p)
			else:
				current = [p]
				current_len = p_len
	if current:
		sections.append("\n\n".join(current))
	return sections


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".aspx", ".xlsx")


def iter_files(root_dir: pathlib.Path, extensions: Tuple[str, ...]) -> Iterable[pathlib.Path]:
	for dirpath, _, filenames in os.walk(root_dir):
		for filename in filenames:
			path = pathlib.Path(dirpath) / filename
			if path.suffix.lower() in extensions:
				yield path


def ensure_parent_dir(path: pathlib.Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def _create_pdf_parser(enable_ocr: bool) -> "DoclingPdfParser":
	"""Create a DoclingPdfParser with OCR toggled when supported.

	Tries multiple common constructor signatures/attributes to disable OCR for speed.
	Falls back to default constructor if options are unavailable in the installed version.
	"""
	if DoclingPdfParser is None:
		raise RuntimeError("Docling is not installed. Install 'docling-parse' and 'docling-core'.")
	# Try known kwargs first
	for kwargs in (
		{"ocr": enable_ocr},
		{"do_ocr": enable_ocr},
		{"enable_ocr": enable_ocr},
		# Some versions may support backend selection; dlparse_v2 typically avoids OCR
		({"pdf_backend": "dlparse_v2"} if not enable_ocr else {}),
	):
		try:
			if kwargs:
				return DoclingPdfParser(**kwargs)  # type: ignore[arg-type]
		except TypeError:
			pass
	# Fallback to default then try to set attributes
	parser = DoclingPdfParser()
	for attr in ("ocr", "do_ocr", "enable_ocr"):
		if hasattr(parser, attr):
			try:
				setattr(parser, attr, enable_ocr)
			except Exception:
				pass
	return parser


def process_pdf(file_path: pathlib.Path, enable_ocr: bool = False) -> str:
	if DoclingPdfParser is None or TextCellUnit is None:
		raise RuntimeError("Docling is not installed. Install 'docling-parse' and 'docling-core'.")

	parser = _create_pdf_parser(enable_ocr)
	pdf_doc = parser.load(path_or_stream=str(file_path))
	page_texts: List[str] = []
	page_index = 0
	for _, pred_page in pdf_doc.iterate_pages():
		page_index += 1
		words: List[str] = []
		for cell in pred_page.iterate_cells(unit_type=TextCellUnit.WORD):
			words.append(cell.text)
		page_raw = " ".join(words)
		page_clean = clean_text(page_raw)
		if page_clean:
			header = _format_metadata(file_path.resolve(), "page", str(page_index))
			page_texts.append(f"{header}\n{page_clean}")
	return "\n\n".join(page_texts)


def process_aspx(file_path: pathlib.Path) -> str:
	# ASPX files are HTML; parse and extract visible text
	with file_path.open("r", encoding="utf-8", errors="ignore") as f:
		html = f.read()
	soup = BeautifulSoup(html, "lxml")
	# Remove script/style and structural boilerplate
	for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
		tag.decompose()
	# Remove elements with common nav/footer identifiers
	for el in soup.select('[role="navigation"], [role="banner"], [role="contentinfo"], [role="complementary"], .nav, .navbar, .breadcrumb, .menu, #nav, #navbar, #header, #footer, .header, .footer'):
		try:
			el.decompose()
		except Exception:
			pass
	text = soup.get_text(separator=" ", strip=True)
	cleaned = clean_text(text)
	sections = _split_into_sections(cleaned)
	out: List[str] = []
	for i, sec in enumerate(sections, start=1):
		header = _format_metadata(file_path.resolve(), "section", str(i))
		out.append(f"{header}\n{sec}")
	return "\n\n".join(out)


def process_xls(file_path: pathlib.Path) -> str:
	# Use pandas; xlrd supports .xls
	dfs: Dict[str, pd.DataFrame] = pd.read_excel(str(file_path), sheet_name=None, engine="xlrd")
	parts: List[str] = []
	for sheet_name, df in dfs.items():
		# Convert to TSV-like text for readability
		tsv = df.to_csv(sep="\t", index=False)
		cleaned = clean_text(tsv)
		sections = _split_into_sections(cleaned)
		for i, sec in enumerate(sections, start=1):
			header = _format_metadata(file_path.resolve(), "sheet", f"{sheet_name} | section {i}")
			parts.append(f"{header}\n{sec}")
	return "\n\n".join(parts)


def derive_output_path(input_root: pathlib.Path, output_root: pathlib.Path, file_path: pathlib.Path) -> pathlib.Path:
	rel = file_path.relative_to(input_root)
	return (output_root / rel).with_suffix(".txt")


@click.command()
@click.option("--input-dir", type=click.Path(path_type=pathlib.Path, exists=True, file_okay=False), default=pathlib.Path("data"), help="Directory containing source documents (.pdf, .aspx, .xls)")
@click.option("--output-dir", type=click.Path(path_type=pathlib.Path, file_okay=False), default=pathlib.Path("parsed"), help="Directory to write extracted .txt files")
@click.option("--silent/--no-silent", default=False, help="Suppress per-file logs")
@click.option("--ocr/--no-ocr", default=False, help="Enable OCR during PDF parsing (disabled by default for speed)")
def main(input_dir: pathlib.Path, output_dir: pathlib.Path, silent: bool, ocr: bool) -> None:
	input_dir = input_dir.resolve()
	output_dir = output_dir.resolve()

	if not input_dir.exists():
		click.echo(f"Input directory not found: {input_dir}", err=True)
		sys.exit(1)

	processed = 0
	errors: List[Tuple[pathlib.Path, str]] = []

	for path in iter_files(input_dir, SUPPORTED_EXTENSIONS):
		try:
			ext = ""
			if path.suffix.lower() == ".pdf":
				ext = process_pdf(path, enable_ocr=ocr)
			elif path.suffix.lower() == ".aspx":
				ext = process_aspx(path)
			elif path.suffix.lower() == ".xls":
				ext = process_xls(path)
			else:
				continue

			out_path = derive_output_path(input_dir, output_dir, path)
			ensure_parent_dir(out_path)
			out_path.write_text(ext, encoding="utf-8")
			processed += 1
			if not silent:
				click.echo(f"Parsed: {path} -> {out_path}")
		except Exception as e:
			errors.append((path, str(e)))
			if not silent:
				click.echo(f"Failed: {path} ({e})", err=True)

	click.echo(f"Done. Processed: {processed}, Errors: {len(errors)}")
	if errors:
		for p, msg in errors[:10]:
			click.echo(f" - {p}: {msg}", err=True)


if __name__ == "__main__":
	main()
