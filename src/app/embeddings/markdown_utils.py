"""Markdown conversion helpers for embeddings ingestion."""
from __future__ import annotations

from typing import Dict

import re
from bs4 import BeautifulSoup, Comment
import html2text
from markdownify import markdownify as mdf


def page_as_md(body_html: str) -> str:
    """Convert Confluence storage HTML (already provided) to markdown."""
    return mdf(body_html, strip=["ac:structured-macro"])


def clean_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    remove_tags = ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    for tag in soup(remove_tags):
        tag.decompose()
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()
    noisy_keywords = ["advert", "promo", "sidebar", "cookie", "tracking"]
    for div in soup.find_all("div"):
        classes = " ".join(div.get("class", []))
        if any(k in classes.lower() for k in noisy_keywords):
            div.decompose()
    return soup


def html_to_markdown(cleaned_soup) -> str:
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    return converter.handle(str(cleaned_soup))


def normalize_markdown(md: str) -> str:
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
