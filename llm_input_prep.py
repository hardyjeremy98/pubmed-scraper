"""
LLM Input Preparation Module

This module provides flexible text extraction functions for figure context preparation.
"""

import re
import os
from typing import Dict, Tuple, Optional
from utils import Figure


def extract_figure_text(
    article_text: str,
    figure_number: int,
    figure_data: Figure,
    pmid: str,
    base_dir: str = "articles_data",
    include_surrounding_text: bool = True,
    context_words: int = 250,
) -> Tuple[str, str]:
    """
    Extract text context for a single figure with flexible options.

    Args:
        article_text: Full text content of the article
        figure_number: Figure number to process
        figure_data: Figure object with metadata
        pmid: PubMed ID for constructing image path
        base_dir: Base directory for article data
        include_surrounding_text: If True, includes surrounding text; if False, only caption
        context_words: Number of words to extract before and after figure references (ignored if include_surrounding_text=False)

    Returns:
        Tuple of (combined_text, image_path)
    """
    # Start with the figure caption
    caption_text = figure_data.caption.strip() if figure_data.caption else ""

    if not include_surrounding_text:
        # Return just the caption
        combined_text = caption_text
    else:
        # Extract surrounding text and combine with caption
        preceding_text, following_text = _extract_surrounding_text(
            article_text, figure_number, figure_data, context_words
        )

        # Combine caption with surrounding text
        text_parts = []
        if preceding_text:
            text_parts.append(preceding_text)
        if caption_text:
            text_parts.append(f"Figure {figure_number} Caption: {caption_text}")
        if following_text:
            text_parts.append(following_text)

        combined_text = " ".join(text_parts)

    # Construct image path
    image_path = os.path.join(base_dir, pmid, "images", f"figure_{figure_number}.jpg")

    return combined_text, image_path


def _extract_surrounding_text(
    text: str, figure_number: int, figure_data: Figure, context_words: int
) -> Tuple[str, str]:
    """
    Extract text before and after figure caption matches.

    Args:
        text: Full article text
        figure_number: Figure number to search for
        figure_data: Figure object with caption
        context_words: Number of words to extract before and after

    Returns:
        Tuple of (preceding_text, following_text)
    """

    # Try to find figure caption in the text
    caption_positions = []
    if figure_data.caption:
        caption_text = figure_data.caption.strip()

        # Try exact match first
        caption_match = re.search(re.escape(caption_text), text, re.IGNORECASE)
        if caption_match:
            caption_positions.append(caption_match.start())

        # If no exact match, try partial matches (first sentence of caption)
        if not caption_positions and len(caption_text) > 20:
            first_sentence = caption_text.split(".")[0]
            if len(first_sentence) > 20:
                partial_match = re.search(
                    re.escape(first_sentence), text, re.IGNORECASE
                )
                if partial_match:
                    caption_positions.append(partial_match.start())

    # If caption matching failed, fall back to figure reference patterns
    if not caption_positions:
        patterns = [
            rf"\bFig\.?\s*{figure_number}\b",
            rf"\bFigure\s*{figure_number}\b",
            rf"\bfig\.?\s*{figure_number}\b",
            rf"\bfigure\s*{figure_number}\b",
        ]

        reference_positions = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                reference_positions.append(match.start())

        if not reference_positions:
            return "", ""

        main_position = min(reference_positions)
    else:
        main_position = min(caption_positions)

    # Split text into words for context extraction
    words = text.split()

    # Find word positions
    word_positions = []
    current_pos = 0
    for word in words:
        word_start = text.find(word, current_pos)
        word_positions.append(word_start)
        current_pos = word_start + len(word)

    # Find the word index closest to the figure reference
    reference_word_idx = 0
    for i, pos in enumerate(word_positions):
        if pos <= main_position:
            reference_word_idx = i
        else:
            break

    # Extract surrounding words
    start_idx = max(0, reference_word_idx - context_words)
    end_idx = min(len(words), reference_word_idx + context_words + 1)

    preceding_words = words[start_idx:reference_word_idx]
    following_words = words[reference_word_idx:end_idx]

    preceding_text = " ".join(preceding_words)
    following_text = " ".join(following_words)

    return preceding_text, following_text
