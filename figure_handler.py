import re
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from utils import Figure
from http_session import HTTPSession


class ImageDownloader:
    """Handles figure extraction and downloading functionality."""

    def __init__(self, http_session: HTTPSession, base_dir: str = "articles_data"):
        self.http_session = http_session
        self.base_dir = base_dir

    def _extract_caption(self, img_tag) -> str:
        """Extract caption text from image context."""
        caption_selectors = [
            "figcaption",
            ".caption",
            ".fig-caption",
            ".figure-caption",
        ]

        parent = img_tag.parent
        while parent and parent.name != "body":
            for selector in caption_selectors:
                caption = parent.select_one(selector)
                if caption:
                    return caption.get_text().strip()
            parent = parent.parent

        return ""

    def _is_likely_protein_figure(self, img_tag, src: str) -> bool:
        """Determine if image is likely a protein figure."""
        skip_patterns = [
            r"logo",
            r"icon",
            r"button",
            r"avatar",
            r"banner",
            r"header",
            r"footer",
            r"nav",
            r"menu",
            r"social",
            r"advertisement",
            r"ad_",
            r"tracking",
            r"author",
            r"journal",
            r"publisher",
            r"copyright",
            # Add medical imaging exclusions at the HTML level
            r"scintigraphy",
            r"bone.scan",
            r"x.?ray",
            r"ct.scan",
            r"mri",
            r"ultrasound",
            r"skeleton",
            r"anatomical",
            r"medical",
            r"clinical",
            r"patient",
            r"radiograph",
        ]

        src_lower = src.lower()
        alt_lower = img_tag.get("alt", "").lower()

        # Skip obvious non-figures
        for pattern in skip_patterns:
            if re.search(pattern, src_lower, re.IGNORECASE) or re.search(
                pattern, alt_lower, re.IGNORECASE
            ):
                return False

        # Protein aggregation kinetic indicators
        kinetic_indicators = [
            r"fig",
            r"figure",
            r"kinetic",
            r"aggregation",
            r"curve",
            r"plot",
            r"graph",
            r"time",
            r"fluorescence",
            r"thioflavin",
            r"congo",
            r"red",
            r"absorbance",
            r"intensity",
            r"fibril",
            r"amyloid",
            r"protein",
            r"growth",
            r"nucleation",
            r"lag",
            r"tht",
            r"cr",
            r"alpha.*synuclein",
            r"tau",
            r"huntingtin",
        ]

        caption = self._extract_caption(img_tag).lower()
        text_to_check = f"{src_lower} {alt_lower} {caption}"

        kinetic_score = 0
        for indicator in kinetic_indicators:
            if re.search(indicator, text_to_check, re.IGNORECASE):
                kinetic_score += 1

        # Strong indicators for protein aggregation
        if kinetic_score >= 2:
            return True

        # Check parent elements for figure context
        parent = img_tag.parent
        while parent and parent.name != "body":
            classes = parent.get("class", [])
            class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
            if re.search(r"fig|figure|image", class_str, re.IGNORECASE):
                return True
            parent = parent.parent

        return kinetic_score > 0

    def get_pmc_figures_from_html(
        self, html_content: str, pmc_url: str
    ) -> List[Figure]:
        """Extract figures from already-fetched HTML content.
        Only returns figures that have captions."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            figures = []

            base_url = f"{urlparse(pmc_url).scheme}://{urlparse(pmc_url).netloc}"

            for img in soup.find_all("img", src=True):
                src = img["src"]

                # Convert relative URLs to absolute
                if src.startswith("/"):
                    src = base_url + src
                elif not src.startswith("http"):
                    src = urljoin(pmc_url, src)

                # Only process figures that are likely protein-related AND have captions
                if self._is_likely_protein_figure(img, src) and self._has_caption(img):
                    figures.append(
                        Figure(
                            url=src,
                            alt=img.get("alt", ""),
                            caption=self._extract_caption(img),
                            element=img,
                        )
                    )

            return figures

        except Exception as e:
            print(f"Error extracting figures from HTML: {e}")
            return []

    def get_pmc_figures(self, pmc_url: str) -> List[Figure]:
        """Extract figures from PMC article (legacy method - prefer using HTML from ContentFetcher)."""
        try:
            response = self.http_session.get(pmc_url)
            response.raise_for_status()
            return self.get_pmc_figures_from_html(response.text, pmc_url)

        except Exception as e:
            print(f"Error extracting figures from {pmc_url}: {e}")
            return []

    def download_image(self, img_url: str, filename: str, pmid: str) -> Optional[str]:
        """Download image from URL to the PMID-specific folder."""
        try:
            response = self.http_session.get(img_url, stream=True)
            response.raise_for_status()

            # Create PMID-specific directory
            pmid_dir = Path(self.base_dir) / pmid / "images"
            pmid_dir.mkdir(parents=True, exist_ok=True)

            filepath = pmid_dir / filename

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return str(filepath)

        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
            return None

    def _has_caption(self, img_tag) -> bool:
        """Check if the image has a non-empty caption."""
        caption = self._extract_caption(img_tag)
        return bool(caption and caption.strip())
