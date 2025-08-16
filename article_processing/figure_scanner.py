import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils import Figure


@dataclass
class ScanResult:
    """Result of scanning figures for keywords."""

    has_relevant_figures: bool
    relevant_figure_numbers: List[int]
    total_figures: int
    keyword_matches: Dict[int, List[str]]  # figure_id -> list of matched keywords


class FigureKeywordScanner:
    """
    Scans figure captions for specific keywords related to protein aggregation research.

    This class is designed to identify figures containing Thioflavin T (ThT) or related
    keywords that indicate potential protein aggregation assays or measurements.
    """

    def __init__(
        self, keywords: Optional[List[str]] = None, case_sensitive: bool = False
    ):
        """
        Initialize the scanner with keywords to search for.

        Args:
            keywords: List of keywords to search for. Defaults to ThT-related terms.
            case_sensitive: Whether to perform case-sensitive matching. Default is False.
        """
        if keywords is None:
            # Default keywords related to Thioflavin T and protein aggregation
            self.keywords = [
                "ThT",
                "Thioflavin",
                "thioflavin-T",
                "thioflavin-t",
            ]
        else:
            self.keywords = keywords

        self.case_sensitive = case_sensitive

        # Compile regex patterns for efficient searching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self.patterns = []

        for keyword in self.keywords:
            # Create word boundary patterns to avoid partial matches
            # Use \b for word boundaries, but handle special cases like "ThT"
            if keyword.upper() in ["THT"]:
                # For abbreviations like ThT, be more flexible with boundaries
                pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            else:
                # For full words, use standard word boundaries
                pattern = rf"\b{re.escape(keyword)}\b"

            self.patterns.append(re.compile(pattern, flags))

    def scan_figure_caption(
        self, caption: str, figure_id: int
    ) -> Tuple[bool, List[str]]:
        """
        Scan a single figure caption for keywords.

        Args:
            caption: The figure caption text to scan
            figure_id: The figure identifier (for logging/tracking)

        Returns:
            Tuple of (has_keywords, matched_keywords)
        """
        if not caption or not caption.strip():
            return False, []

        matched_keywords = []

        for i, pattern in enumerate(self.patterns):
            if pattern.search(caption):
                matched_keywords.append(self.keywords[i])

        return len(matched_keywords) > 0, matched_keywords

    def scan_figures(self, figures: List[Figure]) -> ScanResult:
        """
        Scan a list of Figure objects for keyword matches in captions.

        Args:
            figures: List of Figure dataclass instances to scan

        Returns:
            ScanResult containing analysis results
        """
        if not figures:
            return ScanResult(
                has_relevant_figures=False,
                relevant_figure_numbers=[],
                total_figures=0,
                keyword_matches={},
            )

        relevant_figure_numbers = []
        keyword_matches = {}

        for i, figure in enumerate(figures):
            figure_number = i + 1  # 1-based indexing for human readability
            has_keywords, matched_keywords = self.scan_figure_caption(
                figure.caption, figure_number
            )

            if has_keywords:
                relevant_figure_numbers.append(figure_number)
                keyword_matches[figure_number] = matched_keywords

        return ScanResult(
            has_relevant_figures=len(relevant_figure_numbers) > 0,
            relevant_figure_numbers=relevant_figure_numbers,
            total_figures=len(figures),
            keyword_matches=keyword_matches,
        )

    def scan_figures_from_json_data(self, figures_data: List[Dict]) -> ScanResult:
        """
        Scan figures from JSON data format (as saved by ArticleManager.save_figures_to_json).

        Args:
            figures_data: List of dictionaries containing figure metadata

        Returns:
            ScanResult containing analysis results
        """
        if not figures_data:
            return ScanResult(
                has_relevant_figures=False,
                relevant_figure_numbers=[],
                total_figures=0,
                keyword_matches={},
            )

        relevant_figure_numbers = []
        keyword_matches = {}

        for figure_data in figures_data:
            figure_id = figure_data.get("id", 0)
            caption = figure_data.get("caption", "")

            has_keywords, matched_keywords = self.scan_figure_caption(
                caption, figure_id
            )

            if has_keywords:
                relevant_figure_numbers.append(figure_id)
                keyword_matches[figure_id] = matched_keywords

        return ScanResult(
            has_relevant_figures=len(relevant_figure_numbers) > 0,
            relevant_figure_numbers=relevant_figure_numbers,
            total_figures=len(figures_data),
            keyword_matches=keyword_matches,
        )

    def print_scan_results(self, scan_result: ScanResult, pmid: str = None):
        """
        Print a formatted summary of scan results.

        Args:
            scan_result: The ScanResult to display
            pmid: Optional PMID for context in the output
        """
        header = f"Keyword Scan Results"
        if pmid:
            header += f" for PMID: {pmid}"

        print(f"\n{header}")
        print("=" * len(header))

        print(f"Total figures scanned: {scan_result.total_figures}")
        print(f"Relevant figures found: {len(scan_result.relevant_figure_numbers)}")

        if scan_result.has_relevant_figures:
            print(
                f"Figure numbers with keywords: {scan_result.relevant_figure_numbers}"
            )
            print("\nKeyword matches by figure:")
            for figure_num, keywords in scan_result.keyword_matches.items():
                print(f"  Figure {figure_num}: {', '.join(keywords)}")
            print("\n✓ Article flagged as relevant for further processing")
        else:
            print("No keyword matches found")
            print("✗ Article can be skipped for keyword-based processing")

    def add_keywords(self, new_keywords: List[str]):
        """
        Add additional keywords to search for.

        Args:
            new_keywords: List of additional keywords to add
        """
        self.keywords.extend(new_keywords)
        self._compile_patterns()

    def remove_keywords(self, keywords_to_remove: List[str]):
        """
        Remove keywords from the search list.

        Args:
            keywords_to_remove: List of keywords to remove
        """
        self.keywords = [kw for kw in self.keywords if kw not in keywords_to_remove]
        self._compile_patterns()

    def get_keywords(self) -> List[str]:
        """Return the current list of keywords being searched for."""
        return self.keywords.copy()


def scan_article_figures_for_keywords(
    figures: List[Figure],
    pmid: str = None,
    keywords: Optional[List[str]] = None,
    verbose: bool = True,
) -> ScanResult:
    """
    Convenience function to scan article figures for keywords.

    Args:
        figures: List of Figure objects to scan
        pmid: Optional PMID for context in output
        keywords: Optional custom keywords (defaults to ThT-related terms)
        verbose: Whether to print results

    Returns:
        ScanResult with analysis results
    """
    scanner = FigureKeywordScanner(keywords=keywords)
    result = scanner.scan_figures(figures)

    if verbose:
        scanner.print_scan_results(result, pmid)

    return result
