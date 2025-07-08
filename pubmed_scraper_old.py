import re
import os
import json
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import urllib
from urllib.parse import urlparse, urljoin


def create_pmc_url(pmcid: str) -> str:
    """Create PMC article URL from PMCID."""
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}"


def ensure_pmid_directory(pmid: str, base_dir: str = "articles_data") -> str:
    """Create and return PMID-specific directory path."""
    pmid_dir = os.path.join(base_dir, pmid)
    os.makedirs(pmid_dir, exist_ok=True)
    return pmid_dir


@dataclass
class ArticleMetadata:
    """Data class to hold article metadata."""

    pmid: str
    pmcid: Optional[str] = None
    title: str = ""
    doi: str = ""
    journal: str = ""
    source: str = ""
    content: str = ""
    html_content: Optional[str] = None  # Store raw HTML content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "title": self.title,
            "doi": self.doi,
            "journal": self.journal,
            "source": self.source,
            "content": self.content,
            "html_content": self.html_content,
        }


@dataclass
class Figure:
    url: str
    alt: str
    caption: str
    element: Optional[any] = None


class HTTPSession:
    """Manages HTTP session with proper headers."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make GET request using the session."""
        return self.session.get(url, **kwargs)


class MetadataFetcher:
    """Handles fetching article metadata from various sources."""

    def __init__(self, email: str, http_session: HTTPSession):
        Entrez.email = email
        self.http_session = http_session
        self._cache: Dict[str, ArticleMetadata] = {}

    def get_article_metadata(self, pmid: str) -> ArticleMetadata:
        """
        Retrieve article metadata (PMID, PMCID, title, DOI, journal).
        Uses caching to avoid repeated API calls.
        """
        # Check cache first
        if pmid in self._cache:
            return self._cache[pmid]

        # Try Europe PMC first
        metadata = self._fetch_metadata_from_epmc(pmid)

        # Fallback to PubMed if EPMC fails
        if metadata is None:
            metadata = self._fetch_metadata_from_pubmed(pmid)

        if not metadata.pmcid:
            fallback_pmcid = self._fetch_pmcid_from_ncbi(pmid)
            if fallback_pmcid:
                metadata.pmcid = fallback_pmcid

        # Cache the result
        self._cache[pmid] = metadata
        return metadata

    def _fetch_metadata_from_epmc(self, pmid: str) -> Optional[ArticleMetadata]:
        """Fetch metadata from Europe PMC API."""
        epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json"

        try:
            r = self.http_session.get(epmc_url)
            if r.status_code == 200 and r.json().get("hitCount", 0) > 0:
                doc = r.json()["resultList"]["result"][0]
                return ArticleMetadata(
                    pmid=pmid,
                    pmcid=doc.get("pmcid"),
                    title=doc.get("title", ""),
                    doi=doc.get("doi", ""),
                    journal=doc.get("journalTitle", ""),
                )
        except Exception as e:
            print(f"Error fetching from EPMC for PMID {pmid}: {e}")

        return None

    def _fetch_pmcid_from_ncbi(self, pmid: str) -> Optional[str]:
        """Try to fetch the PMCID using NCBI's eLink API as a backup."""
        try:
            handle = Entrez.elink(
                dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc"
            )
            records = Entrez.read(handle)
            linksets = records[0].get("LinkSetDb", [])
            for linkset in linksets:
                for link in linkset.get("Link", []):
                    pmcid_num = link.get("Id")
                    pmcid = f"PMC{pmcid_num}"
                    if pmcid:
                        return pmcid
        except Exception as e:
            print(f"NCBI eLink error for PMID {pmid}: {e}")
        return None

    def _fetch_metadata_from_pubmed(self, pmid: str) -> ArticleMetadata:
        """Fetch metadata from PubMed API as fallback."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record = Entrez.read(handle)
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]

            title = article.get("ArticleTitle", "")
            journal = article.get("Journal", {}).get("Title", "")
            doi = ""

            # Extract DOI if available
            for eloc in article.get("ELocationID", []):
                if (
                    hasattr(eloc, "attributes")
                    and eloc.attributes.get("EIdType") == "doi"
                ):
                    doi = str(eloc)
                    break

            return ArticleMetadata(pmid=pmid, title=title, doi=doi, journal=journal)

        except Exception as e:
            print(f"Error fetching from PubMed for PMID {pmid}: {e}")
            return ArticleMetadata(pmid=pmid)

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._cache.clear()

    def get_cached_article(self, pmid: str) -> Optional[ArticleMetadata]:
        """Get article from cache if available."""
        return self._cache.get(pmid)


class ContentFetcher:
    """Handles fetching full text content, abstracts, and HTML."""

    def __init__(self, email: str, http_session: HTTPSession):
        Entrez.email = email
        self.http_session = http_session
        self._html_cache: Dict[str, str] = {}  # Cache HTML content by PMC URL

    def get_full_article_content(self, article: ArticleMetadata) -> ArticleMetadata:
        """
        Get complete article data including full text, abstract, and HTML.
        """
        # If we have a PMCID, fetch HTML first and derive content from it
        if article.pmcid and not article.html_content:
            pmc_url = create_pmc_url(article.pmcid)
            html_content = self._fetch_html_from_pmc(pmc_url)
            if html_content:
                article.html_content = html_content
                # Extract text content from HTML if we don't have content yet
                if not article.content:
                    content = self._extract_text_from_html(html_content)
                    if content:
                        article.content = content
                        article.source = "fulltext"

        # If still no content, try to get abstract
        if not article.content:
            abstract = self._fetch_abstract_from_pubmed(article.pmid)
            if abstract:
                article.content = abstract
                article.source = "abstract"

        return article

    def _fetch_html_from_pmc(self, pmc_url: str) -> Optional[str]:
        """Fetch raw HTML content from PMC article with caching."""
        # Check cache first
        if pmc_url in self._html_cache:
            print("Using cached HTML content")
            return self._html_cache[pmc_url]

        try:
            print(f"Fetching HTML from: {pmc_url}")
            response = self.http_session.get(pmc_url)
            response.raise_for_status()

            if response.status_code == 200:
                print("✓ Successfully fetched HTML content")
                # Cache the HTML content
                self._html_cache[pmc_url] = response.text
                return response.text
            else:
                print(f"Failed to fetch HTML: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching HTML from {pmc_url}: {e}")
            return None

    def _extract_text_from_html(self, html_content: str) -> Optional[str]:
        """Extract clean text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            text_parts = [
                sec.get_text(strip=True) for sec in soup.find_all(["title", "p"])
            ]
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return None

    def get_html_content(self, article: ArticleMetadata) -> Optional[str]:
        """Get HTML content for an article, using cache if available."""
        if article.html_content:
            return article.html_content

        if article.pmcid:
            pmc_url = create_pmc_url(article.pmcid)
            html_content = self._fetch_html_from_pmc(pmc_url)
            if html_content:
                article.html_content = html_content
            return html_content

        return None

    def _fetch_fulltext_from_pmc(self, pmcid: str) -> Optional[str]:
        """Retrieve fulltext from PMC if available (deprecated - use get_full_article_content)."""
        if not pmcid:
            return None

        print(f"Fetching full text for PMCID: {pmcid}")
        pmc_url = create_pmc_url(pmcid)
        html_content = self._fetch_html_from_pmc(pmc_url)

        if html_content:
            return self._extract_text_from_html(html_content)

        return None

    def _fetch_abstract_from_pubmed(self, pmid: str) -> Optional[str]:
        """Fetch abstract from PubMed."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record = Entrez.read(handle)
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]

            abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
            if abstract_sections:
                if isinstance(abstract_sections, list):
                    return " ".join(str(section) for section in abstract_sections)
                else:
                    return str(abstract_sections)

        except Exception as e:
            print(f"Error fetching abstract for PMID {pmid}: {e}")

        return None


class PDFFinder:
    """Contains various PDF finder strategies for different publishers."""

    def __init__(self, http_session: HTTPSession):
        self.http_session = http_session

    @staticmethod
    def _get_main_url(url: str) -> str:
        """Extract main URL from full URL."""
        return "/".join(url.split("/")[:3])

    def acs_publications(self, req, soup) -> Optional[str]:
        """ACS Publications PDF finder."""
        links = [
            x
            for x in soup.find_all("a")
            if x.get("title") and "pdf" in x.get("title").lower()
        ]
        if links:
            print("** Using ACS Publications finder...")
            return self._get_main_url(req.url) + links[0].get("href")
        return None

    def future_medicine(self, req, soup) -> Optional[str]:
        """Future Medicine PDF finder."""
        links = soup.find_all("a", href=re.compile("/doi/pdf"))
        if links:
            print("** Using Future Medicine finder...")
            return self._get_main_url(req.url) + links[0].get("href")
        return None

    def generic_citation_labelled(self, req, soup) -> Optional[str]:
        """Generic citation PDF finder."""
        meta = soup.find_all("meta", attrs={"name": "citation_pdf_url"})
        if meta:
            print("** Using Generic Citation Labelled finder...")
            return meta[0].get("content")
        return None

    def nejm(self, req, soup) -> Optional[str]:
        """NEJM PDF finder."""
        links = [
            x
            for x in soup.find_all("a")
            if x.get("data-download-type") == "article pdf"
        ]
        if links:
            print("** Using NEJM finder...")
            return self._get_main_url(req.url) + links[0].get("href")
        return None

    def pubmed_central_v2(self, req, soup) -> Optional[str]:
        """PubMed Central V2 PDF finder."""
        links = soup.find_all("a", href=re.compile("/pmc/articles"))
        if links:
            print("** Using PubMed Central V2 finder...")
            return f"https://www.ncbi.nlm.nih.gov{links[0].get('href')}"
        return None

    def science_direct(self, req, soup) -> Optional[str]:
        """Science Direct PDF finder."""
        try:
            new_uri = urllib.parse.unquote(soup.find_all("input")[0].get("value"))
            req = self.http_session.get(new_uri)
            req.raise_for_status()
            soup = BeautifulSoup(req.content, "lxml")
            meta = soup.find_all("meta", attrs={"name": "citation_pdf_url"})
            if meta:
                print("** Using Science Direct finder...")
                return meta[0].get("content")
        except Exception as e:
            print(f"** Science Direct error: {e}")
        return None

    def uchicago_press(self, req, soup) -> Optional[str]:
        """UChicago Press PDF finder."""
        links = [
            x
            for x in soup.find_all("a")
            if x.get("href") and "pdf" in x.get("href") and ".edu/doi/" in x.get("href")
        ]
        if links:
            print("** Using UChicago Press finder...")
            return self._get_main_url(req.url) + links[0].get("href")
        return None

    def europe_pmc_service(self, req, soup) -> Optional[str]:
        """Europe PMC Service PDF finder."""
        pmc_match = re.search(r"PMC\d+", req.url)
        if pmc_match:
            pmc_id = pmc_match.group()
            print(f"** Using Europe PMC Service finder for {pmc_id}...")
            return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc_id}&blobtype=pdf"
        return None


class PDFDownloader:
    """Handles PDF downloading functionality."""

    def __init__(self, http_session: HTTPSession, base_dir: str = "articles_data"):
        self.http_session = http_session
        self.base_dir = base_dir
        self.pdf_finder = PDFFinder(http_session)

    def download_pdf_from_pmcid(
        self,
        pmcid: str,
        pmid: str,
        error_file: str = "unfetched_pmcids.tsv",
        max_retries: int = 3,
    ) -> None:
        """Download PDF for a given PMCID into the PMID-specific folder."""
        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(pmid, self.base_dir)

        finders = [
            self.pdf_finder.europe_pmc_service,
            self.pdf_finder.generic_citation_labelled,
            self.pdf_finder.pubmed_central_v2,
            self.pdf_finder.acs_publications,
            self.pdf_finder.uchicago_press,
            self.pdf_finder.nejm,
            self.pdf_finder.future_medicine,
            self.pdf_finder.science_direct,
        ]

        # Create error file in base directory
        error_file_path = os.path.join(self.base_dir, error_file)
        with open(error_file_path, "a") as error_pmids:  # Changed to append mode
            name = pmcid
            print(f"\n--- Fetching {pmcid} ---")
            retries = 0
            while retries < max_retries:
                try:
                    self._fetch_pdf(pmcid, finders, name, error_pmids, pmid_dir)
                    break
                except requests.ConnectionError as e:
                    if "104" in str(e):
                        retries += 1
                        print(f"** Retry {retries}/{max_retries} for error 104")
                        time.sleep(2)
                    else:
                        error_pmids.write(f"{pmcid}\t{name}\n")
                        break
                except Exception as e:
                    print(f"** Error fetching {pmcid}: {e}")
                    error_pmids.write(f"{pmcid}\t{name}\n")
                    break

            time.sleep(0.3)  # to avoid rate limiting

    def _fetch_pdf(
        self, pmcid: str, finders: List, name: str, error_pmids, pdf_dir: str
    ) -> None:
        """Internal method to fetch PDF using various finders."""
        uri = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid.strip()}"
        success = False
        output_path = f"{pdf_dir}/{pmcid}.pdf"

        if os.path.exists(output_path):
            print(f"** Reprint #{pmcid} already downloaded; skipping.")
            return

        try:
            req = self.http_session.get(uri)
            req.raise_for_status()
            soup = BeautifulSoup(req.content, "lxml")

            for finder in finders:
                print(f"Trying {finder.__name__}")
                pdf_url = finder(req, soup)
                if pdf_url:
                    self._save_pdf_from_url(pdf_url, pdf_dir, name)
                    success = True
                    break

            if not success:
                print(
                    f"** Reprint {pmcid} could not be fetched with the current finders."
                )
                error_pmids.write(f"{pmcid}\t{name}\n")

        except requests.RequestException as e:
            print(f"** Request failed for PMCID {pmcid}: {e}")
            error_pmids.write(f"{pmcid}\t{name}\n")

    def _save_pdf_from_url(self, pdf_url: str, directory: str, name: str) -> None:
        """Save PDF from URL to directory."""
        try:
            response = self.http_session.get(pdf_url, allow_redirects=True)
            response.raise_for_status()

            if not response.content.startswith(b"%PDF"):
                content_str = response.content.decode("utf-8", errors="ignore")
                if "Preparing to download" in content_str:
                    pmc_match = re.search(r"PMC\d+", pdf_url)
                    if pmc_match:
                        pmc_id = pmc_match.group()
                        alt_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc_id}&blobtype=pdf"
                        print(f"** Trying alternative URL: {alt_url}")
                        response = self.http_session.get(alt_url, allow_redirects=True)
                        response.raise_for_status()

            with open(f"{directory}/{name}.pdf", "wb") as f:
                f.write(response.content)
            print(
                f"** Successfully fetched and saved PDF for PMCID {name}. File size: {len(response.content)} bytes"
            )

        except requests.RequestException as e:
            print(f"** Failed to download PDF from {pdf_url}: {e}")


class ImageDownloader:
    """Handles image downloading functionality."""

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
        """Extract figures from already-fetched HTML content."""
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

                if self._is_likely_protein_figure(img, src):
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


class ArticleManager:
    """Manages article storage and retrieval."""

    @staticmethod
    def save_article_to_json(
        article: ArticleMetadata, base_dir: str = "articles_data"
    ) -> str:
        """Save article data to JSON file in the PMID-specific directory."""
        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(article.pmid, base_dir)

        # Use PMID as filename
        filename = f"{article.pmid}_metadata.json"
        filepath = os.path.join(pmid_dir, filename)

        # Save article data
        with open(filepath, "w") as f:
            json.dump(article.to_dict(), f, indent=2)
        print(f"Saved article metadata to {filepath}")
        return filepath

    @staticmethod
    def save_figures_to_json(
        figures: List[Figure], pmid: str, base_dir: str = "articles_data"
    ) -> str:
        """Save figures metadata to JSON file in the PMID-specific directory.
        Only saves figures that have non-empty captions."""
        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(pmid, base_dir)

        # Prepare figures data for JSON serialization
        # Only include figures with non-empty captions
        figures_data = []
        figure_counter = 1
        for figure in figures:
            # Skip figures without captions or with empty captions
            if not figure.caption or not figure.caption.strip():
                continue

            # Extract file extension from URL or use .jpg as default
            file_ext = (
                figure.url.split(".")[-1] if "." in figure.url.split("/")[-1] else "jpg"
            )

            figures_data.append(
                {
                    "id": figure_counter,
                    "url": figure.url,
                    "alt": figure.alt,
                    "caption": figure.caption,
                    "local_filename": f"figure_{figure_counter}.{file_ext}",
                }
            )
            figure_counter += 1

        filename = f"{pmid}_figures.json"
        filepath = os.path.join(pmid_dir, filename)

        # Save figures data
        with open(filepath, "w") as f:
            json.dump(figures_data, f, indent=2)

        print(f"Saved {len(figures_data)} figures with captions to {filepath}")
        if len(figures_data) != len(figures):
            print(
                f"  Filtered out {len(figures) - len(figures_data)} figures without captions"
            )

        return filepath

    @staticmethod
    def load_pmids_from_json(json_file: str = "unique_pmids.json") -> List[str]:
        """Load list of PMIDs from a JSON file."""
        with open(json_file, "r") as f:
            pmids = json.load(f)
        print(f"Loaded {len(pmids)} PMIDs from {json_file}")
        return pmids


class PubMedClient:
    """Main client class that orchestrates all components."""

    def __init__(self, email: str, base_dir: str = "articles_data"):
        self.base_dir = base_dir
        self.http_session = HTTPSession()
        self.metadata_fetcher = MetadataFetcher(email, self.http_session)
        self.content_fetcher = ContentFetcher(email, self.http_session)
        self.pdf_downloader = PDFDownloader(self.http_session, base_dir)
        self.image_downloader = ImageDownloader(self.http_session, base_dir)
        self.article_manager = ArticleManager()

    def search_pubmed(self, query: str, retmax: int = 100) -> List[str]:
        """Search PubMed and return list of PubMed IDs."""
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        return record["IdList"]

    def fetch_details(self, id_list: List[str]) -> str:
        """Fetch article details given a list of PubMed IDs."""
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
        return handle.read()

    def fetch_summary(self, id_list: List[str]):
        """Fetch summary metadata (title, journal, date)."""
        ids = ",".join(id_list)
        handle = Entrez.esummary(db="pubmed", id=ids)
        return Entrez.read(handle)

    def get_article_metadata(self, pmid: str) -> ArticleMetadata:
        """Get article metadata."""
        return self.metadata_fetcher.get_article_metadata(pmid)

    def get_full_article(self, pmid: str) -> ArticleMetadata:
        """Get complete article data including full text or abstract."""
        # Get basic metadata first
        article = self.metadata_fetcher.get_article_metadata(pmid)
        # Get content
        return self.content_fetcher.get_full_article_content(article)

    def download_pdf_from_pmcid(self, pmcid: str, pmid: str, **kwargs) -> None:
        """Download PDF for a given PMCID."""
        self.pdf_downloader.download_pdf_from_pmcid(pmcid, pmid, **kwargs)

    def download_image(self, img_url: str, filename: str, pmid: str) -> Optional[str]:
        """Download image from URL."""
        return self.image_downloader.download_image(img_url, filename, pmid)

    def get_article_figures(self, pmid: str) -> List[Figure]:
        """Get figures for an article by PMID using cached HTML when possible."""
        article = self.get_article_metadata(pmid)
        if article.pmcid:
            # Get HTML content through ContentFetcher to use caching
            html_content = self.content_fetcher.get_html_content(article)
            if html_content:
                pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{article.pmcid}"
                return self.image_downloader.get_pmc_figures_from_html(
                    html_content, pmc_url
                )
            else:
                print(f"Could not fetch HTML for PMCID {article.pmcid}")
                return []
        else:
            print(f"No PMCID available for PMID {pmid}, cannot extract figures")
            return []

    def get_pmc_figures(self, pmc_url: str) -> List[Figure]:
        """Extract figures from PMC article (legacy method)."""
        return self.image_downloader.get_pmc_figures(pmc_url)

    def get_article_html(self, pmid: str) -> Optional[str]:
        """Extract raw HTML content from PMC article if available."""
        article = self.get_article_metadata(pmid)
        return self.content_fetcher.get_html_content(article)

    def save_article_html(
        self, html_content: str, pmid: str, base_dir: str = None
    ) -> Optional[str]:
        """Save article HTML to file in the PMID-specific directory."""
        if base_dir is None:
            base_dir = self.base_dir

        if not html_content:
            print(f"No HTML content to save for PMID {pmid}")
            return None

        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(pmid, base_dir)

        # Save HTML file
        filename = f"{pmid}_article.html"
        filepath = os.path.join(pmid_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Saved HTML content to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving HTML for PMID {pmid}: {e}")
            return None

    def save_article_to_json(
        self, article: ArticleMetadata, base_dir: str = None
    ) -> str:
        """Save article to JSON."""
        if base_dir is None:
            base_dir = self.base_dir
        return self.article_manager.save_article_to_json(article, base_dir)

    def save_figures_to_json(
        self, figures: List[Figure], pmid: str, base_dir: str = None
    ) -> str:
        """Save figures metadata to JSON."""
        if base_dir is None:
            base_dir = self.base_dir
        return self.article_manager.save_figures_to_json(figures, pmid, base_dir)

    def load_pmids_from_json(self, json_file: str = "unique_pmids.json") -> List[str]:
        """Load PMIDs from JSON file."""
        return self.article_manager.load_pmids_from_json(json_file)

    def get_cached_article(self, pmid: str) -> Optional[ArticleMetadata]:
        """Get article from cache if available."""
        return self.metadata_fetcher.get_cached_article(pmid)

    def clear_cache(self) -> None:
        """Clear the article cache."""
        self.metadata_fetcher.clear_cache()

    def _candidate_pdf_urls(self, pmcid: str) -> list[str]:
        """Generate candidate PDF URLs for a given PMCID."""
        pmcid = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        base = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}"
        return [
            f"{base}/pdf?download=1",  # compiled PDF
            f"{base}/pdf/{pmcid}.pdf",  # common fallback
            f"{base}/pdf",  # HTML wrapper (parse if reached)
        ]
