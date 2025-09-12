from typing import Optional, Dict, List
import os
import json
import requests
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from Bio import Entrez
from bs4 import BeautifulSoup
from utils.utils import ArticleMetadata, create_pmc_url, Figure
from article_processing.http_session import HTTPSession
from config import Config


class DataFetcher:
    """Handles fetching article metadata from various sources."""

    def __init__(self, config: Config, http_session: HTTPSession):
        Entrez.email = config.email
        self.config = config
        self.http_session = http_session
        self._cache: Dict[str, ArticleMetadata] = {}
        self._html_cache: Dict[str, str] = {}

        # Store Elsevier API key for direct API calls
        self._elsevier_api_key = config.elsevier_api_key
        if self._elsevier_api_key:
            print("✓ Elsevier API key configured for direct API access")
            if config.elsevier_insttoken:
                print("✓ Elsevier institutional token configured for enhanced access")
            else:
                print(
                    "Info: No Elsevier institutional token configured. Off-network full text access may be limited."
                )
        else:
            print(
                "Info: No Elsevier API key configured. Full text access from Elsevier will be limited."
            )

    def get_article_metadata(self, pmid: str) -> ArticleMetadata:
        """
        Retrieve article metadata (PMID, PMCID, title, DOI, journal, publisher).
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

        # Get publisher from DOI if available
        if metadata.doi and not metadata.publisher:
            publisher = self.get_publisher_from_doi(metadata.doi)
            if publisher:
                metadata.publisher = publisher

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
                doi = doc.get("doi", "")
                publisher = self.get_publisher_from_doi(doi) if doi else ""

                return ArticleMetadata(
                    pmid=pmid,
                    pmcid=doc.get("pmcid"),
                    title=doc.get("title", ""),
                    doi=doi,
                    journal=doc.get("journalTitle", ""),
                    publisher=publisher,
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

    def _fetch_pubmed_record(self, pmid: str) -> Optional[dict]:
        """Fetch the raw PubMed record for a given PMID."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record = Entrez.read(handle)
            return record["PubmedArticle"][0]["MedlineCitation"]["Article"]
        except Exception as e:
            print(f"Error fetching PubMed record for PMID {pmid}: {e}")
            return None

    def _fetch_metadata_from_pubmed(self, pmid: str) -> ArticleMetadata:
        """Fetch metadata from PubMed API as fallback."""
        article_data = self._fetch_pubmed_record(pmid)
        if not article_data:
            return ArticleMetadata(pmid=pmid, publisher="")

        title = article_data.get("ArticleTitle", "")
        journal = article_data.get("Journal", {}).get("Title", "")
        doi = ""

        # Extract DOI if available
        for eloc in article_data.get("ELocationID", []):
            if hasattr(eloc, "attributes") and eloc.attributes.get("EIdType") == "doi":
                doi = str(eloc)
                break

        # Get publisher from DOI if available
        publisher = self.get_publisher_from_doi(doi) if doi else ""

        return ArticleMetadata(
            pmid=pmid, title=title, doi=doi, journal=journal, publisher=publisher
        )

    def clear_cache(self) -> None:
        """Clear both metadata and HTML caches."""
        self._cache.clear()
        self._html_cache.clear()

    def clear_metadata_cache(self) -> None:
        """Clear only the metadata cache."""
        self._cache.clear()

    def clear_html_cache(self) -> None:
        """Clear only the HTML cache."""
        self._html_cache.clear()

    def get_cached_article(self, pmid: str) -> Optional[ArticleMetadata]:
        """Get article from cache if available."""
        return self._cache.get(pmid)

    def get_full_article_content(self, article: ArticleMetadata) -> ArticleMetadata:
        """
        Get complete article data including full text, abstract, and HTML.
        """

        # Try PMC first if we have a PMCID (open access priority)
        if article.pmcid and not article.html_content:
            html_content = self.get_html_content(article)
            # Extract text content from HTML if we don't have content yet
            if html_content and not article.content:
                content = self._extract_text_from_html(html_content)
                if content:
                    article.content = content
                    article.source = "fulltext"
                    return article

        # Try Elsevier API if PMC not available and publisher is Elsevier with DOI
        if (
            article.publisher == "Elsevier"
            and article.doi
            and self._elsevier_api_key
            and not article.content
        ):
            if not self.config.elsevier_insttoken:
                print(
                    "Note: No Elsevier institutional token configured; off-network full text is unlikely."
                )
            elsevier_content = self._fetch_elsevier_fulltext(article.doi)
            if elsevier_content:
                article.content = elsevier_content
                article.source = "elsevier_api"
                return article

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

    def _fetch_abstract_from_pubmed(self, pmid: str) -> Optional[str]:
        """Fetch abstract from PubMed."""
        article_data = self._fetch_pubmed_record(pmid)
        if not article_data:
            return None

        abstract_sections = article_data.get("Abstract", {}).get("AbstractText", [])
        if abstract_sections:
            if isinstance(abstract_sections, list):
                return " ".join(str(section) for section in abstract_sections)
            else:
                return str(abstract_sections)

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

    def check_elsevier_entitlement(self, doi: str, api_key: str):
        """
        Check Elsevier entitlement for a given DOI using a lightweight META view probe.
        This is a cheap way to check if we have access before attempting full text retrieval.
        """
        url = f"https://api.elsevier.com/content/article/doi/{doi}"
        params = {
            "httpAccept": "application/xml",
            "view": "META",  # cheap probe; doesn't try to pull full text
        }
        headers = {"X-ELS-APIKey": api_key, "User-Agent": "Literature Mining Tool"}
        if self.config.elsevier_insttoken:
            headers["X-ELS-Insttoken"] = self.config.elsevier_insttoken
        r = requests.get(url, params=params, headers=headers, timeout=20)
        print("Status:", r.status_code)
        if r.status_code != 200:
            print(r.text[:500])
            return

        data = r.json()
        ftr = data.get("full-text-retrieval-response", {})
        core = ftr.get("coredata", {})
        links = core.get("link", [])
        if isinstance(links, dict):
            links = [links]

        # Look for entitlement-ish rels
        rels = [l.get("@rel") for l in links if isinstance(l, dict)]
        print("link rels:", rels)

        # A very rough signal: presence of self/entitled/full-text-ish rels
        entitled = any("entitled" in (rel or "") for rel in rels)
        print("Entitled (heuristic):", entitled)

        # Extra: echo what full-text link (if any) looks like
        for l in links:
            if isinstance(l, dict) and "full" in (l.get("@rel") or "").lower():
                print("Full link example:", l.get("@href"))
                break

    def _fetch_elsevier_fulltext(self, doi: str) -> Optional[str]:
        """
        Fetch full text content from Elsevier using their Article Retrieval API.
        Uses XML parsing to extract complete article content including figures.
        """
        if not self._elsevier_api_key:
            return None

        try:
            clean_doi = doi.replace("https://doi.org/", "").replace(
                "http://dx.doi.org/", ""
            )
            print(f"Fetching Elsevier XML content for DOI: {clean_doi}")

            # Make the API call for XML content
            url = f"https://api.elsevier.com/content/article/doi/{clean_doi}"
            headers = {
                "X-ELS-APIKey": self._elsevier_api_key,
                "Accept": "application/xml",
            }
            if self.config.elsevier_insttoken:
                headers["X-ELS-Insttoken"] = self.config.elsevier_insttoken

            params = {"httpAccept": "application/xml"}

            response = self.http_session.get(
                url, headers=headers, params=params, timeout=60
            )

            # DEBUG: Check raw response size like notebook
            print(f"Raw response size: {len(response.text)} characters")

            if response.status_code == 401:
                print("Unauthorized (401). Check API key.")
                return None
            if response.status_code == 403:
                print(
                    "Forbidden (403). Likely not entitled (need inst token or on-campus IP)."
                )
                return None
            if response.status_code == 404:
                print("Not found (404) in Article API.")
                return None
            if response.status_code == 429:
                print("Rate limit exceeded (429).")
                return None
            if response.status_code != 200:
                print(f"Elsevier API request failed: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return None

            # Parse the XML response
            root = ET.fromstring(response.text)

            # Define namespace mapping for Elsevier XML
            namespaces = {
                "dc": "http://purl.org/dc/elements/1.1/",
                "prism": "http://prismstandard.org/namespaces/basic/2.0/",
                "ce": "http://www.elsevier.com/xml/common/dtd",
                "sb": "http://www.elsevier.com/xml/common/struct-bib/dtd",
                "xlink": "http://www.w3.org/1999/xlink",
            }

            # Extract article components
            article_parts = []

            # Extract title
            title_elem = root.find(".//dc:title", namespaces)
            if title_elem is not None and title_elem.text:
                article_parts.append(f"TITLE: {title_elem.text}")

            # Extract authors
            author_elems = root.findall(".//dc:creator", namespaces)
            if author_elems:
                authors = []
                for author in author_elems:
                    if author.text:
                        authors.append(author.text)
                if authors:
                    article_parts.append(f"AUTHORS: {', '.join(authors)}")

            # Extract journal info
            journal_elem = root.find(".//prism:publicationName", namespaces)
            if journal_elem is not None and journal_elem.text:
                article_parts.append(f"JOURNAL: {journal_elem.text}")

            # Extract publication date
            date_elem = root.find(".//prism:coverDate", namespaces)
            if date_elem is not None and date_elem.text:
                article_parts.append(f"DATE: {date_elem.text}")

            # Extract DOI
            doi_elem = root.find(".//prism:doi", namespaces)
            if doi_elem is not None and doi_elem.text:
                article_parts.append(f"DOI: {doi_elem.text}")

            # Extract abstract
            abstract_elem = root.find(".//dc:description", namespaces)
            if abstract_elem is not None and abstract_elem.text:
                article_parts.append(f"ABSTRACT:\n{abstract_elem.text}")

            # Extract keywords
            keyword_elems = root.findall(".//ce:keyword", namespaces)
            if keyword_elems:
                keywords = []
                for keyword in keyword_elems:
                    if keyword.text:
                        keywords.append(keyword.text)
                if keywords:
                    article_parts.append(f"KEYWORDS: {', '.join(keywords)}")

            # COMPREHENSIVE text extraction - capture ALL text like the notebook
            article_parts = []

            # Extract title (like notebook)
            title_elem = root.find(".//dc:title", namespaces)
            if title_elem is not None and title_elem.text:
                article_parts.append(f"TITLE: {title_elem.text}")

            # Extract authors (like notebook)
            article_parts.append("\nAUTHORS:")
            author_elems = root.findall(".//dc:creator", namespaces)
            for i, author in enumerate(author_elems, 1):
                if author.text:
                    article_parts.append(f"{i}. {author.text}")

            # Extract journal info (like notebook)
            journal_elem = root.find(".//prism:publicationName", namespaces)
            if journal_elem is not None and journal_elem.text:
                article_parts.append(f"\nJOURNAL: {journal_elem.text}")

            # Extract publication date (like notebook)
            date_elem = root.find(".//prism:coverDate", namespaces)
            if date_elem is not None and date_elem.text:
                article_parts.append(f"DATE: {date_elem.text}")

            # Extract DOI (like notebook)
            doi_elem = root.find(".//prism:doi", namespaces)
            if doi_elem is not None and doi_elem.text:
                article_parts.append(f"DOI: {doi_elem.text}")

            # Extract abstract (like notebook)
            article_parts.append("\nABSTRACT:")
            abstract_elem = root.find(".//dc:description", namespaces)
            if abstract_elem is not None and abstract_elem.text:
                article_parts.append(abstract_elem.text)

            # Extract keywords (like notebook)
            article_parts.append("\nKEYWORDS:")
            keyword_elems = root.findall(".//ce:keyword", namespaces)
            for keyword in keyword_elems:
                if keyword.text:
                    article_parts.append(f"- {keyword.text}")

            # Extract full article text - EXACTLY like the notebook does it
            article_parts.append("\n" + "=" * 80)
            article_parts.append("FULL ARTICLE TEXT:")
            article_parts.append("=" * 80)

            # CRITICAL: Extract from originalText section - this contains the full article content
            # Try comprehensive extraction from the originalText section

            # Find originalText element (no namespace prefix)
            original_text = root.find(".//originalText")
            if original_text is None:
                # Fallback: search for any element ending with 'originalText'
                for elem in root.iter():
                    if elem.tag.endswith("originalText"):
                        original_text = elem
                        break

            if original_text is not None:
                # Extract all text from the originalText section using itertext()
                # This captures ALL text content including nested elements
                original_text_parts = []
                for text in original_text.itertext():
                    if text.strip():
                        original_text_parts.append(text.strip())

                if original_text_parts:
                    print(
                        f"✓ Extracted {len(original_text_parts)} text parts from originalText section"
                    )
                    article_parts.extend(original_text_parts)
                else:
                    print(f"⚠ originalText found but no content extracted")
            else:
                print(f"⚠ originalText section not found in XML")

            # Look for sections and paragraphs (EXACT notebook logic) - as fallback
            sections = root.findall(".//ce:section", namespaces)
            if sections:
                for section in sections:
                    # Section title
                    section_title = section.find(".//ce:section-title", namespaces)
                    if section_title is not None and section_title.text:
                        article_parts.append(f"\n=== {section_title.text.upper()} ===")

                    # Section paragraphs - EXACT notebook approach
                    paragraphs = section.findall(".//ce:para", namespaces)
                    for para in paragraphs:
                        # Start with paragraph's direct text (like notebook)
                        if para.text:
                            article_parts.append(f"\n{para.text}")

                        # Also check for text in sub-elements (EXACT notebook approach)
                        for elem in para.iter():
                            if elem.text and elem.text.strip() and elem != para:
                                article_parts.append(elem.text.strip())
                        # Note: notebook uses print() with end=" " then print() - simulating that

            # If no sections found, look for any paragraphs (EXACT notebook logic)
            if not sections:
                paragraphs = root.findall(".//ce:para", namespaces)
                for para in paragraphs:
                    text_content = ""
                    if para.text:
                        text_content += para.text

                    # Collect text from all sub-elements (EXACT notebook approach)
                    for elem in para.iter():
                        if elem.text and elem.text.strip():
                            text_content += elem.text
                        if elem.tail and elem.tail.strip():
                            text_content += elem.tail

                    if text_content.strip():
                        article_parts.append(f"\n{text_content.strip()}")

            # Add the closing separator like notebook
            article_parts.append("\n" + "=" * 80)

            full_text = "\n".join(article_parts)

            if full_text.strip():
                print(
                    f"✓ Successfully fetched Elsevier XML content ({len(full_text)} characters)"
                )
                return full_text
            else:
                print("No content extracted from XML")
                return None

        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return None
        except Exception as e:
            print(f"Error fetching from Elsevier API for DOI {doi}: {e}")
            return None

    def _extract_elsevier_body_text(self, content: dict) -> Optional[str]:
        """
        Extract body text from Elsevier API response.

        Args:
            content: The content from Elsevier API response

        Returns:
            Extracted body text
        """
        try:
            text_parts = []

            # Look for different possible content structures in Elsevier API
            if "sections" in content:
                sections = content["sections"]
                if isinstance(sections, dict) and "section" in sections:
                    sections_list = sections["section"]
                    if not isinstance(sections_list, list):
                        sections_list = [sections_list]

                    for section in sections_list:
                        if isinstance(section, dict):
                            # Section title
                            if "ce:section-title" in section:
                                text_parts.append(f"\n{section['ce:section-title']}\n")

                            # Paragraphs
                            if "ce:para" in section:
                                paras = section["ce:para"]
                                if not isinstance(paras, list):
                                    paras = [paras]
                                for para in paras:
                                    if isinstance(para, str):
                                        text_parts.append(para)
                                    elif isinstance(para, dict):
                                        if "#text" in para:
                                            text_parts.append(para["#text"])
                                        elif "ce:para" in para:
                                            text_parts.append(str(para["ce:para"]))

            if "body" in content:
                body_content = content["body"]
                if isinstance(body_content, str):
                    text_parts.append(body_content)
                elif isinstance(body_content, dict):
                    body_text = self._extract_text_from_elsevier_body(body_content)
                    if body_text:
                        text_parts.append(body_text)

            return "\n".join(text_parts) if text_parts else None

        except Exception as e:
            print(f"Error extracting body text from Elsevier response: {e}")
            return None

    def _extract_elsevier_objects_text(self, objects: dict) -> Optional[str]:
        """
        Extract text from objects section of Elsevier API response.

        Args:
            objects: The objects section from API response

        Returns:
            Extracted text from objects
        """
        try:
            text_parts = []

            if isinstance(objects, dict):
                for obj_type, obj_content in objects.items():
                    if isinstance(obj_content, list):
                        for item in obj_content:
                            if isinstance(item, dict) and "ce:caption" in item:
                                text_parts.append(f"Caption: {item['ce:caption']}")
                    elif isinstance(obj_content, dict) and "ce:caption" in obj_content:
                        text_parts.append(f"Caption: {obj_content['ce:caption']}")

            return "\n".join(text_parts) if text_parts else None

        except Exception as e:
            print(f"Error extracting objects text: {e}")
            return None

    def _extract_text_from_elsevier_body(self, body_data) -> Optional[str]:
        """
        Extract text content from Elsevier API body structure.

        Args:
            body_data: The body data from Elsevier API response

        Returns:
            Extracted text content
        """
        try:
            text_parts = []

            if isinstance(body_data, dict):
                if "section" in body_data:
                    sections = body_data["section"]
                    if not isinstance(sections, list):
                        sections = [sections]

                    for section in sections:
                        if isinstance(section, dict):
                            if "ce:section-title" in section:
                                text_parts.append(f"\n{section['ce:section-title']}\n")

                            if "ce:para" in section:
                                paras = section["ce:para"]
                                if not isinstance(paras, list):
                                    paras = [paras]

                                for para in paras:
                                    if isinstance(para, str):
                                        text_parts.append(para)
                                    elif isinstance(para, dict) and "#text" in para:
                                        text_parts.append(para["#text"])

                if not text_parts and "#text" in body_data:
                    text_parts.append(body_data["#text"])

            elif isinstance(body_data, str):
                text_parts.append(body_data)

            return "\n".join(text_parts) if text_parts else None

        except Exception as e:
            print(f"Error extracting text from Elsevier body: {e}")
            return None

    def get_elsevier_figures(self, doi: str) -> List[Figure]:
        """
        Extract figures from Elsevier XML response for a given DOI.
        Returns a list of Figure objects with extracted metadata and URLs.
        """
        if not self._elsevier_api_key:
            return []

        try:
            clean_doi = doi.replace("https://doi.org/", "").replace(
                "http://dx.doi.org/", ""
            )
            print(f"Extracting figures from Elsevier XML for DOI: {clean_doi}")

            # Make the API call for XML content
            url = f"https://api.elsevier.com/content/article/doi/{clean_doi}"
            headers = {
                "X-ELS-APIKey": self._elsevier_api_key,
                "Accept": "application/xml",
            }
            if self.config.elsevier_insttoken:
                headers["X-ELS-Insttoken"] = self.config.elsevier_insttoken

            params = {"httpAccept": "application/xml"}

            response = self.http_session.get(
                url, headers=headers, params=params, timeout=60
            )

            if response.status_code != 200:
                print(f"Failed to fetch XML for figures: {response.status_code}")
                return []

            # Parse the XML response
            root = ET.fromstring(response.text)

            # Define namespace mapping for Elsevier XML
            namespaces = {
                "dc": "http://purl.org/dc/elements/1.1/",
                "prism": "http://prismstandard.org/namespaces/basic/2.0/",
                "ce": "http://www.elsevier.com/xml/common/dtd",
                "sb": "http://www.elsevier.com/xml/common/struct-bib/dtd",
                "xlink": "http://www.w3.org/1999/xlink",
            }

            # Get PII for constructing image URLs
            pii = self._extract_pii_from_xml_or_doi(root, clean_doi)

            # Find all figure elements
            figures = []
            figure_elements = root.findall(".//ce:figure", namespaces)

            # Process figures sequentially, but keep track of potential graphical abstracts
            normal_figures = []
            graphical_abstract = None

            for fig in figure_elements:
                # Get figure ID
                fig_id = fig.get("id", "")
                is_graphical_abstract = False

                # First check for label to determine if this is a real figure with a number
                is_numbered_figure = False
                label_elem = fig.find(".//ce:label", namespaces)
                if label_elem is not None and label_elem.text:
                    label_text = label_elem.text.strip()
                    # Check if it's a numbered figure (like "Fig. 1" or "Figure 2")
                    if re.search(
                        r"fig\.?\s*\d+|figure\s*\d+", label_text, re.IGNORECASE
                    ):
                        is_numbered_figure = True
                    # Check if it's explicitly labeled as graphical abstract
                    if "abstract" in label_text.lower():
                        is_graphical_abstract = True

                # Extract caption to help determine if it's a graphical abstract
                caption_elem = fig.find(".//ce:caption", namespaces)
                caption = ""
                if caption_elem is not None:
                    for elem in caption_elem.iter():
                        if elem.text:
                            caption += elem.text
                        if elem.tail:
                            caption += elem.tail
                    caption = caption.strip()

                # If caption contains "graphical abstract", mark as graphical abstract
                if caption and "graphical abstract" in caption.lower():
                    is_graphical_abstract = True

                # Only use ID-based detection as a fallback if we haven't determined it's a numbered figure
                if not is_numbered_figure and not is_graphical_abstract:
                    # Check ID patterns typical of graphical abstracts
                    if fig_id.lower() == "ga" or fig_id.startswith("ga"):
                        is_graphical_abstract = True

                # If no caption, use label to create one
                if not caption and label_elem is not None and label_elem.text:
                    caption = f"Figure {label_elem.text.strip()}"

                # If still no caption, use alt-text
                if not caption:
                    alt_elem = fig.find(".//ce:alt-text", namespaces)
                    if alt_elem is not None and alt_elem.text:
                        caption = alt_elem.text.strip()
                    else:
                        caption = "No caption available"

                # Store this figure data in appropriate list
                figure_data = {
                    "element": fig,
                    "id": fig_id,
                    "caption": caption,
                    "is_graphical_abstract": is_graphical_abstract,
                }

                if is_graphical_abstract:
                    graphical_abstract = figure_data
                else:
                    normal_figures.append(figure_data)

            # Now create Figure objects with sequential numbering for normal figures
            for i, fig_data in enumerate(normal_figures, 1):
                fig_number = str(i)
                ref_id = f"gr{i}"

                # Construct the image URL in PII format
                if pii:
                    image_url = f"pii:{pii}/{ref_id}"
                else:
                    image_url = f"doi:{clean_doi}/{ref_id}"

                # Create Figure object
                figure = Figure(
                    url=image_url,
                    alt=f"Figure {fig_number}",
                    caption=fig_data["caption"],
                    element=None,  # We don't need the XML element stored
                )

                figures.append(figure)
                print(
                    f"  Found Figure {fig_number} (ID: {fig_data['id']}, Ref: {ref_id}): {fig_data['caption'][:100]}..."
                )

            # Add graphical abstract at the end if present
            if graphical_abstract:
                if pii:
                    image_url = f"pii:{pii}/ga1"
                else:
                    image_url = f"doi:{clean_doi}/ga1"

                figure = Figure(
                    url=image_url,
                    alt="Graphical Abstract",
                    caption=graphical_abstract["caption"],
                    element=None,
                )

                figures.append(figure)
                print(
                    f"  Found Graphical Abstract (ID: {graphical_abstract['id']}): {graphical_abstract['caption'][:100]}..."
                )

            print(f"✓ Extracted {len(figures)} figures from Elsevier XML")
            return figures

        except ET.ParseError as e:
            print(f"XML parsing error while extracting figures: {e}")
            return []
        except Exception as e:
            print(f"Error extracting figures from Elsevier API for DOI {doi}: {e}")
            return []

    def _extract_pii_from_xml_or_doi(
        self, root: ET.Element, clean_doi: str
    ) -> Optional[str]:
        """Extract PII from XML or derive from DOI for image URL construction."""
        try:
            # Try to extract PII from XML
            namespaces = {
                "prism": "http://prismstandard.org/namespaces/basic/2.0/",
                "ce": "http://www.elsevier.com/xml/common/dtd",
            }

            # Look for PII in various places in the XML
            pii_elem = root.find(".//prism:pii", namespaces)
            if pii_elem is not None and pii_elem.text:
                return pii_elem.text

            # Also try the ce namespace
            pii_elem = root.find(".//ce:pii", namespaces)
            if pii_elem is not None and pii_elem.text:
                return pii_elem.text

            # Try to extract from aggregationType attribute or similar
            for elem in root.iter():
                if "pii" in str(elem.attrib).lower():
                    for attr_name, attr_value in elem.attrib.items():
                        if "pii" in attr_name.lower() and attr_value:
                            return attr_value

            # If not found in XML, try to resolve PII from DOI redirect
            doi_url = f"https://doi.org/{clean_doi}"
            try:
                head_response = self.http_session.head(doi_url, allow_redirects=True)
                if (
                    head_response.status_code == 200
                    and "/pii/" in head_response.url.lower()
                ):
                    pii = (
                        head_response.url.split("/pii/")[1].split("?")[0].split("/")[0]
                    )
                    return pii
            except Exception as e:
                print(f"Failed to resolve PII from DOI redirect: {e}")

            # Fallback: try a different approach by making a test API call to get PII
            try:
                test_url = f"https://api.elsevier.com/content/article/doi/{clean_doi}"
                headers = {
                    "X-ELS-APIKey": self._elsevier_api_key,
                    "Accept": "application/json",
                }
                if self.config.elsevier_insttoken:
                    headers["X-ELS-Insttoken"] = self.config.elsevier_insttoken

                params = {"view": "META"}
                response = self.http_session.get(
                    test_url, headers=headers, params=params, timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    ftr = data.get("full-text-retrieval-response", {})
                    core = ftr.get("coredata", {})

                    # Look for PII in the response
                    pii = core.get("pii")
                    if pii:
                        return pii

                    # Also check if it's in aggregationType or elsewhere
                    for key, value in core.items():
                        if "pii" in key.lower() and value:
                            return value

            except Exception as e:
                print(f"Failed to get PII from META view: {e}")

            print(f"Could not extract PII for DOI {clean_doi}")
            return None

        except Exception as e:
            print(f"Error extracting PII: {e}")
            return None

    def download_elsevier_image(
        self, image_url: str, save_path: str, view: str = "high"
    ) -> bool:
        """
        Download an Elsevier image using the Object API.

        Args:
            image_url: URL in format "pii:PII/ref" or "doi:DOI/ref"
            save_path: Path where to save the downloaded image
            view: Image quality ("thumbnail", "standard", "high")

        Returns:
            True if download successful, False otherwise
        """
        if not self._elsevier_api_key:
            print("No Elsevier API key available for image download")
            return False

        try:
            # Parse the image URL
            if image_url.startswith("pii:"):
                # Format: pii:S0301462225001140/gr1
                parts = image_url[4:].split("/")  # Remove "pii:" prefix
                if len(parts) >= 2:
                    pii = parts[0]
                    ref = parts[1]
                    url_path = f"pii/{pii}/ref/{ref}"
                else:
                    print(f"Invalid PII format: {image_url}")
                    return False
            elif image_url.startswith("doi:"):
                # Format: doi:10.1016/j.bpc.2025.107502/gr1
                parts = image_url[4:].split("/")  # Remove "doi:" prefix
                if len(parts) >= 4:  # doi:10.1016/j.bpc.2025.107502/gr1
                    doi_part = "/".join(parts[:-1])  # Reconstruct DOI
                    ref = parts[-1]
                    url_path = f"doi/{doi_part}/ref/{ref}"
                else:
                    print(f"Invalid DOI format: {image_url}")
                    return False
            else:
                print(f"Unsupported image URL format: {image_url}")
                return False

            # Construct the API request
            base_url = "https://api.elsevier.com/content/object"
            full_url = f"{base_url}/{url_path}"

            headers = {"X-ELS-APIKey": self._elsevier_api_key, "Accept": "*/*"}
            if self.config.elsevier_insttoken:
                headers["X-ELS-Insttoken"] = self.config.elsevier_insttoken

            params = {"httpAccept": "image/jpeg", "view": view}

            print(f"Downloading image from: {full_url}")
            response = self.http_session.get(
                full_url, headers=headers, params=params, timeout=60
            )

            if response.status_code == 200 and response.headers.get(
                "Content-Type", ""
            ).startswith("image/"):
                # Determine file extension from content type
                content_type = response.headers.get("Content-Type", "image/jpeg")
                ext = content_type.split("/")[-1]
                if ext == "jpeg":
                    ext = "jpg"  # Standardize jpeg to jpg

                # Ensure save path has correct extension, removing any query parameters
                save_path = Path(save_path)
                if save_path.suffix:
                    # Remove any query parameters from existing filename
                    clean_name = save_path.stem.split("?")[0]
                    save_path = save_path.parent / f"{clean_name}.{ext}"
                else:
                    save_path = save_path.with_suffix(f".{ext}")

                # Create directory if it doesn't exist
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the image
                with open(save_path, "wb") as f:
                    f.write(response.content)

                print(f"✓ Saved image to {save_path}")
                return True
            else:
                print(f"Failed to download image: HTTP {response.status_code}")
                if response.text:
                    print(f"Response: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"Error downloading Elsevier image {image_url}: {e}")
            return False

    def get_publisher_from_doi(self, doi: str) -> Optional[str]:
        """
        Determine the publisher/hosting platform from a DOI.

        Args:
            doi: The DOI string (with or without 'https://doi.org/' prefix)

        Returns:
            Publisher name or hosting platform, or None if not determinable
        """
        if not doi:
            return None

        # Clean the DOI - remove URL prefix if present
        clean_doi = doi.replace("https://doi.org/", "").replace(
            "http://dx.doi.org/", ""
        )

        # Common DOI prefix to publisher mappings
        publisher_mappings = {
            "10.1038": "Nature Publishing Group",
            "10.1016": "Elsevier",
            "10.1021": "American Chemical Society",
            "10.1371": "PLOS",
            "10.1073": "Proceedings of the National Academy of Sciences",
            "10.1126": "Science/AAAS",
            "10.1186": "BMC/BioMed Central",
            "10.1083": "Rockefeller University Press",
            "10.1091": "American Society for Cell Biology",
            "10.1042": "Portland Press/Biochemical Society",
            "10.1074": "American Society for Biochemistry and Molecular Biology",
            "10.1128": "American Society for Microbiology",
            "10.1242": "Company of Biologists",
            "10.1002": "Wiley",
            "10.1080": "Taylor & Francis",
            "10.1007": "Springer",
            "10.1093": "Oxford University Press",
            "10.1017": "Cambridge University Press",
            "10.3389": "Frontiers",
            "10.1172": "American Society for Clinical Investigation",
            "10.1096": "Federation of American Societies for Experimental Biology",
            "10.4161": "Taylor & Francis (Landes Bioscience)",
            "10.15252": "EMBO Press",
            "10.1101": "Cold Spring Harbor Laboratory Press",
            "10.1155": "Hindawi",
            "10.3390": "MDPI",
            "10.1177": "SAGE Publications",
            "10.1089": "Mary Ann Liebert",
            "10.1098": "Royal Society Publishing",
            "10.1113": "The Physiological Society",
            "10.1152": "American Physiological Society",
        }

        # Check for exact matches first
        for prefix, publisher in publisher_mappings.items():
            if clean_doi.startswith(prefix):
                return publisher

        # For more detailed analysis, try to resolve the DOI
        try:
            doi_url = f"https://doi.org/{clean_doi}"
            response = self.http_session.head(doi_url, allow_redirects=True)

            if response.status_code == 200:
                final_url = response.url.lower()

                # Check the final redirected URL for publisher domains
                domain_mappings = {
                    "nature.com": "Nature Publishing Group",
                    "sciencedirect.com": "Elsevier",
                    "pubs.acs.org": "American Chemical Society",
                    "journals.plos.org": "PLOS",
                    "pnas.org": "Proceedings of the National Academy of Sciences",
                    "science.org": "Science/AAAS",
                    "biomedcentral.com": "BMC/BioMed Central",
                    "rupress.org": "Rockefeller University Press",
                    "molbiolcell.org": "American Society for Cell Biology",
                    "portlandpress.com": "Portland Press",
                    "jbc.org": "American Society for Biochemistry and Molecular Biology",
                    "asm.org": "American Society for Microbiology",
                    "biologists.org": "Company of Biologists",
                    "onlinelibrary.wiley.com": "Wiley",
                    "tandfonline.com": "Taylor & Francis",
                    "link.springer.com": "Springer",
                    "academic.oup.com": "Oxford University Press",
                    "cambridge.org": "Cambridge University Press",
                    "frontiersin.org": "Frontiers",
                    "jci.org": "American Society for Clinical Investigation",
                    "fasebj.org": "Federation of American Societies for Experimental Biology",
                    "embopress.org": "EMBO Press",
                    "cshlp.org": "Cold Spring Harbor Laboratory Press",
                    "hindawi.com": "Hindawi",
                    "mdpi.com": "MDPI",
                    "sagepub.com": "SAGE Publications",
                    "liebertpub.com": "Mary Ann Liebert",
                    "royalsocietypublishing.org": "Royal Society Publishing",
                    "physiology.org": "American Physiological Society",
                }

                for domain, publisher in domain_mappings.items():
                    if domain in final_url:
                        return publisher

        except Exception as e:
            print(f"Error resolving DOI {doi}: {e}")

        return None
