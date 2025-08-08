from typing import Optional, Dict
import os
import json
import requests
from Bio import Entrez
from bs4 import BeautifulSoup
from utils import ArticleMetadata, create_pmc_url
from http_session import HTTPSession
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

        # Try Elsevier API first if publisher is Elsevier and we have DOI
        if (
            article.publisher == "Elsevier"
            and article.doi
            and self._elsevier_api_key
            and not article.content
        ):
            if not getattr(self.config, "elsevier_insttoken", None):
                print(
                    "Note: No Elsevier institutional token configured; off-network full text is unlikely."
                )
            elsevier_content = self._fetch_elsevier_fulltext(article.doi)
            if elsevier_content:
                article.content = elsevier_content
                article.source = "elsevier_api"
                return article

        # Try to get HTML content first if we have a PMCID
        if article.pmcid and not article.html_content:
            html_content = self.get_html_content(article)
            # Extract text content from HTML if we don't have content yet
            if html_content and not article.content:
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
            "httpAccept": "application/json",
            "view": "META",  # cheap probe; doesn't try to pull full text
        }
        r = requests.get(
            url, params=params, headers={"X-ELS-APIKey": api_key}, timeout=20
        )
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
        Requires entitlement for paywalled items (institutional token or recognized IP).
        """
        if not self._elsevier_api_key:
            return None

        try:
            clean_doi = doi.replace("https://doi.org/", "").replace(
                "http://dx.doi.org/", ""
            )
            print(f"Attempting to fetch full text from Elsevier for DOI: {clean_doi}")

            # Check entitlement before trying to extract text
            print("Checking Elsevier entitlement...")
            self.check_elsevier_entitlement(clean_doi, self._elsevier_api_key)

            base = "https://api.elsevier.com/content/article"
            inst_token = getattr(self.config, "elsevier_insttoken", None)

            def _req(path: str):
                headers = {
                    "X-ELS-APIKey": self._elsevier_api_key,
                    "User-Agent": "Literature Mining Tool",
                }
                if inst_token:
                    headers["X-ELS-Insttoken"] = inst_token
                # Important: pass httpAccept as a query param
                params = {"view": "FULL", "httpAccept": "application/json"}
                return self.http_session.get(
                    f"{base}/{path}", headers=headers, params=params
                )

            # 1) Try DOI endpoint first
            r = _req(f"doi/{clean_doi}")

            # If DOI endpoint returns 400, try to resolve to PII and re-request
            if r.status_code == 400:
                doi_url = f"https://doi.org/{clean_doi}"
                try:
                    head = self.http_session.head(doi_url, allow_redirects=True)
                    if head.status_code == 200 and "/pii/" in head.url.lower():
                        pii = head.url.split("/pii/")[1].split("?")[0]
                        print(f"Resolved DOI to PII: {pii}; retrying via PII endpoint.")
                        r = _req(f"pii/{pii}")
                    else:
                        print("Could not resolve PII from DOI redirect.")
                except Exception as e:
                    print(f"DOI->PII resolution error: {e}")

            # Handle common statuses
            if r.status_code == 401:
                print("Unauthorized (401). Check API key.")
                return None
            if r.status_code == 403:
                print(
                    "Forbidden (403). Likely not entitled (need inst token or on-campus IP)."
                )
                return None
            if r.status_code == 404:
                print("Not found (404) in Article API. It may not be indexed yet.")
                return None
            if r.status_code == 429:
                print("Rate limit exceeded (429).")
                return None
            if r.status_code != 200:
                print(f"Elsevier API request failed: {r.status_code}")
                return None

            data = r.json()
            ftr = data.get("full-text-retrieval-response")
            if not ftr:
                print(
                    "No full-text-retrieval-response present. Probably not entitled to full text."
                )
                return None

            parts = []

            # Title
            core = ftr.get("coredata", {}) or {}
            title = core.get("dc:title")
            if title:
                parts.append(f"Title: {title}")

            # Abstract
            abstract = core.get("dc:description")
            if abstract:
                parts.append(
                    f"Abstract: {abstract if isinstance(abstract, str) else str(abstract)}"
                )

            # Authors (defensive)
            authors = ftr.get("authors", {}).get("author")
            if authors:
                if not isinstance(authors, list):
                    authors = [authors]
                names = []
                for a in authors:
                    if isinstance(a, dict):
                        gn = a.get("ce:given-name", "")
                        sn = a.get("ce:surname", "")
                        nm = " ".join([gn, sn]).strip()
                        if nm:
                            names.append(nm)
                if names:
                    parts.append("Authors: " + ", ".join(names))

            # Try to extract actual full text
            grabbed_any_fulltext = False

            if isinstance(ftr.get("originalText"), str):
                parts.append(ftr["originalText"])
                grabbed_any_fulltext = True

            body_text = self._extract_elsevier_body_text(ftr)
            if body_text:
                parts.append(body_text)
                grabbed_any_fulltext = True

            # Optional: captions from objects
            if "objects" in ftr:
                obj_text = self._extract_elsevier_objects_text(ftr["objects"])
                if obj_text:
                    parts.append(obj_text)

            if not grabbed_any_fulltext:
                print(
                    "Entitled full text not present in response. Likely not entitled with current IP/token."
                )
                return None

            full_text = "\n\n".join(p for p in parts if p)
            print(
                f"✓ Successfully fetched Elsevier full text ({len(full_text)} characters)"
            )
            return full_text

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
