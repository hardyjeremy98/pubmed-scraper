import re
import os
import time
import urllib
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from utils import ensure_pmid_directory
from http_session import HTTPSession


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
