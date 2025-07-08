from pubmed_scraper import PubMedClient
from figure_scanner import scan_article_figures_for_keywords
from dotenv import load_dotenv
import os

# Updated usage example
if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    email = os.getenv(
        "EMAIL"
    )  # Ensure you have set the EMAIL variable in your .env file
    if not email:
        raise ValueError(
            "EMAIL environment variable is not set. Please set it in your .env file."
        )

    pubmed_client = PubMedClient(email=email, base_dir="articles_data")
    query = "protein AND (aggregation OR amyloid OR fibril) AND (temperature OR pH OR (environmental conditions)) AND ThT"

    # Search for articles
    # pmids = pubmed_client.search_pubmed(query, retmax=20)
    # print(f"Found {len(pmids)} articles for query '{query}':\n {pmids}")

    # pmids = pubmed_client.load_pmids_from_json(
    #     "/home/jeremy_h/projects/protein_aggregation_predictor/unique_pmids.json"
    # )

    pmids = ["38937578"]
    # "11106582"

    articles = []

    for pmid in pmids:
        print(f"Fetching details for PMID: {pmid}")

        # Get complete article data (now includes HTML content automatically)
        article = pubmed_client.get_full_article(pmid)
        articles.append(article)

        # Print article content
        print(f"Title: {article.title}")
        print(f"Content: {article.content}")  # Show first 500 chars

        # Save article metadata to JSON in the unified folder structure
        json_path = pubmed_client.save_article_to_json(article)
        print(f"Saved metadata to: {json_path}")

        # Save HTML content if available (now cached in article object)
        if article.html_content:
            html_path = pubmed_client.save_article_html(article.html_content, pmid)
            if html_path:
                print(f"Saved HTML content to: {html_path}")
        else:
            print(f"No HTML content available for PMID: {pmid}")

        # Get and download figures from the article (uses cached HTML)
        print(f"Extracting figures for PMID: {pmid}")
        figures = pubmed_client.get_article_figures(pmid)

        if figures:
            print(f"Found {len(figures)} figures")

            # Save figures metadata to JSON
            figures_json_path = pubmed_client.save_figures_to_json(figures, pmid)
            print(f"Saved figures metadata to: {figures_json_path}")

            # Scan figures for ThT/Thioflavin keywords
            print(f"\nScanning figures for relevant keywords...")
            scan_result = scan_article_figures_for_keywords(
                figures, pmid=pmid, verbose=True
            )

            # Decide whether to continue processing based on scan results
            if scan_result.has_relevant_figures:
                print(
                    f"✓ Article contains relevant figures - proceeding with download and processing"
                )
                should_process = True
            else:
                print(
                    f"✗ No relevant figures found - you may want to skip detailed processing"
                )
                should_process = False
                # Note: You can still continue processing if needed, this is just a flag

            for i, figure in enumerate(figures):
                print(f"Figure {i+1}:")
                print(f"  URL: {figure.url}")
                print(f"  Alt text: {figure.alt}")
                print(
                    f"  Caption: {figure.caption[:100]}{'...' if len(figure.caption) > 100 else ''}"
                )

                # Check if this specific figure was flagged as relevant
                figure_number = i + 1
                if figure_number in scan_result.relevant_figure_numbers:
                    keywords_found = scan_result.keyword_matches[figure_number]
                    print(
                        f"  🎯 RELEVANT: Contains keywords: {', '.join(keywords_found)}"
                    )

                # Download the image to the unified folder structure
                if (
                    figure.url and should_process
                ):  # Only download if flagged as relevant
                    # Extract file extension from URL or use .jpg as default
                    file_ext = (
                        figure.url.split(".")[-1]
                        if "." in figure.url.split("/")[-1]
                        else "jpg"
                    )
                    filename = f"figure_{i+1}.{file_ext}"

                    print(f"  Downloading figure {i+1}...")
                    downloaded_path = pubmed_client.download_image(
                        figure.url, filename, pmid
                    )

                    if downloaded_path:
                        print(f"  ✓ Successfully downloaded: {downloaded_path}")
                    else:
                        print(f"  ✗ Failed to download image")
                elif figure.url and not should_process:
                    print(f"  ⏸️ Skipping download (no relevant keywords found)")
                print()
        else:
            print("No figures found for this article")

        # =================================================================================================
        # LLM Processing Section:
        # - article.content: Contains the full text or abstract
        # - article.html_content: Contains the raw HTML from PMC (if available)
        # - Both can be used by a local LLM for processing information into a structured JSON file
        # =================================================================================================

        # Download PDF if available
        if article.pmcid:  # only try if we actually have a PMCID
            print(f"Downloading PDF for PMCID: {article.pmcid}")
            try:
                pubmed_client.download_pdf_from_pmcid(article.pmcid, pmid)
                print(f"✓ PDF download initiated for {article.pmcid}")
            except Exception as e:
                print(f"✗ Failed to download PDF for {article.pmcid}: {e}")
        else:
            print(f"{pmid} has no PMCID – skipping PDF download")

        # Print article information
        print(f"PMID: {article.pmid}")
        print(f"PMCID: {article.pmcid or 'Not available'}")
        print("-" * 80)
