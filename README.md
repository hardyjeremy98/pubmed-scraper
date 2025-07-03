# PubMed Literature Mining Toolkit

A comprehensive Python toolkit for scraping and extracting scientific literature from PubMed and PMC (PubMed Central), specifically designed for protein aggregation research but applicable to any biomedical literature mining task.

## Features

- **🔍 PubMed Search**: Search PubMed database with complex queries
- **📄 Full-text Retrieval**: Download complete article content and metadata
- **🖼️ Figure Extraction**: Automatically extract and download figures from PMC articles
- **📁 PDF Downloads**: Download full-text PDFs when available
- **💾 Structured Storage**: Organize downloaded content in a hierarchical folder structure
- **⚡ Smart Caching**: Avoid redundant API calls with intelligent caching
- **🔄 Multiple Sources**: Fallback between Europe PMC and PubMed APIs for maximum coverage

## Project Structure

```
literature_search_utils/
├── main.py                      # Main execution script
├── pubmed_scraper.py           # Core scraping functionality
├── requirements.txt            # Python dependencies

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd literature_search_utils
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   EMAIL=your.email@example.com
   ```
   > **Note**: The email is required by NCBI's Entrez API for identification purposes.

## Quick Start

```python
from pubmed_scraper import PubMedClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
email = os.getenv("EMAIL")

# Initialize client
client = PubMedClient(email=email, base_dir="articles_data")

# Search for articles
query = "protein AND aggregation AND ThT"
pmids = client.search_pubmed(query, retmax=10)

# Process each article
for pmid in pmids:
    # Get full article data
    article = client.get_full_article(pmid)
    
    # Save metadata
    client.save_article_to_json(article)
    
    # Extract and download figures
    figures = client.get_article_figures(pmid)
    if figures:
        client.save_figures_to_json(figures, pmid)
        
        # Download each figure
        for i, figure in enumerate(figures):
            filename = f"figure_{i+1}.jpg"
            client.download_image(figure.url, filename, pmid)
    
    # Download PDF if available
    if article.pmcid:
        client.download_pdf_from_pmcid(article.pmcid, pmid)
```

## Core Components

### PubMedClient
The main orchestrator class that provides a unified interface to all functionality:

- `search_pubmed(query, retmax)`: Search PubMed with query terms
- `get_full_article(pmid)`: Get complete article with metadata and content
- `get_article_figures(pmid)`: Extract figures from PMC articles
- `download_pdf_from_pmcid(pmcid, pmid)`: Download full-text PDFs
- `save_article_to_json(article)`: Save article metadata as JSON
- `save_figures_to_json(figures, pmid)`: Save figure metadata as JSON

### Data Classes

#### ArticleMetadata
Stores comprehensive article information:
```python
@dataclass
class ArticleMetadata:
    pmid: str                    # PubMed ID
    pmcid: Optional[str]         # PMC ID (if available)
    title: str                   # Article title
    doi: str                     # Digital Object Identifier
    journal: str                 # Journal name
    source: str                  # Source database
    content: str                 # Full text or abstract
```

#### Figure
Represents extracted figures:
```python
@dataclass
class Figure:
    url: str                     # Image URL
    alt: str                     # Alt text
    caption: str                 # Figure caption
    element: Optional[any]       # Raw HTML element
```

## Advanced Usage

### Custom Search Queries

The toolkit supports complex PubMed queries:

```python
# Protein aggregation with environmental conditions
query = "protein AND (aggregation OR amyloid OR fibril) AND (temperature OR pH OR (environmental conditions)) AND ThT"

# Specific protein with methodology
query = "alpha-synuclein AND (fluorescence OR ThT) AND aggregation"

# Date-restricted search
query = "protein aggregation AND (\"2020\"[Date - Publication] : \"2024\"[Date - Publication])"
```

### Batch Processing

```python
# Load PMIDs from file
pmids = client.load_pmids_from_json("pmid_list.json")

# Process in batches
batch_size = 10
for i in range(0, len(pmids), batch_size):
    batch = pmids[i:i+batch_size]
    for pmid in batch:
        try:
            article = client.get_full_article(pmid)
            client.save_article_to_json(article)
            # Add small delay to respect API limits
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed to process {pmid}: {e}")
```

### Error Handling and Monitoring

The toolkit includes robust error handling:

- Failed downloads are logged to `unfetched_pmcids.tsv`
- Automatic fallback between different API endpoints
- Graceful handling of missing PMCIDs or PDFs
- Rate limiting to respect API guidelines

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `EMAIL` | Your email address for NCBI API identification | Yes |

### Output Structure

Downloaded content is organized as follows:

```
articles_data/
├── {pmid}/
│   ├── {pmid}_metadata.json     # Article metadata and content
│   ├── {pmid}_figures.json      # Figure metadata
│   ├── PMC{pmcid}.pdf          # Full-text PDF (when available)
│   └── images/                  # Downloaded figures
│       ├── figure_1.svg
│       ├── figure_2.jpg
│       └── ...
```

## API Rate Limits

The toolkit respects NCBI's API guidelines:

- Maximum 3 requests per second for Entrez API
- Built-in delays between requests
- Automatic retry with exponential backoff
- Caching to minimize redundant calls

## Jupyter Notebooks

### literature_search.ipynb
Interactive notebook for:
- Exploratory data analysis
- Query testing and refinement
- Visualization of search results
- Data quality assessment

### convert_pdf_to_images.ipynb
Utilities for:
- Converting PDFs to images
- Extracting specific pages
- Image preprocessing for downstream analysis

## Dependencies

- **requests**: HTTP client for API calls
- **beautifulsoup4**: HTML/XML parsing for figure extraction
- **biopython**: NCBI Entrez API interface
- **python-dotenv**: Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Missing EMAIL environment variable**:
   ```
   ValueError: EMAIL environment variable is not set
   ```
   Solution: Create a `.env` file with your email address.

2. **API rate limit errors**:
   ```
   HTTP 429: Too Many Requests
   ```
   Solution: The toolkit includes automatic rate limiting, but you may need to increase delays for high-volume processing.

3. **Missing PMCIDs**:
   ```
   No PMCID available for PMID {pmid}
   ```
   Solution: Not all articles have open-access versions in PMC. This is expected behavior.

4. **PDF download failures**:
   ```
   Failed to download PDF for PMC{pmcid}
   ```
   Solution: Some articles may have restricted access or the PDF may not be available.

### Getting Help

- Check the issue tracker for known problems
- Review NCBI's API documentation for Entrez guidelines
- Ensure your queries follow PubMed search syntax

## Acknowledgments

- NCBI for providing the PubMed and PMC APIs
- Europe PMC for additional metadata sources
- BioPython community for the Entrez interface

---

**Note**: This toolkit is designed for research purposes. Please ensure your usage complies with publisher terms of service and copyright regulations when downloading full-text content.
