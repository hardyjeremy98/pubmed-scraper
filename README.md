# PubMed Literature Mining & ThT Data Extraction Toolkit

A comprehensive Python toolkit for scraping scientific literature from PubMed/PMC and extracting Thioflavin T (ThT) fluorescence assay data, specifically designed for protein aggregation research. The toolkit provides end-to-end functionality from literature search to structured experimental data extraction.

## 🔥 Key Features

### Literature Mining
- **🔍 PubMed Search**: Search PubMed database with complex queries
- **📄 Full-text Retrieval**: Download complete article content and metadata
- **🖼️ Figure Extraction**: Automatically extract and download figures from PMC articles
- **📁 PDF Downloads**: Download full-text PDFs when available

### Advanced Data Extraction
- **🧪 ThT Plot Detection**: AI-powered identification of Thioflavin T fluorescence plots
- **🎯 Figure Segmentation**: Automatic segmentation of multi-panel figures
- **📊 Plot Digitization**: Integration with external plot digitization tools
- **🔗 Data Matching**: LLM-powered matching of digitized data to experimental conditions
- **📈 Data Consolidation**: Export to structured CSV for analysis

### Smart Processing
- **💾 Structured Storage**: Hierarchical folder organization
- **⚡ Smart Caching**: Avoid redundant API calls
- **🔄 Multiple Sources**: Fallback between APIs for maximum coverage
- **🤖 LLM Integration**: GPT-4 powered experimental condition extraction

## 🚀 Complete Workflow

The toolkit implements a complete 3-step workflow for extracting ThT fluorescence data from scientific literature:

### Step 1: Literature Mining & Figure Extraction
**Script: `step1_extract.py`**

1. **PubMed Search**: Query PubMed for relevant articles
2. **Content Download**: Retrieve full-text articles and metadata
3. **Figure Extraction**: Download all figures from PMC articles
4. **ThT Plot Detection**: Use AI to identify ThT fluorescence vs time plots
5. **Figure Segmentation**: Automatically segment multi-panel figures into individual plots
6. **Experimental Condition Extraction**: Use LLM to extract experimental parameters

### Step 2: External Plot Digitization
**External Tool Integration**

- Segmented plot images are processed by external plot digitization software
- Each ThT plot is converted to CSV format with Time and Fluorescence columns
- CSVs are saved in `converted_tables/` directory with naming convention: `{pmid}_{figure}_{plot}.csv`

### Step 3: Data Processing & Consolidation
**Script: `step2_process.py`**

1. **CSV Cleaning**: Process and reformat digitized CSV files
2. **Data Matching**: Use LLM to match CSV columns to experimental conditions
3. **Data Integration**: Combine experimental metadata with time-series data  
4. **Quality Control**: Validate matches and report statistics
5. **Data Export**: Generate consolidated CSV with all experimental data

## 📁 Project Structure

```
├── step1_extract.py             # Literature mining and figure processing
├── step2_process.py             # Data matching and consolidation
├── pubmed_scraper.py           # Core scraping functionality
├── llm_data_extractor.py       # LLM integration for data extraction
├── llm_input_prep.py           # LLM input preparation utilities
├── figure_scanner.py           # ThT plot detection and segmentation
├── image_segmenter.py          # Figure segmentation utilities
├── utils.py                    # Data processing and consolidation utilities
├── plot_detector1.pt           # Trained YOLO model for ThT plot detection
├── requirements.txt            # Python dependencies
└── articles_data/              # Article storage
    ├── {pmid}/
    │   ├── {pmid}_metadata.json
    │   ├── {pmid}_figures.json
    │   ├── experimental_conditions/
    │   │   └── figure{n}.json   # Extracted experimental conditions
    │   ├── images/             # Downloaded figures
    │   ├── segmented_images/   # Individual plot segments
    │   └── bbox/               # Bounding box data
    ├── converted_tables/       # Digitized CSV files from external tool
    ├── cleaned_tables/         # Processed CSV files
    ├── final_data/             # Matched experimental data (JSON)
    └── consolidated_experimental_data.csv  # Final consolidated dataset
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd literature_mining/main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   EMAIL=your.email@example.com
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   > **Note**: 
   > - Email is required by NCBI's Entrez API
   > - OpenAI API key is required for LLM-powered data extraction

4. **Download YOLO model for plot detection**:
   The toolkit requires a trained YOLO model (`plot_detector1.pt`) for AI-powered ThT plot identification. This model should be placed in the project root directory.
   
   > **Important**: The YOLO model is essential for automatic plot detection in Step 1. Without it, the figure processing pipeline will not function correctly.

## 📊 Output Data Structure

The final consolidated CSV contains one row per experimental condition:

| Column | Description | Example |
|--------|-------------|---------|
| PMID | PubMed ID | 19258323 |
| Figure | Figure number | 1 |
| Plot | Plot number | 1 |
| Legend_Symbol | Legend symbol from plot | • |
| Time | Time points as list | `[0.0, 5.0, 10.0, ...]` |
| Fluorescence | Fluorescence values as list | `[0.0, 2.0, 21.0, ...]` |
| Protein | Protein studied | Ure2p |
| Mutation | Protein variant | WT |
| Temperature | Experimental temperature | 8 °C |
| pH | Buffer pH | 7.4 |
| Additives | Chemical additives | H2O2 |
| ... | Other experimental parameters | ... |

## 🔧 Core Components

### Step 1: Literature Mining (`step1_extract.py`)
**Main Functions:**
- `search_and_download_articles()`: PubMed search and content retrieval
- `process_figures_for_tht_plots()`: ThT plot detection and segmentation  
- `extract_experimental_conditions()`: LLM-powered condition extraction

### Step 2: Data Processing (`step2_process.py`)
**Main Functions:**
- `process_all_csv_files()`: Clean and reformat digitized CSV files
- `find_matching_csv_files()`: Match CSV data to experimental conditions
- `consolidate_final_data_to_csv()`: Generate final consolidated dataset

### LLM Integration (`llm_data_extractor.py`)
**Analysis Types:**
- `ThT_plot_identifier`: Identify ThT plots in figures
- `ThT_data_extractor`: Extract experimental conditions
- `match_maker`: Match CSV columns to legend symbols

### Figure Processing (`figure_scanner.py`, `image_segmenter.py`)
**Capabilities:**
- AI-powered plot detection using trained YOLO models (`plot_detector1.pt`)
- Automatic figure segmentation with bounding box detection
- Multi-panel figure handling
- Image preprocessing and optimization

**Required Model:**
- **YOLO Model**: `plot_detector1.pt` - A trained YOLOv8 model specifically designed to detect ThT fluorescence plots in scientific figures
- **Model Location**: Must be placed in the project root directory
- **Training Data**: Model trained on scientific literature figures containing ThT fluorescence vs time plots

### Data Processing (`utils.py`)
**Key Functions:**
- `map_csv_data_to_conditions()`: Match digitized data to conditions
- `consolidate_final_data_to_csv()`: Create final consolidated dataset
- `reformat_table()`: Convert between long/wide data formats

## 💡 Advanced Usage

### Custom Search Queries

```python
# Protein aggregation with environmental conditions
query = "protein AND (aggregation OR amyloid OR fibril) AND (temperature OR pH) AND ThT"

# Specific protein studies
query = "alpha-synuclein AND ThT AND aggregation AND kinetics"

# Date-restricted search
query = "protein aggregation ThT AND 2020:2024[dp]"
```

### Processing Specific Articles

```python
# Step 1: Process specific PMIDs
from step1_extract import process_specific_pmids
pmids = ["19258323", "18350169"]  # Your PMIDs
process_specific_pmids(pmids)

# Step 2: Process only certain files
from step2_process import find_matching_csv_files
find_matching_csv_files("articles_data", "converted_tables")
```

### Working with the Consolidated Data

```python
import pandas as pd
import ast

# Load the final dataset
df = pd.read_csv("consolidated_experimental_data.csv")

# Convert string lists back to actual lists
for idx, row in df.iterrows():
    time_data = ast.literal_eval(row['Time'])
    fluorescence_data = ast.literal_eval(row['Fluorescence'])
    
    # Now you can analyze the time series data
    print(f"Condition: {row['Protein']} {row['Mutation']}")
    print(f"Data points: {len(time_data)}")
    print(f"Max fluorescence: {max(fluorescence_data)}")
```
## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `EMAIL` | Your email address for NCBI API identification | Yes |
| `OPENAI_API_KEY` | OpenAI API key for LLM-powered extraction | Yes |

### External Plot Digitization

The workflow includes an external digitization step between Step 1 and Step 2:

1. **Supported Tools**: Any plot digitization software (WebPlotDigitizer, PlotDigitizer, etc.)
2. **Input**: Segmented plot images from `articles_data/{pmid}/segmented_images/`
3. **Output**: CSV files in `converted_tables/` with format `{pmid}_{figure}_plot{n}.csv`
4. **CSV Format**: 
   - First column: Time values
   - Subsequent columns: Fluorescence data (one per experimental condition)
   - Column headers should match legend symbols or descriptive names

### Processing Statistics

The toolkit provides detailed statistics throughout processing:

- **Step 1**: Articles found, figures extracted, ThT plots identified
- **Step 2**: CSV files processed, columns matched, data points consolidated
- **Final**: Total experimental conditions, unique proteins, data quality metrics

## 🔍 Output Structure

### Step 1 Outputs
```
articles_data/{pmid}/
├── {pmid}_metadata.json           # Article metadata
├── {pmid}_figures.json            # Figure information  
├── experimental_conditions/        # Extracted experimental data
│   └── figure{n}.json
├── images/                        # Original figures
│   └── figure_{n}.jpg
├── segmented_images/              # Individual plot segments  
│   └── figure{n}_plot{m}.jpg
└── bbox/                          # Bounding box data
    └── figure{n}_plot{m}/
```

### Step 2 Outputs
```
├── converted_tables/              # Input: Digitized CSV files
├── cleaned_tables/                # Processed CSV files
├── final_data/                    # Matched experimental data
│   └── {pmid}_{figure}_plot{n}_mapped_data.json
└── consolidated_experimental_data.csv  # Final dataset
```

## 📈 Data Quality & Validation

The toolkit includes comprehensive quality control:

### Automatic Validation
- **Plot Detection**: AI confidence scores for ThT plot identification
- **Data Matching**: LLM confidence in column-to-condition matching
- **Completeness**: Statistics on matched vs unmatched data columns
- **Consistency**: Cross-validation of experimental parameters

### Manual Review Points
1. **After Step 1**: Review segmented images for accuracy
2. **During Digitization**: Ensure proper column naming and data extraction
3. **After Step 2**: Review matching statistics and unmatched columns

### Error Handling
- Failed LLM calls are logged and skipped
- Missing CSV files are reported but don't stop processing
- Malformed data is flagged in output statistics

## 🤖 LLM Integration Details

### Model Usage
- **Primary Model**: GPT-4o for maximum accuracy
- **Temperature**: 0.1 for consistent outputs
- **Fallback**: Graceful degradation when API unavailable

### Prompt Engineering
- **ThT Plot Detection**: Trained to identify fluorescence vs time plots
- **Condition Extraction**: Extracts experimental parameters from figure context
- **Data Matching**: Matches CSV headers to experimental legend symbols

### Cost Optimization
- Caching of LLM responses
- Batch processing where possible
- Smart prompt sizing to minimize token usage

## 🚨 Troubleshooting

### Common Issues

1. **Missing Environment Variables**:
   ```
   ValueError: EMAIL environment variable is not set
   ValueError: OPENAI_API_KEY environment variable is not set
   ```
   Solution: Create a `.env` file with both required variables.

2. **LLM API Errors**:
   ```
   OpenAI API Error: Rate limit exceeded
   ```
   Solution: The toolkit includes automatic retry with exponential backoff. For high-volume processing, consider upgrading your OpenAI plan.

3. **Missing CSV Files**:
   ```
   ✗ No CSV found: {pmid}_{figure}_plot{n}_cleaned.csv
   ```
   Solution: Ensure external digitization step is completed and CSV files are properly named and located in `converted_tables/`.

4. **Plot Detection Issues**:
   ```
   No ThT plots identified in figure
   ```
   Solution: Manually review figure content. The AI model focuses on fluorescence vs time plots with ThT-related axes labels.

5. **Data Matching Failures**:
   ```
   ⚠ Invalid match tuple format, skipping data mapping
   ```
   Solution: Review CSV column headers and ensure they can be reasonably matched to experimental legend symbols.

### Debugging Tips

- **Verbose Output**: Both scripts provide detailed console output for monitoring progress
- **Intermediate Files**: Check `segmented_images/` and `final_data/` for intermediate processing results
- **Statistics**: Review processing statistics for data quality insights
- **Manual Review**: Manually inspect a few examples to validate automatic processing

### Performance Optimization

- **Parallel Processing**: For large datasets, consider processing PMIDs in parallel
- **LLM Caching**: LLM responses are cached to avoid redundant API calls
- **Selective Processing**: Process only specific figures or plots by modifying file filters

## 📋 API Rate Limits & Best Practices

### NCBI API Guidelines
- Maximum 3 requests per second for Entrez API
- Built-in delays between requests
- Automatic retry with exponential backoff
- Email identification required

### OpenAI API Guidelines  
- Rate limits vary by subscription tier
- Automatic retry for rate limit errors
- Token usage optimization through smart prompting
- Caching to minimize redundant calls

### Best Practices
- Process data in batches rather than all at once
- Monitor API usage and costs
- Validate outputs on small samples first
- Keep environment variables secure

## 🤝 Contributing

We welcome contributions to improve the toolkit:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Commit changes** (`git commit -m 'Add amazing feature'`)
6. **Push to branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Areas for Contribution
- Additional plot detection models
- Support for other fluorescence assays  
- Improved data validation algorithms
- Integration with other digitization tools
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NCBI** for providing PubMed and PMC APIs
- **OpenAI** for GPT-4 capabilities enabling intelligent data extraction
- **Scientific Community** for open-access publication of research data
- **BioPython** community for the Entrez interface
- **Plot Digitization Tools** that enable the external digitization step

---

**⚠️ Important Notes**: 
- This toolkit is designed for research purposes
- Ensure compliance with publisher terms of service and copyright regulations
- LLM processing requires careful validation for research applications
- External digitization step requires manual quality control
