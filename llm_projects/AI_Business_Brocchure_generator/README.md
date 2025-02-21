# AI Business Brochure Generator

An automated tool that generates professional company brochures by intelligently analyzing website content using GPT models. The generator crawls company websites, identifies relevant pages, and creates engaging brochures with customizable tone and language.

## Features

- Intelligent link analysis to identify relevant company information
- Automated content extraction from multiple website pages
- Customizable brochure generation with adjustable tone and language
- Streaming response display for real-time brochure creation
- Support for company culture, customers, product pricing, and career information

## How It Works

1. **Link Analysis**: The tool first analyzes the website's links to identify relevant pages such as About, Company, Careers, and Pricing pages using GPT.

2. **Content Extraction**: It then extracts content from the selected pages, including the landing page and other relevant sections.

3. **Brochure Generation**: Using the extracted content, it generates a well-structured brochure in markdown format with a customizable tone.

## Usage

```python
# Generate a brochure for a company
create_brochure(
    company_name="Company Name",
    url="https://company-website.com",
    language="English",
    tone="humorous, entertaining, jokey"
)
```

## Key Functions

- `get_links(url)`: Analyzes website links and selects relevant pages
- `get_all_details(url)`: Extracts content from selected pages
- `create_brochure(company_name, url, language, tone)`: Generates the final brochure with streaming output

## Requirements

- OpenAI API access
- Python 3.x
- Required libraries:
  - openai
  - json
  - IPython.display

## Limitations

- Content is truncated to 5,000 characters for API compatibility
- Requires valid website URLs with accessible content
- Website must be publicly accessible

## Output Format

The brochure is generated in markdown format and displayed in real-time using IPython's display functionality. It includes sections on:
- Company overview
- Products/Services
- Company culture
- Customer information
- Pricing (if available)
- Career opportunities (if available)
