# AI Website Summarizer

A Python tool that extracts and summarizes website content using BeautifulSoup and GPT models. This tool removes irrelevant elements like scripts and styling, focusing on the main content for generating concise summaries.

## Features

- Clean content extraction with BeautifulSoup
- Automatic removal of irrelevant HTML elements (scripts, styles, images, inputs)
- GPT-powered content summarization
- Markdown-formatted output
- Support for custom User-Agent headers

## Installation

Required Python packages:
- requests
- beautifulsoup4
- openai
- IPython

## Usage

```python
# Generate a summary for a website
summarize("https://example.com")
```

## Components

### Website Class

The `Website` class handles web scraping and content extraction:
- Initializes with a URL
- Uses custom headers to mimic browser requests
- Extracts page title and main content
- Removes irrelevant HTML elements
- Formats text with newline separators

### Summarization Functions

- `user_prompt_for(website)`: Generates the prompt for GPT summarization
- `messages_for(website)`: Creates the message structure for GPT API
- `summarize(url)`: Main function that combines scraping and summarization

## Configuration

The tool uses a custom User-Agent header to ensure reliable website access:
```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
```

## System Prompt

The default system prompt instructs the AI to:
- Analyze website contents
- Ignore navigation-related text
- Provide summaries in markdown format
- Include news and announcements when present

## Output

The summary is displayed using IPython's Markdown display functionality, providing formatted, readable output.

## Customization

The system prompt can be modified to:
- Change the output language
- Adjust the summary style
- Focus on specific types of content

## Note

Ensure you have proper API credentials and respect websites' robots.txt and terms of service when using this tool.
