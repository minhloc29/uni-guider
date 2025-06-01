import argparse
import json
from chatbot.utils import scrape_links

# Set up argument parser
parser = argparse.ArgumentParser(description="Scrape links for a given category.")
parser.add_argument(
    "category_name", type=str, help="The category name to scrape links for."
)

# Parse arguments
args = parser.parse_args()

# Load category data
with open("chatbot/scraping/category.json", "r") as f:
    category = json.load(f)

# Extract category name from arguments
category_name = args.category_name

# Ensure category exists
if category_name not in category:
    print(f"[ERROR] Category '{category_name}' not found in category.json")
    exit(1)

# Call scrape_links function
links = scrape_links(category[category_name]["url"], category_name)

# Print the scraped links (or handle them as needed)
print("[INFO] Scraped Links:", links)

# Run this: python -m chatbot.scraping.link_scraper activities
