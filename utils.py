import re
from typing import List, Optional
from urllib.parse import urljoin
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from langchain_core.documents import Document


pattern = r"(.*hust\.edu\.vn.*|.*\.pdf$|.*husteduvn\.sharepoint\.com.*)"  # Fixed regex
visited_urls = set()

# use this function to scrape all links from a specified URL
def scrape_page(driver, url, depth=0, max_depth=2, match_pattern=pattern, max_urls=5000):
    global visited_urls
    if len(visited_urls) >= max_urls:
        print(f"Reached max URLs: {max_urls}")
        return
    print(f"Current url number: {len(visited_urls)}")
    
    # Stop if depth exceeds max_depth or URL is already visited
    if depth > max_depth or url in visited_urls:
        return
    print(f"Scraping {url} at depth {depth}")

    # Add the URL to visited URLs, regardless of whether it matches the pattern
    visited_urls.add(url)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))

        # Collect all links' hrefs first to avoid stale elements
        links = driver.find_elements(By.TAG_NAME, 'a')
        hrefs = []

        for link in links:
            try:
                href = link.get_attribute('href')
                if href:
                    absolute_url = urljoin(url, href)
                    hrefs.append(absolute_url)
            except StaleElementReferenceException:
                print("Stale element while collecting href, skipping.")
                continue

        print(f"Found {len(hrefs)} links on {url}")

        # Process each collected URL
        for next_url in hrefs:
            if len(visited_urls) >= max_urls:
                print(f"Reached max URLs: {max_urls}")
                return
            
            # Only follow links that match the pattern
            if re.match(match_pattern, next_url) and next_url not in visited_urls:
                print(f"Following link: {next_url}")
                scrape_page(driver, next_url, depth + 1, max_depth)
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")

#use this class to Loader all links we got to Document Langchain
class CustomSeleniumURLLoader(SeleniumURLLoader):
    def __init__(self, urls: List[str], target_content_class: str, **kwargs):
        # Call the parent constructor with necessary arguments
        super().__init__(urls=urls, **kwargs)
        self.target_content_class = target_content_class

    def _build_metadata(self, url: str, driver):
        """Extracts metadata such as the title and last updated time."""
        metadata = {
            "source": url,
            "title": "No title found.",
            "lastUpdated": "No found.",
        }

        try:
            # Find both h2 and i elements inside the div.header
            elements = WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.header h2, div.header i"))
            )
            if elements:
                metadata['title'] = elements[0].text.strip()
                if len(elements) > 1:  # Ensure there is a second element for the date
                    last_update = re.search(r'Cập nhật:\s*(\d{2}:\d{2}\s\d{2}/\d{2}/\d{4})', elements[1].text)
                    if last_update:
                        metadata['lastUpdated'] = last_update.group(1)
        except (NoSuchElementException, TimeoutException):
            print(f"[WARNING] Metadata extraction failed for {url}")

        return metadata

    def load(self):
        """Loads content from URLs and extracts metadata while handling page rendering issues."""
        docs = []
        driver = self._get_driver()

        for url in self.urls:
            try:
                print(f"[INFO] Fetching URL: {url}")
                driver.get(url)
                time.sleep(2)  # Allow potential redirection

                # Debugging: Ensure correct URL is loaded
                actual_url = driver.current_url
                print(f"[INFO] Actual Loaded URL: {actual_url}")

                if actual_url.startswith("data:"):
                    print(f"[ERROR] Skipping due to unexpected 'data:' URL: {actual_url}")
                    continue

                if "login" in actual_url:
                    print("[WARNING] Redirected to a login page. Skipping this URL.")
                    continue

                # Wait for the page to fully load
                WebDriverWait(driver, 30).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete'
                )

                # Wait for the target content class element
                try:
                    element = WebDriverWait(driver, 60).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, self.target_content_class))
                    )
                except TimeoutException:
                    print(f"[ERROR] Timeout while waiting for content in {url}")
                    continue

                # Scroll to ensure full content is loaded
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                # Extract text content
                text = element.text.strip()
                print(f"[INFO] Extracted {len(text)} characters from {url}")

                # Build metadata
                metadata = self._build_metadata(url, driver)

                # Append the document
                docs.append(Document(page_content=text, metadata=metadata))

                # Randomized delay to prevent detection
                time.sleep(random.uniform(2, 5))

            except Exception as e:
                if self.continue_on_failure:
                    print(f"[ERROR] Skipping {url} due to: {e}")
                else:
                    driver.quit()
                    raise e

        driver.quit()
        return docs
    
# use this function to write data from Document Langchain to Json file for efficient RAG retriever
def write_data_to_json(filename: str, new_docs: List[Document]):
    dict_docs = []
    for doc in new_docs:
        document_dict = {
            "metadata": doc.metadata,
            "page_content": doc.page_content
        }
        dict_docs.append(document_dict)
    try:
        # Try to open and load the data from the file
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # If the file is empty or invalid JSON, initialize an empty list
                data = []
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty list
        data = []

    # Append new documents to the existing data
    data.extend(dict_docs)

    # Write the updated data back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        
def load_doc_from_json(filename: str) -> List[Document]:
    with open(filename, "r") as f:
        docs = json.load(f)
    documents = []
    for doc in docs:
        loaded_document = Document(metadata=doc["metadata"],
        page_content=doc["page_content"]
        )
        documents.append(loaded_document)
    return documents