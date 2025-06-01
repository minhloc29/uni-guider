import re
from typing import List, Optional, Literal
from urllib.parse import urljoin
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
)
from langchain_core.documents import Document
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import random

pattern = r"(.*hust\.edu\.vn.*|.*\.pdf$|.*husteduvn\.sharepoint\.com.*)"  # Fixed regex
visited_urls = set()


# use this function to scrape all links from a specified URL
def scrape_page(
    driver, url, depth=0, max_depth=2, match_pattern=pattern, max_urls=5000
):
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
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )

        # Collect all links' hrefs first to avoid stale elements
        links = driver.find_elements(By.TAG_NAME, "a")
        hrefs = []

        for link in links:
            try:
                href = link.get_attribute("href")
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


# use this class to Loader all links we got to Document Langchain
class CustomSeleniumURLLoader(SeleniumURLLoader):
    def __init__(
        self,
        urls: List[str],
        category_name: Literal[
            "scholarship", "career", "freshman_knowledge", "guide", "activities"
        ],
        **kwargs,
    ):
        # Call the parent constructor with necessary arguments
        # get urls from json file of data folder

        super().__init__(urls=urls, **kwargs)
        self.category_name = category_name
        with open("rag/category.json", "r") as f:
            self.category = json.load(f)

    def _build_metadata(self, url: str, driver):
        """Extracts metadata such as the title and last updated time."""
        metadata = {
            "category": self.category_name,
            "source": url,
            "title": "No title found.",
        }
        try:
            element = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.category[self.category_name]["title_tag"])
                )
            )
            metadata["title"] = element.text.strip()
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
                    print(
                        f"[ERROR] Skipping due to unexpected 'data:' URL: {actual_url}"
                    )
                    continue

                if "login" in actual_url:
                    print("[WARNING] Redirected to a login page. Skipping this URL.")
                    continue

                # Wait for the page to fully load
                WebDriverWait(driver, 30).until(
                    lambda d: d.execute_script("return document.readyState")
                    == "complete"
                )

                # Wait for the target content class element
                try:
                    element = WebDriverWait(driver, 60).until(
                        EC.visibility_of_element_located(
                            (
                                By.CSS_SELECTOR,
                                self.category[self.category_name]["page_content_tag"],
                            )
                        )
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
        document_dict = {"metadata": doc.metadata, "page_content": doc.page_content}
        dict_docs.append(document_dict)
    try:
        # Try to open and load the data from the file
        with open(filename, "r") as file:
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
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_doc_from_json(filename: str) -> List[Document]:
    with open(filename, "r") as f:
        docs = json.load(f)
    documents = []
    for doc in docs:
        loaded_document = Document(
            metadata=doc["metadata"], page_content=doc["page_content"]
        )
        documents.append(loaded_document)
    return documents


# Scrape the given links to get all the links inside and write as JSON file to data folder
def scrape_links(
    root_url,
    category: Literal[
        "scholarship", "career", "activities", "guide", "freshman_knowledge"
    ],
    i=1,
):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    all_links = set()
    driver.get(root_url)
    driver.implicitly_wait(10)
    # Extract all links from a category
    print(f"Scrape the category: {category}")
    if category == "career":
        next_buttons = driver.find_elements(By.CSS_SELECTOR, "button.btn-next")
        for next_button in next_buttons:
            while True:
                # Extract all links on the current page
                elements = driver.find_elements(By.CSS_SELECTOR, "div.card-CompanyLst")
                a_values = [
                    element.find_element(By.TAG_NAME, "a")
                    for element in elements
                    if len(element.find_elements(By.TAG_NAME, "a")) > 0
                ]
                href_values = [a_value.get_attribute("href") for a_value in a_values]
                all_links.update(set(href_values))
                try:
                    if next_button.get_attribute("disabled") is not None:
                        print("Pagination ended.")
                        break  # Exit loop if button is disabled

                    next_button.click()
                    time.sleep(2)  # Wait for the next page to load

                except Exception as e:
                    print("No more pages or error:", e)
                    break  # Exit loop if the button is missing

                print(f"Page {i} done.")
                i += 1
        all_links = list(all_links)
    else:
        elements = driver.find_elements(
            By.CSS_SELECTOR, "div.el-col"
        )  # for img in images:
        a_values = [element.find_element(By.TAG_NAME, "a") for element in elements]
        all_links = [a_value.get_attribute("href") for a_value in a_values]

    link_file = f"data/{category}_links.json"
    with open(link_file, "w") as f:
        json.dump(all_links, f, indent=4)
    print("Scraping completed. Links saved.")
    driver.quit()


# Load one category into json, later that we read as document for langchain vectorstore
def document_writer(
    category_name: Literal[
        "activities", "career", "freshman_knowledge", "guide", "schoolarship"
    ],
):
    json_file = f"data/{category_name}_links.json"
    with open(json_file, "r") as f:
        urls = json.load(f)
    loader = CustomSeleniumURLLoader(urls, category_name=category_name)
    docs = loader.load()
    write_data_to_json("data/document_langchain.json", docs)
    return docs


# with open("rag/category.json", "r") as f:
#     category = json.load(f)

# category_list = list(category.keys())
# for c in tqdm(category_list):
#     document_writer(c)
# write_data_to_json("data/document_langchain.json", docs)
