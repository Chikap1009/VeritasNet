import os
import csv
import time
from newspaper import Article
from tqdm import tqdm

BIAS_URLS = [
    # Biased opinion pieces (left and right)
    "https://www.foxnews.com/opinion",
    "https://www.nytimes.com/section/opinion",
    "https://www.theguardian.com/commentisfree",
    "https://www.breitbart.com/politics/",
    "https://www.huffpost.com/news/politics",
]

NEUTRAL_URLS = [
    # Neutral, fact-based reporting
    "https://www.reuters.com/",
    "https://www.apnews.com/",
    "https://www.bbc.com/news",
    "https://www.worldbank.org/en/news",
    "https://www.imf.org/en/News",
]

def extract_articles(url_list, label, limit=30):
    import requests
    from bs4 import BeautifulSoup

    all_texts = []

    for base_url in url_list:
        try:
            response = requests.get(base_url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            links = set()
            for link in soup.find_all("a", href=True):
                href = link['href']
                if href.startswith("http") and "article" in href.lower():
                    links.add(href)
                elif href.startswith("/") and "article" in href.lower():
                    links.add(base_url.rstrip("/") + href)

            links = list(links)[:limit]

            for link in tqdm(links, desc=f"Scraping {label} from {base_url}"):
                try:
                    article = Article(link)
                    article.download()
                    article.parse()
                    if len(article.text.split()) >= 50:
                        all_texts.append((article.text.strip(), label))
                except Exception as e:
                    continue

        except Exception as e:
            continue

    return all_texts

def save_to_csv(samples, output_file):
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for row in samples:
            writer.writerow(row)

if __name__ == "__main__":
    biased_samples = extract_articles(BIAS_URLS, "biased")
    neutral_samples = extract_articles(NEUTRAL_URLS, "neutral")
    all_data = biased_samples + neutral_samples
    save_to_csv(all_data, "expanded_dataset.csv")
    print(f"\n✅ Saved {len(all_data)} articles to expanded_dataset.csv")