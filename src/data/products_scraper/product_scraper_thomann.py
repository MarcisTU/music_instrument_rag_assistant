import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


BASE_URL = "https://www.thomann.de/intl/"

CATEGORY_URLS = [
    "https://www.thomann.de/intl/all-products-from-the-category-guitars_and_basses.html?ls=50&pg=1",
    "https://www.thomann.de/intl/all-products-from-the-category-drums_and_percussion.html?ls=50&pg=1",
    "https://www.thomann.de/intl/all-products-from-the-category-keys.html?ls=50&pg=1",
]

REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}


def random_sleep(min_seconds=2.0, max_seconds=6.0):
    sleep_time = random.uniform(min_seconds, max_seconds)
    print(f"Sleeping {sleep_time:.2f}s")
    time.sleep(sleep_time)


def soupify(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def fetch_html(
    client: httpx.Client,
    url: str,
    retries: int = 3
) -> str:
    """
    Fetch HTML with retries and anti-403 cooldown handling.
    """

    for attempt in range(retries):
        try:
            print(f"[GET] {url}")
            response = client.get(url)

            # Explicit 403 handling
            if response.status_code == 403:
                cooldown = random.uniform(30, 90)
                print(f"[403] Forbidden detected. Cooling down for {cooldown:.2f}s")
                time.sleep(cooldown)

                continue

            response.raise_for_status()
            return response.text

        except Exception as e:
            print(f"[Retry {attempt + 1}/{retries}] Failed: {url}")

            print(e)
            if attempt == retries - 1:
                raise

            backoff = (2 ** attempt) + random.uniform(1, 5)
            print(f"Retrying in {backoff:.2f}s")
            time.sleep(backoff)

    return ""


def scrape_category_page(
    client: httpx.Client,
    page_url: str
) -> List[str]:

    html = fetch_html(client, page_url)
    soup = soupify(html)

    products = soup.select("div.product")
    product_links = []
    for product in products:
        a_tag = product.select_one("a.product__image")
        if not a_tag:
            continue

        href = a_tag.get("href")
        if not href:
            continue

        full_url = urljoin(BASE_URL, href)
        product_links.append(full_url)

    return product_links


def collect_product_links(
    client: httpx.Client,
    category_url: str,
    file_cache_path: str
) -> List[str]:

    # Load different category progress
    if os.path.exists(file_cache_path):
        with open(file_cache_path, "r", encoding="utf-8") as f:
            all_links = set([line.strip() for line in f if line.strip()])
    else:
        all_links = set()

    page = int(category_url.split("pg=")[1])   # get page number, allows for restarts if defined larger
    while True:
        paged_url = (category_url.split("&pg=")[0] + f"&pg={page}")

        print(f"[CATEGORY PAGE {page}]:: {paged_url}")
        try:
            links = scrape_category_page(
                client,
                paged_url
            )
        except Exception as e:
            print(f"Failed category page: {paged_url}")
            print(e)
            break

        if not links:
            print("No products found.")
            break

        before_count = len(all_links)
        all_links.update(links)
        added = len(all_links) - before_count

        print(f"Added {added} products")

        # Last page heuristic
        if len(links) < 50:
            print("Detected last page.")
            break

        # Cache product links so next time the script runs it skips the product links that exist
        if page % 10 == 0:
            with open(file_cache_path, "w", encoding="utf-8") as f:
                f.writelines(link + "\n" for link in all_links)

        page += 1
        random_sleep(3, 7)

    return list(all_links)


def scrape_reviews(client, product_page_soup):
    reviews_data = {
        "reviews": []
    }

    try:
        # Find "Read all reviews" button
        reviews_link_tag = product_page_soup.select_one(
            "a.product-reviews-detail-teaser__button"
        )

        if not reviews_link_tag:
            print(f"No reviews page")
            return reviews_data

        href = reviews_link_tag.get("href")
        if not href:
            return reviews_data

        reviews_url = urljoin(BASE_URL, href)

        # Small delay before next request
        time.sleep(random.uniform(3.0, 3.5))

        # Load reviews page
        reviews_html = fetch_html(client, reviews_url)
        reviews_soup = soupify(reviews_html)

        review_blocks = reviews_soup.select("div.customer-review")

        for block in review_blocks:
            title = None

            title_tag = block.select_one("[itemprop='headline']")

            if title_tag:
                title = title_tag.get_text(" ", strip=True)

            author = None
            date = None

            author_tag = block.select_one(".review-intro__author")

            if author_tag:
                author_text = author_tag.get_text(" ", strip=True)

                parts = author_text.rsplit(" ", 1)

                if len(parts) == 2:
                    author = parts[0]
                    date = parts[1]
                else:
                    author = author_text

            review_text = None

            text_tag = block.select_one(".fx-text-collapsible__fallback")

            if text_tag:
                review_text = text_tag.get_text("\n", strip=True)

            rating = None
            filler = block.select_one(".fx-rating-stars__filler")

            if filler:
                style = filler.get("style", "")
                match = re.search(r"width:\s*(\d+)%", style)

                if match:
                    percent = int(match.group(1))
                    rating = round((percent / 100) * 5, 1)

            votes_up = None
            votes_down = None
            vote_items = block.select(".action__item.js-vote")

            if len(vote_items) >= 2:
                up_tag = vote_items[0].select_one(".js-vote-value")
                down_tag = vote_items[1].select_one(".js-vote-value")

                if up_tag:
                    try:
                        votes_up = int(up_tag.get_text(strip=True))
                    except:
                        pass

                if down_tag:
                    try:
                        votes_down = int(down_tag.get_text(strip=True))
                    except:
                        pass

            reviews_data["reviews"].append({
                "title": title,
                "author": author,
                "date": date,
                "rating": rating,
                "text": review_text,
                "votes_up": votes_up,
                "votes_down": votes_down,
            })

        print(f"Extracted {len(reviews_data['reviews'])} reviews")

    except Exception as e:
        print(f"Failed reviews scrape")
        print(e)

    return reviews_data


def scrape_product(
    client: httpx.Client,
    product_url: str
) -> Dict:

    print(f"[PRODUCT]")
    html = fetch_html(client, product_url)
    soup = soupify(html)

    name = None
    name_tag = soup.select_one(
        "h1[itemprop='name']"
    )

    if name_tag:
        name = name_tag.get_text(strip=True)

    price = None
    price_selectors = [
        ".fx-typography-price-primary",
        ".product__price-primary",
        "[data-track-id='priceContainer']"
    ]

    for selector in price_selectors:
        tag = soup.select_one(selector)
        if tag:
            price = tag.get_text(strip=True)
            break

    in_stock = None

    stock_tag = soup.select_one(
        ".fx-availability"
    )

    if stock_tag:
        stock_text = stock_tag.get_text(
            " ",
            strip=True
        ).lower()

        in_stock = "in stock" in stock_text

    # Product description points
    description_points = [
        li.get_text(" ", strip=True)
        for li in soup.select("ul.product-text__list li")
        if li.get_text(" ", strip=True)
    ]


    image_urls = []
    fallback_imgs = soup.select(
        ".spotlight__item-image"
    )
    for img in fallback_imgs:
        src = img.get("src")

        if not src:
            continue

        full_src = urljoin(BASE_URL, src)

        if full_src not in image_urls:
            image_urls.append(full_src)


    breadcrumbs = []
    breadcrumb_tags = soup.select(
        ".fx-breadcrumb .stages__item [itemprop='name']"
    )

    for tag in breadcrumb_tags:
        text = tag.get_text(strip=True)

        if text:
            breadcrumbs.append(text)

    # Human-like delay
    reviews_data = scrape_reviews(client, soup)

    product_data = {
        "url": product_url,
        "name": name,
        "price": price,
        "in_stock": in_stock,
        "description_points": description_points,
        "image_urls": image_urls,
        "breadcrumbs": breadcrumbs,
        "reviews": reviews_data["reviews"]
    }

    return product_data


def scrape_products(
    client: httpx.Client,
    product_urls: List[str],
    product_cache_path: str = "../thomann_products.jsonl"
):
    total = len(product_urls)

    print("\nStarting product scraping...")
    print(f"Total products: {total}")

    # load cache if exists
    product_url_cache = set()
    if os.path.exists(product_cache_path):
        with open(product_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                if "url" in item:
                    product_url_cache.add(item["url"])

    print(f"Loaded existing scraped products: len={len(product_url_cache)}")

    for idx, url in enumerate(product_urls, start=1):
        if url in product_url_cache:
            print("Url already scraped. Skipping...")
            continue

        try:
            print(f"[{idx}/{total}]")

            product = scrape_product(
                client,
                url
            )

            # Save progressively
            with open(product_cache_path, "a+", encoding="utf-8") as f:
                f.write(json.dumps(product, ensure_ascii=False) + "\n")

            # Human-like delay
            random_sleep(5, 5.8)

        except Exception as e:
            print(f"Failed product: {url}")
            print(e)

            cooldown = random.uniform(10, 20)
            print(f"Cooling down for {cooldown:.2f}s")
            time.sleep(cooldown)


def main():
    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    limits = httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
    )

    with httpx.Client(
        headers=HEADERS,
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        http2=False,
    ) as client:
        all_product_links: Set[str] = set()

        file_cache_path = "./cache/product_links.txt"

        links = []
        if os.path.exists(file_cache_path):
            with open(file_cache_path, "r", encoding="utf-8") as f:
                links = [line.strip() for line in f if line.strip()]

        if len(links) == 0:
            for category_url in CATEGORY_URLS:
                links = collect_product_links(
                    client,
                    category_url,
                    file_cache_path
                )
                all_product_links.update(links)
        else:
            print(f"Using cached links: len={len(links)}")
            all_product_links.update(links)

        product_links = list(all_product_links)
        print(f"TOTAL PRODUCTS FOUND: {len(product_links)}")

        scrape_products(client, product_links, product_cache_path="./thomann_products.jsonl")


if __name__ == "__main__":
    main()