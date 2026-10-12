from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import time

import requests
import trafilatura
from bs4 import BeautifulSoup


SOURCES = {
    "pandas": {
        "start_url": "https://pandas.pydata.org/docs/",
        "allowed_domains": ["pandas.pydata.org"],
        "allowed_prefixes": ["/docs/"],
    },
    "transformers": {
        "start_url": "https://huggingface.co/docs/transformers/",
        "allowed_domains": ["huggingface.co"],
        "allowed_prefixes": ["/docs/transformers/"],
    },
    "sentence_transformers": {
        "start_url": "https://www.sbert.net/",
        "allowed_domains": ["www.sbert.net", "sbert.net"],
        "allowed_prefixes": ["/"],
    },
}

OUTPUT_DIR = Path("data/raw")
URLS_DIR = Path("data/urls")
URLS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "rag-docs-project/0.1 documentation crawler"
}


def clean_url(url):
    return url.split("#")[0].rstrip("/")


def is_allowed(url, source_config):
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if parsed.netloc not in source_config["allowed_domains"]:
        return False

    return any(
        parsed.path.startswith(prefix)
        for prefix in source_config["allowed_prefixes"]
    )


def discover_sitemap(start_url):
    robots_url = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}/robots.txt"

    try:
        response = requests.get(
            robots_url,
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    for line in response.text.splitlines():
        if line.lower().startswith("sitemap:"):
            return line.split(":", 1)[1].strip()

    return None


def discover_urls(start_url, source_config):
    sitemap_url = discover_sitemap(start_url)

    if sitemap_url:
        print(f"Sitemap trouvé : {sitemap_url}")

        try:
            response = requests.get(
                sitemap_url,
                headers=HEADERS,
                timeout=60,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "xml")

            sitemap_links = [
                loc.get_text(strip=True)
                for loc in soup.find_all("loc")
            ]

            urls = {
                clean_url(url)
                for url in sitemap_links
                if is_allowed(url, source_config)
            }

            if urls:
                return sorted(urls)

        except requests.RequestException as error:
            print(f"Erreur pendant la lecture du sitemap : {error}")

    print(f"Aucun sitemap exploitable pour {start_url}")
    print("Utilisation de l'URL de départ uniquement.")

    return [clean_url(start_url)]


def file_name_from_url(url):
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{digest}.txt"


for category, config in SOURCES.items():
    print(f"\nDécouverte des URLs : {category}")

    urls = discover_urls(
        config["start_url"],
        config,
    )

    urls_file = URLS_DIR / f"{category}.json"
    urls_file.write_text(
        json.dumps(urls, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    category_dir = OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(urls)} URL(s) trouvée(s).")

    for number, url in enumerate(urls, start=1):
        output_file = category_dir / file_name_from_url(url)

        if output_file.exists():
            continue

        print(f"[{number}/{len(urls)}] {url}")

        try:
            downloaded = trafilatura.fetch_url(url)

            if not downloaded:
                print("  Échec du téléchargement.")
                continue

            text = trafilatura.extract(
                downloaded,
                include_links=True,
                include_tables=True,
            )

            if not text or len(text.strip()) < 100:
                print("  Contenu ignoré : texte trop court.")
                continue

            content = {
                "url": url,
                "category": category,
                "text": text,
            }

            output_file.write_text(
                json.dumps(
                    content,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            time.sleep(1)

        except Exception as error:
            print(f"  Erreur : {error}")

print("\nTéléchargement terminé.")