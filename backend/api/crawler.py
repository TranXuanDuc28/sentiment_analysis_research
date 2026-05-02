import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

def is_valid_url(url: str):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

import json

def extract_from_schema(soup):
    """
    Extracts reviews from Schema.org (JSON-LD) structured data.
    High precision method used by major e-commerce sites.
    """
    reviews = []
    scripts = soup.find_all('script', type='application/ld+json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            # Schema can be a single object or a list
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                # Look for 'review' property in Product or independent Review objects
                found_reviews = []
                if item.get('@type') == 'Review':
                    found_reviews.append(item)
                elif 'review' in item:
                    rev = item['review']
                    found_reviews = rev if isinstance(rev, list) else [rev]
                
                for r in found_reviews:
                    body = r.get('reviewBody') or r.get('description')
                    if body and len(body) > 10:
                        reviews.append(body.strip())
        except:
            continue
    return list(set(reviews))

def parse_comments_from_soup(soup, url):
    """
    Core logic to extract comments from a BeautifulSoup object.
    Combines Schema.org and Heuristic extraction.
    """
    # 1. Try High-Precision Schema.org first
    schema_reviews = extract_from_schema(soup)[:5]
    if schema_reviews:
        print(f"[Schema.org] Found and returning {len(schema_reviews)} structured reviews from {url}")
        return schema_reviews

    # 2. Fallback to Review Hunter (Heuristics)
    # 1. Remove unwanted elements
    unwanted = [
        "script", "style", "nav", "footer", "header", "aside", "form", 
        ".faq", ".question-section", ".accordion", ".related", ".featured", 
        ".news-item", ".ads", ".banner", ".sidebar", ".suggested",
        ".trending", ".widget", ".featured-threads", ".trending-threads"
    ]
    for selector in unwanted:
        for element in soup.select(selector) if selector.startswith('.') or selector.startswith('#') else soup.find_all(selector):
            element.extract()

    # 2. Target Review/Comment containers ONLY
    target_elements = []
    review_selectors = [
        '.YNedDV', '.shopee-product-comment-list', # Specific for Shopee (via User F12)
        '.comment-content', '.review-content', '.user-review', 
        '.vne-comment-content', '.content-comment', '.comment-item',
        '.review-list', '.comment-list', '#comment-list', '#reviews',
        '.customer-reviews', '.comment-body', '.comment-text',
        '.desc-coment', '.content-comment', '.shopee-product-rating__main',
        '[itemprop="description"]', '.css-14vscas', '.content_comment',
        '.bbWrapper', '.message-content'
    ]
    
    for selector in review_selectors:
        found = soup.select(selector)
        if found:
            target_elements.extend(found)
    
    if not target_elements:
        # SMART FALLBACK: Identify clusters of repeating elements (typical for comments)
        # We look for containers that have multiple similar child elements
        candidates = []
        for tag in ['div', 'section', 'article']:
            for container in soup.find_all(tag):
                children = container.find_all(recursive=False)
                if len(children) >= 3:
                    # Check if children have similar text lengths (a sign of a comment list)
                    lengths = [len(c.get_text(strip=True)) for c in children if len(c.get_text(strip=True)) > 30]
                    if len(lengths) >= 3:
                        avg_len = sum(lengths) / len(lengths)
                        # If average length is substantial and variance isn't extreme
                        if avg_len > 40:
                            candidates.append((container, avg_len, len(lengths)))
        
        # Pick the candidate with the most children (likely the comment section)
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            target_elements = [candidates[0][0]]

    # 3. Extract text
    texts = []
    for el in target_elements:
        # For comment containers, we take the text more directly
        if 'bbWrapper' in el.get('class', []) or 'message-content' in el.get('class', []):
            sub_elements = [el]
        elif not el.find_all():
            sub_elements = [el]
        else:
            sub_elements = el.find_all(['p', 'div', 'span', 'blockquote'])
        
        for sub in sub_elements:
            # Increase limit to allow emojis, links, and formatting
            if len(sub.find_all()) > 10: continue
            
            txt = sub.get_text(" ", strip=True)
            
            # Stricter Filter:
            # - Reviews usually have spaces and multiple words
            # - Avoid common UI buttons/text
            noise_keywords = [
                'trả lời', 'thích', 'chia sẻ', 'viết bình luận', 'gửi', 'xem thêm', 
                'so sánh', 'đặc điểm nổi bật', 'đăng nhập'
            ]
            
            if 10 < len(txt) < 1000 and txt not in texts:
                if not any(k in txt.lower() for k in noise_keywords):
                    # For short comments, we ensure they don't look like UI buttons
                    words = txt.split()
                    # Skip if looks like a list ID (e.g. #5, #6)
                    if txt.strip().startswith('#') and len(txt.strip()) < 5 and any(c.isdigit() for c in txt):
                        continue
                        
                    if len(words) >= 2 and not txt.strip().endswith('?'): 
                        clean_txt = re.sub(r'\s+', ' ', txt)
                        if not any(clean_txt in existing for existing in texts):
                            texts.append(clean_txt)
    
    # Apply limit of 5 comments
    final_texts = texts[:5]
    print(f"[Review Hunter] Returning {len(final_texts)} comments (out of {len(texts)} found) from {url}")
    return final_texts

def extract_text_from_url(url: str):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return parse_comments_from_soup(soup, url)
    except Exception as e:
        print(f"[Crawler] Error: {e}")
        return []

def extract_text_from_html(html: str, url: str = "provided_html"):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        return parse_comments_from_soup(soup, url)
    except Exception as e:
        print(f"[Crawler] Error parsing HTML: {e}")
        return []

if __name__ == "__main__":
    # Quick test
    test_url = "https://vnexpress.net/nang-nong-tan-cong-nao-tim-the-nao-5067276.html"
    print(extract_text_from_url(test_url))
