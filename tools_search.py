import json
import os
import re
import urllib.parse
from typing import List

import httpx


IQS_API_KEY = os.getenv("IQS_API_KEY", "")
IQS_BASE = "https://cloud-iqs.aliyuncs.com"
IQS_TIMEOUT = 15

# Serper (Google) search API
SERPER_API_KEYS = [k.strip() for k in os.getenv(
    "SERPER_API_KEYS",
    ""
).split(",") if k.strip()]
SERPER_BASE = "https://google.serper.dev/search"
SERPER_TIMEOUT = 15
_serper_key_index = 0


def _contains_chinese(text: str) -> bool:
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


def _simplify_query(query: str) -> str:
    """Simplify complex query by removing modifiers and keeping core entities."""
    if not query or len(query) < 10:
        return query  # Already simple enough

    # Remove common Chinese/English modifiers and conjunctions
    simplified = re.sub(r'的|which|that|whose|who|when|where', ' ', query, flags=re.IGNORECASE)
    # Remove multiple spaces
    simplified = re.sub(r'\s+', ' ', simplified).strip()

    # If simplified is too different (too short), return original
    if len(simplified) < len(query) * 0.3:
        return query

    return simplified if simplified != query else query


def iqs_search(query: str, retries: int = 3) -> str:
    """Single query using IQS GenericSearch API, returns formatted results."""
    headers = {"X-API-Key": IQS_API_KEY}
    params = {
        "query": query,
        "timeRange": "NoLimit",
    }

    for attempt in range(retries):
        try:
            resp = httpx.get(
                f"{IQS_BASE}/search/genericSearch",
                headers=headers,
                params=params,
                timeout=IQS_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                return _format_search_results(query, data.get("pageItems", []))
            else:
                print(f"[search] IQS error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[search] Attempt {attempt + 1} failed: {e}")

    return f"Search failed for '{query}'. Please try a different query."


def serper_search(query: str) -> str:
    """Search using Serper (Google) API with key rotation and IQS fallback."""
    global _serper_key_index
    if not SERPER_API_KEYS:
        print("[search] No Serper API keys configured, falling back to IQS")
        return iqs_search(query)

    tried = 0
    while tried < len(SERPER_API_KEYS):
        key = SERPER_API_KEYS[_serper_key_index % len(SERPER_API_KEYS)]
        headers = {"X-API-Key": key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 10}
        try:
            resp = httpx.post(SERPER_BASE, headers=headers, json=payload, timeout=SERPER_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return _format_serper_results(query, data)
            elif resp.status_code in (400, 403, 429):
                # Key exhausted / no credits, rotate to next
                print(f"[search] Serper key {_serper_key_index} exhausted (HTTP {resp.status_code}), rotating")
                _serper_key_index += 1
                tried += 1
                continue
            else:
                print(f"[search] Serper error {resp.status_code}: {resp.text[:200]}")
                break
        except Exception as e:
            print(f"[search] Serper request failed: {e}")
            break

    # All keys exhausted or error, fallback to IQS
    print("[search] All Serper keys exhausted or failed, falling back to IQS")
    return iqs_search(query)


def _format_serper_results(query: str, data: dict) -> str:
    """Format Serper JSON response into the same style as IQS results."""
    organic = data.get("organic", [])
    if not organic:
        return f"No results found for '{query}'. Try with a more general query."

    web_snippets = []
    for idx, item in enumerate(organic, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        date = item.get("date", "")

        date_str = ""
        if date:
            date_str = f"\nDate published: {date}"

        source_str = ""
        if link:
            try:
                hostname = urllib.parse.urlparse(link).hostname or ""
                if hostname:
                    source_str = f"\nSource: {hostname}"
            except Exception:
                pass

        snippet_text = ""
        if snippet:
            snippet_text = "\n" + snippet

        entry = f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_text}"
        web_snippets.append(entry)

    content = (
        f"A search for '{query}' found {len(web_snippets)} results:\n\n"
        f"## Web Results\n"
        + "\n\n".join(web_snippets)
    )
    return content


def batch_search(queries: List[str], engines: List[str] = None) -> str:
    """
    Batch search with LLM-driven engine selection and fallback logic:
    - LLM can specify engine per query: 'google' (Serper) or 'bing' (IQS)
    - If engines not provided, auto-select by query language
    - Poor results trigger cross-engine fallback
    """
    if isinstance(queries, str):
        queries = [queries]
    if engines is None:
        engines = [None] * len(queries)
    # Pad engines list if shorter than queries
    while len(engines) < len(queries):
        engines.append(None)

    results = []
    for q, engine in zip(queries, engines):
        # Determine primary engine: LLM choice > auto-detect by language
        if engine == "google":
            use_google = True
        elif engine == "bing":
            use_google = False
        else:
            # Auto-detect: Chinese -> bing/IQS, English -> google/Serper
            use_google = not _contains_chinese(q)

        engine_name = 'Google/Serper' if use_google else 'Bing/IQS'
        source = f'LLM:{engine}' if engine else 'auto'
        print(f'[search] Query: "{q[:60]}" -> {engine_name} ({source})')
        if use_google:
            result = serper_search(q)
        else:
            result = iqs_search(q)

        # Check if results are poor (empty or very short)
        if _is_poor_result(result):
            # Try cross-engine fallback
            if use_google:
                print(f"[search] Poor Google results for '{q}', trying Bing/IQS fallback")
                fallback = iqs_search(q)
            else:
                print(f"[search] Poor Bing/IQS results for '{q}', trying Google fallback")
                fallback = serper_search(q)

            if not _is_poor_result(fallback):
                result = fallback
            else:
                # Both engines failed, try simplified query
                simplified = _simplify_query(q)
                if simplified != q:
                    print(f"[search] Both engines poor for '{q}', retrying simplified: '{simplified}'")
                    retry_result = serper_search(simplified) if use_google else iqs_search(simplified)
                    if not _is_poor_result(retry_result):
                        result = retry_result

        results.append(result)

    return "\n=======\n".join(results)


def _is_poor_result(result: str) -> bool:
    """Check if search result is poor quality (empty, too short, or error message)."""
    if not result or len(result) < 100:
        return True
    if "No results found" in result or "Search failed" in result:
        return True
    if "found 0 results" in result:
        return True
    return False


def _format_search_results(query: str, page_items: list) -> str:
    """Format IQS search results in DeepResearch style."""
    if not page_items:
        return f"No results found for '{query}'. Try with a more general query."

    web_snippets = []
    for idx, item in enumerate(page_items, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        html_snippet = item.get("htmlSnippet", "")
        publish_time = item.get("publishTime", "")
        hostname = item.get("hostname", "")

        date_str = ""
        if publish_time:
            date_str = f"\nDate published: {publish_time}"

        source_str = ""
        if hostname:
            source_str = f"\nSource: {hostname}"

        snippet_text = snippet if snippet else html_snippet
        if snippet_text:
            snippet_text = "\n" + snippet_text

        entry = f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_text}"
        web_snippets.append(entry)

    content = (
        f"A search for '{query}' found {len(web_snippets)} results:\n\n"
        f"## Web Results\n"
        + "\n\n".join(web_snippets)
    )
    return content
