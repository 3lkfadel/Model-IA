# src/external/wiki_client.py
import re
import urllib.parse
from typing import Optional, Tuple, Dict, Any
import requests

WIKI_API = "https://fr.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://fr.wikipedia.org/api/rest_v1/page/summary/"

# Wikipédia exige un User-Agent explicite
HEADERS = {
    "User-Agent": "chatbot-auto-apprenant/0.1 (+https://example.local; contact: dev@example.local)",
    "Accept": "application/json",
}

def _http_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r
    except requests.RequestException:
        return None

def _search_title_fr(query: str) -> Optional[str]:
    """Retourne le meilleur titre FR (opensearch puis fallback query search)."""
    r = _http_get(WIKI_API, params={
        "action": "opensearch",
        "search": query,
        "limit": 1,
        "namespace": 0,
        "format": "json"
    })
    if r is not None:
        try:
            data = r.json()
            if len(data) >= 2 and data[1]:
                return data[1][0]
        except Exception:
            pass

    r = _http_get(WIKI_API, params={
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json"
    })
    if r is not None:
        try:
            data = r.json()
            hits = data.get("query", {}).get("search", [])
            if hits:
                return hits[0].get("title")
        except Exception:
            pass

    return None

def _get_summary_fr(title: str) -> Optional[dict]:
    """REST summary de la page -> dict avec 'extract', 'content_urls', etc."""
    url = WIKI_REST_SUMMARY + urllib.parse.quote(title)
    r = _http_get(url)
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None

def _extract_population_text(text: str) -> Optional[str]:
    """Récupère une phrase contenant 'habitants' ou 'population'."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        s_clean = s.strip()
        low = s_clean.lower()
        if ("habitants" in low) or ("population" in low):
            if len(s_clean) > 220:
                s_clean = s_clean[:217] + "..."
            return s_clean
    return None

def wiki_lookup_fr(subject: str, attribute: str) -> Tuple[Optional[str], Optional[str]]:
    """
    subject: ex. 'italie'
    attribute: ex. 'population', 'capitale', etc.
    Retourne (réponse_candidate, url_source) ou (None, None)
    """
    if not subject:
        return None, None

    title = _search_title_fr(subject) or subject.capitalize()

    summary = _get_summary_fr(title) or _get_summary_fr(subject.capitalize())
    if summary is None:
        return None, None

    extract = summary.get("extract") or ""
    if not extract.strip():
        return None, None

    if attribute.lower() == "population":
        pop = _extract_population_text(extract)
        if pop:
            url = (summary.get("content_urls") or {}).get("desktop", {}).get("page")
            return pop, url

    # défaut: 1–2 phrases de résumé
    sentences = re.split(r"(?<=[.!?])\s+", extract.strip())
    text = " ".join(sentences[:2]).strip()
    if len(text) > 240:
        text = text[:237] + "..."
    url = (summary.get("content_urls") or {}).get("desktop", {}).get("page")
    return text or None, url
