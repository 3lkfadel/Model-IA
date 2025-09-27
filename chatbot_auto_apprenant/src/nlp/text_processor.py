import re
import unicodedata
import json
from pathlib import Path

_STOPWORDS = {
    "et","ou","mais","donc","ni","car",
    "le","la","les","un","une","des","du","de","d","au","aux","deux",
    "je","tu","il","elle","nous","vous","ils","elles","on",
    "me","te","se","moi","toi","lui","leur",
    "ce","cet","cette","ces","mon","ton","son","ma","ta","sa","mes","tes","ses",
    "qui","que","quoi","dont","où","quand","comment","pourquoi","combien",
    "est","sont","etre","ai","as","avons","avez","ont","a","aujourd","hui",
    "à","chez","vers","sans","avec","sous","sur","entre","par","pour","contre"
}
_SYNONYMS_MAP = None  # chargé à la demande

def load_synonyms(path="data/knowledge/synonyms_wikt.json"):
    global _SYNONYMS_MAP
    if _SYNONYMS_MAP is None:
        p = Path(path)
        if p.exists():
            _SYNONYMS_MAP = json.loads(p.read_text(encoding="utf-8"))
        else:
            _SYNONYMS_MAP = {}
    return _SYNONYMS_MAP


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

def normalize(text: str):
    t = text.lower()
    t = _strip_accents(t)
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    tokens = [tok for tok in re.split(r"\s+", t) if tok]

    # stopwords
    tokens = [tok for tok in tokens if tok not in _STOPWORDS]

    # synonymes (optionnel : activé si le fichier existe)
    syn = load_synonyms()  # si le fichier manque, ça renvoie {}
    if syn:
        tokens = [syn.get(tok, tok) for tok in tokens]

    return tokens

