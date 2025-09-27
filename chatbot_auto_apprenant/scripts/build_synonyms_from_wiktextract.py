import os
import gzip
import json
import unicodedata
import re

SRC = r"data\knowledge\wiktextract\fr-extract.jsonl.gz"
OUT = r"data\knowledge\synonyms_wikt.json"

# Petits mots à ignorer
STOP = {
    "a", "à", "au", "aux", "de", "du", "des", "la", "le", "les", "un", "une", "et", "ou", "mais",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "est", "sont", "etre", "être"
}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm_token(w: str) -> str:
    w = strip_accents(w.lower())
    w = re.sub(r"[^a-z0-9]+", " ", w).strip()
    return w

def maybe_add(map_to_canon: dict, alias: str, canon: str) -> None:
    if not alias or not canon:
        return
    if alias in STOP or canon in STOP:
        return
    prev = map_to_canon.get(alias)
    # Heuristique : garder le canon le plus court (souvent le lemme)
    if prev is None or len(canon) < len(prev):
        map_to_canon[alias] = canon

def extract_synonyms_from_senses(senses) -> set:
    """Retourne un set de synonymes normalisés trouvés dans senses[]."""
    out = set()
    for s in senses or []:
        # Format courant : list d'objets {"word": "..."} ou liste de strings
        for syn in s.get("synonyms") or []:
            if isinstance(syn, dict):
                w = syn.get("word") or syn.get("raw")
            else:
                w = syn
            if not w:
                continue
            w = norm_token(w)
            if w and len(w) >= 2 and w not in STOP:
                out.add(w)

        # Parfois sous "related" avec un tag "synonym" (rare)
        for rel in s.get("related") or []:
            if isinstance(rel, dict):
                tags = rel.get("tags") or []
                if "synonym" in tags:
                    w = rel.get("word") or rel.get("raw")
                    if w:
                        w = norm_token(w)
                        if w and len(w) >= 2 and w not in STOP:
                            out.add(w)
    return out

def main() -> None:
    if not os.path.exists(SRC):
        raise FileNotFoundError(SRC)

    to_canon: dict[str, str] = {}
    total_lines = fr_lines = sense_lines = syn_lines = 0

    with gzip.open(SRC, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            lang = obj.get("lang") or obj.get("lang_code")
            if lang not in ("French", "fr"):
                continue
            fr_lines += 1

            head = obj.get("word")
            if not head:
                continue
            head_norm = norm_token(head)
            if not head_norm or len(head_norm) < 2 or head_norm in STOP:
                continue

            senses = obj.get("senses") or []
            if not senses:
                continue
            sense_lines += 1

            syns = extract_synonyms_from_senses(senses)
            if not syns:
                continue
            syn_lines += 1

            # synonymes -> canon = head_norm
            for w in syns:
                if w == head_norm:
                    continue
                maybe_add(to_canon, w, head_norm)

            # auto-mapping canon->canon
            maybe_add(to_canon, head_norm, head_norm)

            if i % 200000 == 0:
                print(f".. {i:,} lignes lues (FR={fr_lines:,}, senses={sense_lines:,}, avec_syn={syn_lines:,}, map={len(to_canon):,})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(to_canon, f, ensure_ascii=False)

    print(f"OK: {len(to_canon):,} entrées écrites dans {OUT}")
    print(f"Stats: total={total_lines:,}, FR={fr_lines:,}, senses={sense_lines:,}, avec_syn={syn_lines:,}")
    if len(to_canon) == 0:
        print("Note: 0 résultat — soit le dump n'a pas de synonymes FR, soit son schéma a changé.")
        print("Astuce: si besoin, utilise l'alternative WordNet (OMW/WOLF).")

if __name__ == "__main__":
    main()
