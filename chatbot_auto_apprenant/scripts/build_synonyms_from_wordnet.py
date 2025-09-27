import os
import json
import re
import unicodedata
from pathlib import Path
import wn

# Même data home que setup_wordnet.py
HOME = Path(r"data\knowledge\wordnet").resolve()
HOME.mkdir(parents=True, exist_ok=True)
os.environ["WN_DATA_HOME"] = str(HOME)
try:
    wn.config.data_home = str(HOME)  # type: ignore[attr-defined]
except Exception:
    pass

OUT = r"data\knowledge\synonyms_wikt.json"

STOP = {
    "a","à","au","aux","de","du","des","la","le","les","un","une","et","ou","mais",
    "je","tu","il","elle","on","nous","vous","ils","elles","est","sont","etre","être"
}

# On privilégie des canons concrets/usuel (pas "altitude"/"élevation" comme canon pour tout)
PREFERRED = {
    "hauteur","taille","longueur","largeur","profondeur","voiture","vehicule","véhicule",
    "capitale","population","vitesse","distance","temperature","température","ville","pays"
}
# Jamais canon (trop ambigus/courts) → empêche "onde", "haut", "auto" d'être la forme cible
BANNED_CANON = {"onde","haut","auto"}

# Toujours se mapper à soi (ne jamais normaliser vers autre chose)
LOCK_SELF = {"onde","haut"}

SUFFIX_PREF = ("eur","euse","ion","ure","té","esse","ance","ence","ité","tude")

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(w: str) -> str:
    w = strip_accents(w.lower())
    w = re.sub(r"[^a-z0-9]+", " ", w).strip()
    return w

def score_canon(w: str) -> int:
    s = 0
    if w in PREFERRED:
        s += 5
    for suf in SUFFIX_PREF:
        if w.endswith(suf):
            s += 1
    if len(w) <= 3:
        s -= 2
    if w in BANNED_CANON:
        s -= 10
    return s

def main():
    # Choisir un lexique FR (omw-fr de préférence)
    lexicons = [l for l in wn.lexicons() if l.id.startswith(("omw-fr","fr","wn-fr"))]
    if not lexicons:
        raise RuntimeError("Aucun lexique FR. Lance d'abord: python .\\scripts\\setup_wordnet.py")
    lex = lexicons[0]

    to_canon = {}
    count_sets = 0

    # IMPORTANT: Restreindre aux NOMS
    for synset in wn.synsets(lexicon=lex.id, pos='n'):
        lemmas_raw = synset.lemmas()
        lemmas = []
        for l in lemmas_raw:
            n = norm(l)
            if n and n not in STOP and len(n) >= 2:
                lemmas.append(n)
        if len(lemmas) < 2:
            continue

        count_sets += 1
        # Canon = meilleur score, et en cas d'égalité on préfère le PLUS COURT (pas le plus long)
        canon = max(lemmas, key=lambda w: (score_canon(w), -len(w)))

        # Si le canon est banni, on prend le meilleur suivant
        if canon in BANNED_CANON:
            candidates = sorted(lemmas, key=lambda w: (score_canon(w), -len(w)), reverse=True)
            for cand in candidates:
                if cand not in BANNED_CANON:
                    canon = cand
                    break

        # Remplir la map alias -> canon
        for w in lemmas:
            prev = to_canon.get(w)
            if prev is None:
                to_canon[w] = canon
            else:
                # Remplacer si le nouveau canon est mieux scoré (ou aussi bon mais plus court)
                if (score_canon(canon), -len(canon)) > (score_canon(prev), -len(prev)):
                    to_canon[w] = canon
        # Auto-map canon -> canon
        to_canon.setdefault(canon, canon)

    # Overrides ciblés (domaines concrets)
    overrides = {
        "altitude": "hauteur",
        "elevation": "hauteur",
        "élévation": "hauteur",
        "vehicule": "voiture",
        "véhicule": "voiture",
        "auto": "voiture",
    }
    for k, v in overrides.items():
        to_canon[k] = v

    # Verrous: certains termes gardent leur forme
    for k in LOCK_SELF:
        to_canon[k] = k

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(to_canon, f, ensure_ascii=False)

    print(f"OK: {len(to_canon):,} entrées (depuis {count_sets:,} synsets) écrites dans {OUT}")

if __name__ == "__main__":
    main()
