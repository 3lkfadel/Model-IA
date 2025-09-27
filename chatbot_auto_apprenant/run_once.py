import sqlite3
import json
import re, unicodedata
from collections import deque
from typing import Optional, List, Dict
from src.external.wiki_client import wiki_lookup_fr


from src.nlp.text_processor import normalize
from src.learning.similarity_engine import jaccard, order_score, confidence
from src.core.config import THRESHOLDS, WEIGHTS

DB_PATH = r"data\database\chatbot.db"

# Caches & mémoire de session
CACHE = {}
HISTORY = deque(maxlen=10)
LAST_SUBJECT: Optional[List[str]] = None  # ex: ['italie']

# Déclencheurs pour compléter une requête elliptique avec le contexte
ELLIPTIC_TRIGGERS = {
    "capitale", "population", "taille", "superficie", "hauteur", "longueur",
    "age", "année", "date", "prix", "cout", "coût", "definition", "définition"
}

# Stopwords basiques pour extraire le SUJET (sans synonymes)
SUBJECT_STOP = {
    "a","à","au","aux","de","du","des","d","l","la","le","les","un","une","et","ou","mais",
    "je","tu","il","elle","on","nous","vous","ils","elles","est","sont","etre","être",
    "quel","quelle","quels","quelles","quoi","qui","ou","où","quand","comment",
    # on retire aussi les mots de mesure/attributs (on ne veut garder que l'entité)
    "capitale","population","taille","superficie","hauteur","longueur","largeur","grandeur","prix","date","année"
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def simple_tokens(text: str) -> List[str]:
    """Tokenisation simple SANS synonymes, pour les heuristiques de contexte."""
    t = _strip_accents(text.lower())
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    toks = [w for w in re.split(r"\s+", t) if w]
    return toks


def extract_subject_tokens(text: str) -> List[str]:
    """Garde uniquement l’entité/sujet : tokens sans stopwords/attributs."""
    toks = simple_tokens(text)
    subj = [w for w in toks if w not in SUBJECT_STOP and len(w) >= 2 and not w.isdigit()]
    return subj


# ---------------------- Recherche & ranking ----------------------

def candidate_rows_by_index(user_query: str, limit: int = 50):
    toks = normalize(user_query)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        if toks:
            ph = ",".join("?" * len(toks))
            cur.execute(
                f"""
                SELECT pair_id, COUNT(*) as hits
                FROM inverted_index
                WHERE token IN ({ph})
                GROUP BY pair_id
                ORDER BY hits DESC, pair_id ASC
                LIMIT ?
                """,
                (*toks, limit),
            )
            ids = [r[0] for r in cur.fetchall()]
            if ids:
                ph2 = ",".join("?" * len(ids))
                cur.execute(
                    f"SELECT id, question_text, answer_text, upvotes, downvotes FROM qa_pairs WHERE id IN ({ph2})",
                    ids,
                )
                return cur.fetchall()
        # fallback si aucun token/candidat
        cur.execute(
            "SELECT id, question_text, answer_text, upvotes, downvotes FROM qa_pairs LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


def maybe_external_answer(resolved_q: str, last_subject: Optional[List[str]]):
    """
    Si la requête parle d'un attribut type 'population' et qu'on a un sujet,
    tente Wikipédia et renvoie (answer, url) ou (None, None).
    """
    toks = simple_tokens(resolved_q)  # SANS synonymes
    needs_population = "population" in toks
    subject = " ".join(last_subject) if last_subject else " ".join(extract_subject_tokens(resolved_q))

    if needs_population and subject:
        ans, url = wiki_lookup_fr(subject, "population")
        if ans:
            return ans, url
    return None, None




def get_ranked(user_query: str, limit: int = 50):
    """Retourne la liste classée des candidats [(sfinal, pid, q, a, slex, sord, sconf), ...]."""
    tq = normalize(user_query)
    rows = candidate_rows_by_index(user_query, limit=limit)
    ranked = []
    for (pid, q, a, up, down) in rows:
        tqb = normalize(q)
        slex = jaccard(tq, tqb)
        sord = order_score(tq, tqb)
        sconf = confidence(up, down)
        sfinal = WEIGHTS["LEX"] * slex + WEIGHTS["ORD"] * sord + WEIGHTS["CONF"] * sconf
        ranked.append((sfinal, pid, q, a, slex, sord, sconf))
    ranked.sort(reverse=True)
    return ranked


def decide(score: float) -> str:
    if score >= THRESHOLDS["DIRECT"]:
        return "DIRECT"
    if score >= THRESHOLDS["CONFIRM"]:
        return "CONFIRM"
    if score >= THRESHOLDS["ASK_REPHRASE"]:
        return "ASK_REPHRASE"
    return "LEARN"


# ---------------------- Logs & apprentissage ----------------------

def log_decision(user_query: str, chosen_pair_id: Optional[int], action: str, score: float):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_logs(user_query, chosen_pair_id, decision_action, score_final) VALUES (?, ?, ?, ?)",
            (user_query, chosen_pair_id, action, float(score)),
        )
        conn.commit()


def log_ambiguity(user_query: str, candidates: List[Dict]):
    payload = json.dumps(candidates, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ambiguity_events(user_query, candidates_json) VALUES (?, ?)",
            (user_query, payload),
        )
        conn.commit()


def upvote(pair_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE qa_pairs SET upvotes = upvotes + 1 WHERE id = ?", (pair_id,))
        conn.commit()


def downvote(pair_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE qa_pairs SET downvotes = downvotes + 1 WHERE id = ?", (pair_id,))
        conn.commit()


def learn_pair(question_text: str, answer_text: str):
    bad_prefix = "je ne sais pas encore. donne la bonne réponse"
    if not answer_text or answer_text.strip().lower().startswith(bad_prefix):
        print("Réponse invalide — non enregistrée.")
        return None
    if len(answer_text) > 2000:
        answer_text = answer_text[:2000]

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO qa_pairs(question_text, answer_text) VALUES (?, ?)",
            (question_text, answer_text),
        )
        new_id = cur.lastrowid
        conn.commit()

    toks = normalize(question_text)
    if toks:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO inverted_index(token, pair_id) VALUES (?, ?)",
                [(tok, new_id) for tok in set(toks)],
            )
            conn.commit()

    # invalider le cache pour cette clé
    key = " ".join(normalize(question_text))
    CACHE.pop(key, None)
    return new_id


# ---------------------- Contexte court-terme ----------------------

def resolve_context(user_q: str, last_subject: Optional[List[str]]):
    """Complète une requête elliptique avec le DERNIER SUJET (ex: ['italie'])."""
    q_clean = user_q.strip().lower()
    toks_simple = simple_tokens(q_clean)  # SANS synonymes
    few_tokens = len(toks_simple) <= 2
    starts_and = q_clean.startswith("et ") or q_clean in {"et", "et?"}
    keywords_only = any(t in ELLIPTIC_TRIGGERS for t in toks_simple)

    if last_subject and (few_tokens or starts_and or keywords_only):
        subject = " ".join(last_subject)  # ex: "italie"
        return f"{subject} {user_q}", True
    return user_q, False

def maybe_external_answer(resolved_q: str, last_subject: Optional[List[str]]):
    """
    Si la requête parle d'un attribut type 'population' et qu'on a un sujet,
    tente Wikipédia et renvoie (answer, url) ou (None, None).
    """
    toks = simple_tokens(resolved_q)  # SANS synonymes
    needs_population = "population" in toks
    subject = " ".join(last_subject) if last_subject else " ".join(extract_subject_tokens(resolved_q))

    if needs_population and subject:
        ans, url = wiki_lookup_fr(subject, "population")
        if ans:
            return ans, url
    return None, None

def safe_maybe_external(resolved_q: str, last_subject: Optional[List[str]]):
    try:
        return maybe_external_answer(resolved_q, last_subject)
    except Exception:
        return None, None



# ---------------------- Boucle principale ----------------------
if __name__ == "__main__":
    print("Chat prêt. Tape 'quit' pour sortir.\n")
    while True:
        q = input("Votre question: ").strip()
        if q.lower() in {"quit", "exit", ":q"}:
            break

        # Résolution de contexte
        resolved_q, used_ctx = resolve_context(q, LAST_SUBJECT)
        if used_ctx:
            print(f'(Contexte appliqué → requête enrichie: "{resolved_q}")')

        # Classement complet (sur la requête résolue)
        ranked = get_ranked(resolved_q, limit=50)
        good = None  # init pour usage ultérieur

        if not ranked:
            print("Base vide → LEARN")
            action = "LEARN"; sfinal = 0.0; pid = None; answer = None
        else:
            sfinal, pid, qbase, answer, slex, sord, sconf = ranked[0]

            # Règle indulgente : si très proche du seuil, on CONFIRM
            if slex >= 2/3 and sord >= 1.0 and sfinal >= 0.58:
                action = "CONFIRM"
                print(f"\nDecision: {action} (score={sfinal:.3f})")
                print(f" - Jaccard={slex:.3f}, Ordre={sord:.3f}, Confiance={sconf:.3f}")
                print(f"\nRéponse candidate:\n{answer}")
            else:
                # Détection d'ambiguïté (#2 très proche du #1)
                ambiguous = False
                if len(ranked) > 1:
                    s2, pid2, q2, a2, *_ = ranked[1]
                    if (sfinal >= THRESHOLDS["ASK_REPHRASE"]) and (abs(sfinal - s2) < 0.05):
                        ambiguous = True

                if ambiguous:
                    print("\nAmbiguïté détectée entre deux réponses proches.")
                    print(f"1) {a2 if len(a2)<=120 else a2[:117]+'...'}  (score≈{s2:.3f})")
                    print(f"2) {answer if len(answer)<=120 else answer[:117]+'...'}  (score≈{sfinal:.3f})")
                    sel = input("Choisis 1 / 2 (ou 0 pour aucune) : ").strip()

                    # Loggue l'ambiguïté
                    log_ambiguity(resolved_q, [
                        {"pid": int(pid2), "score": float(s2), "question": q2},
                        {"pid": int(pid),  "score": float(sfinal), "question": qbase},
                    ])

                    if sel == "1":
                        pid, answer, sfinal = pid2, a2, s2
                        action = "CONFIRM"
                        print(f"\nRéponse choisie (#1):\n{answer}")
                    elif sel == "2":
                        action = "CONFIRM"
                        print(f"\nRéponse choisie (#2):\n{answer}")
                    else:
                        action = "LEARN"
                        print("\nJe ne sais pas encore. Enseigne-moi la réponse.")
                else:
                    # Flux standard
                    action = decide(sfinal)
                    print(f"\nDecision: {action} (score={sfinal:.3f})")
                    print(f" - Jaccard={slex:.3f}, Ordre={sord:.3f}, Confiance={sconf:.3f}")
                    if action in {"DIRECT", "CONFIRM"}:
                        print(f"\nRéponse candidate:\n{answer}")
                    
                    elif action == "ASK_REPHRASE":
                        # TENTATIVE WIKIPÉDIA
                        ext_ans, ext_url = safe_maybe_external(resolved_q, LAST_SUBJECT)
                        if ext_ans:
                            print("\nProposition (Wikipédia) :")
                            print(ext_ans)
                            if ext_url:
                                print(f"Source : {ext_url}")
                            want = input("Apprendre cette réponse ? (y/n) : ").strip().lower()
                            if want == "y":
                                learn_pair(resolved_q, ext_ans)
                                print("✔ Appris (source Wikipédia).")
                                action = "CONFIRM"
                            else:
                                print("\nPeux-tu reformuler ?")
                        else:
                            print("\nPeux-tu reformuler ?")
                    else:
                        # action == LEARN → tentative Wikipédia aussi
                        ext_ans, ext_url = safe_maybe_external(resolved_q, LAST_SUBJECT)
                        if ext_ans:
                            print("\nProposition (Wikipédia) :")
                            print(ext_ans)
                            if ext_url:
                                print(f"Source : {ext_url}")
                            want = input("Apprendre cette réponse ? (y/n) : ").strip().lower()
                            if want == "y":
                                learn_pair(resolved_q, ext_ans)
                                print("✔ Appris (source Wikipédia).")
                                action = "CONFIRM"
                            else:
                                print("\nJe ne sais pas encore. Enseigne-moi la réponse.")
                        else:
                            print("\nJe ne sais pas encore. Enseigne-moi la réponse.")


        # Log de la décision (toujours la requête résolue)
        chosen_pair = pid if action in {"DIRECT", "CONFIRM"} else None
        log_decision(resolved_q, chosen_pair, action, sfinal)

        # Feedback / apprentissage
        if action in {"DIRECT", "CONFIRM"}:
            fb = input("Like ? (y/n/skip): ").strip().lower()
            if fb == "y":
                upvote(pid)
                print("✔ Merci pour le like.\n")
            elif fb == "n":
                downvote(pid)
                good = input("Donne la bonne réponse : ").strip()
                if good:
                    learn_pair(resolved_q, good)
                    print("✔ Corrigé, indexé et appris.\n")
        elif action == "LEARN":
            good = input("Je ne sais pas encore. Donne la bonne réponse : ").strip()
            if good:
                learn_pair(resolved_q, good)
                print("✔ Appris et indexé.\n")

        # Mise à jour du contexte
        HISTORY.append(q)  # texte brut saisi
        if action in {"DIRECT", "CONFIRM"} or (action == "LEARN" and good):
            LAST_SUBJECT = extract_subject_tokens(resolved_q)
            if LAST_SUBJECT:
                print(f"(Sujet retenu: {' '.join(LAST_SUBJECT)})")
