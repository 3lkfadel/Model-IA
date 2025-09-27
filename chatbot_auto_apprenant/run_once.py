import sqlite3
from src.nlp.text_processor import normalize
from src.learning.similarity_engine import jaccard, order_score, confidence
from src.core.config import THRESHOLDS, WEIGHTS
import json
from collections import deque
from typing import Optional

DB_PATH = r"data\database\chatbot.db"
CACHE = {}
HISTORY = deque(maxlen=10)   # mémorise les 10 derniers tours (texte brut)
LAST_TOPIC = None            # sujet résolu (texte) du dernier tour utile


def candidate_rows_by_index(user_query, limit=50):
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

def get_ranked(user_query, limit=50):
    """Retourne la liste classée des candidats [(sfinal, pid, q, a, slex, sord, sconf), ...]."""
    tq = normalize(user_query)
    rows = candidate_rows_by_index(user_query, limit=limit)
    ranked = []
    for (pid, q, a, up, down) in rows:
        tqb = normalize(q)
        slex = jaccard(tq, tqb)
        sord = order_score(tq, tqb)
        sconf = confidence(up, down)
        sfinal = WEIGHTS["LEX"]*slex + WEIGHTS["ORD"]*sord + WEIGHTS["CONF"]*sconf
        ranked.append((sfinal, pid, q, a, slex, sord, sconf))
    ranked.sort(reverse=True)
    return ranked

def best_match(user_query):
    tq = normalize(user_query)
    key = " ".join(tq)
    if key in CACHE:
        return CACHE[key]

    rows = candidate_rows_by_index(user_query, limit=50)
    ranked = []
    for (pid, q, a, up, down) in rows:
        tqb = normalize(q)
        slex = jaccard(tq, tqb)
        sord = order_score(tq, tqb)
        sconf = confidence(up, down)
        sfinal = WEIGHTS["LEX"] * slex + WEIGHTS["ORD"] * sord + WEIGHTS["CONF"] * sconf
        ranked.append((sfinal, pid, q, a, slex, sord, sconf))
    ranked.sort(reverse=True)
    top = ranked[0] if ranked else None

    if len(CACHE) > 200:
        CACHE.clear()
    CACHE[key] = top
    return top


def decide(score):
    if score >= THRESHOLDS["DIRECT"]:
        return "DIRECT"
    if score >= THRESHOLDS["CONFIRM"]:
        return "CONFIRM"
    if score >= THRESHOLDS["ASK_REPHRASE"]:
        return "ASK_REPHRASE"
    return "LEARN"


def log_decision(user_query, chosen_pair_id, action, score):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_logs(user_query, chosen_pair_id, decision_action, score_final) VALUES (?, ?, ?, ?)",
            (user_query, chosen_pair_id, action, float(score)),
        )
        conn.commit()

def log_ambiguity(user_query, candidates):
    """candidates: liste de dicts {'pid':int,'score':float,'question':str}"""
    payload = json.dumps(candidates, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ambiguity_events(user_query, candidates_json) VALUES (?, ?)",
            (user_query, payload),
        )
        conn.commit()




def upvote(pair_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE qa_pairs SET upvotes = upvotes + 1 WHERE id = ?", (pair_id,))
        conn.commit()


def downvote(pair_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE qa_pairs SET downvotes = downvotes + 1 WHERE id = ?", (pair_id,))
        conn.commit()


def learn_pair(question_text, answer_text):
    # 1) insérer la nouvelle paire
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO qa_pairs(question_text, answer_text) VALUES (?, ?)",
            (question_text, answer_text),
        )
        new_id = cur.lastrowid
        conn.commit()

    # 2) indexer immédiatement
    toks = normalize(question_text)
    if toks:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO inverted_index(token, pair_id) VALUES (?, ?)",
                [(tok, new_id) for tok in set(toks)],
            )
            conn.commit()

    # 3) invalider le cache pour cette clé
    key = " ".join(normalize(question_text))
    CACHE.pop(key, None)
    return new_id

ELLIPTIC_TRIGGERS = {
    "capitale","population","taille","superficie","hauteur","longueur",
    "age","année","date","prix","cout","coût","definition","définition"
}

def resolve_context(user_q: str, last_topic: Optional[str]):
    """Retourne (query_resolue, used_context: bool)"""
    q_clean = user_q.strip().lower()

    # Triggers basiques: début par "et", ou question très courte, ou mots-clés seuls
    few_tokens = len(normalize(q_clean)) <= 2
    starts_and = q_clean.startswith("et ") or q_clean in {"et", "et?"}
    keywords_only = any(k in normalize(q_clean) for k in ELLIPTIC_TRIGGERS)

    if last_topic and (few_tokens or starts_and or keywords_only):
        resolved = f"{last_topic} {user_q}"
        return resolved, True
    return user_q, False

if __name__ == "__main__":
    print("Chat prêt. Tape 'quit' pour sortir.\n")
    while True:
        q = input("Votre question: ").strip()
        if q.lower() in {"quit", "exit", ":q"}:
            break

                # Classement complet pour gérer aussi les cas ambigus
        ranked = get_ranked(q, limit=50)

        if not ranked:
            print("Base vide → LEARN")
            action = "LEARN"; sfinal = 0.0; pid = None; answer = None
        else:
            sfinal, pid, qbase, answer, slex, sord, sconf = ranked[0]

            # Détection d'ambiguïté si le #2 est très proche du #1
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

                # Journalise l'ambiguïté (pour analyse ultérieure)
                log_ambiguity(q, [
                    {"pid": int(pid2), "score": float(s2), "question": q2},
                    {"pid": int(pid),  "score": float(sfinal), "question": qbase},
                ])

                if sel == "1":
                    # bascule sur le second candidat
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
                    print("\nPeux-tu reformuler ?")
                else:
                    print("\nJe ne sais pas encore. Enseigne-moi la réponse.")


        # Log de la décision
        chosen_pair = pid if action in {"DIRECT", "CONFIRM"} else None
        log_decision(q, chosen_pair, action, sfinal)

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
                    learn_pair(q, good)
                    print("✔ Corrigé, indexé et appris.\n")
        elif action == "LEARN":
            good = input("Je ne sais pas encore. Donne la bonne réponse : ").strip()
            if good:
                learn_pair(q, good)
                print("✔ Appris et indexé.\n")
