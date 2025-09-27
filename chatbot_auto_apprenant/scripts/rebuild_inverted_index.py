import os, sys, sqlite3

# -- Permet d'importer 'src' même quand on lance ce script depuis 'scripts/'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.nlp.text_processor import normalize

DB_PATH = r"data\database\chatbot.db"

DDL = """
CREATE TABLE IF NOT EXISTS inverted_index (
  token   TEXT NOT NULL,
  pair_id INTEGER NOT NULL,
  UNIQUE(token, pair_id) ON CONFLICT IGNORE,
  FOREIGN KEY (pair_id) REFERENCES qa_pairs(id)
);
CREATE INDEX IF NOT EXISTS idx_inv_token ON inverted_index(token);
"""

def rebuild():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) s'assurer que la table existe
    cur.executescript(DDL)

    # 2) vider l'index
    cur.execute("DELETE FROM inverted_index")

    # 3) (re)peupler depuis qa_pairs
    cur.execute("SELECT id, question_text FROM qa_pairs")
    rows = cur.fetchall()

    links = 0
    for pid, qtext in rows:
        toks = normalize(qtext)
        for tok in set(toks):  # 1 lien par token/pair
            cur.execute(
                "INSERT OR IGNORE INTO inverted_index(token, pair_id) VALUES (?, ?)",
                (tok, pid),
            )
            links += 1

    conn.commit()
    conn.close()
    print(f"Index reconstruit: {len(rows)} paires, {links} liens.")

if __name__ == "__main__":
    rebuild()
