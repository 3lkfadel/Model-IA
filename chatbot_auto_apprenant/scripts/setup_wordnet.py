import os
from pathlib import Path
import wn

# 1) Dossier local pour la base wn.db et les lexiques
HOME = Path(r"data\knowledge\wordnet").resolve()
HOME.mkdir(parents=True, exist_ok=True)

# 2) Forcer wn à utiliser ce dossier (plus robuste sous Windows)
os.environ["WN_DATA_HOME"] = str(HOME)
try:
    # Selon la version de wn, l’attribut peut exister
    wn.config.data_home = str(HOME)  # type: ignore[attr-defined]
except Exception:
    pass

print("WN data home:", str(HOME))

# 3) Télécharger un lexique FR (WOLF via OMW)
try:
    wn.download("omw-fr")
except Exception as e:
    print("Info:", e)

# 4) Vérifier la présence d’un lexique FR utilisable
lex_ids = [lex.id for lex in wn.lexicons()]
print("Lexiques disponibles:", lex_ids)
