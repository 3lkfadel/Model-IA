from itertools import combinations

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def order_score(a, b):
    """Score d’ordre basé sur paires (a avant b) apparaissant dans le même ordre dans la séquence b."""
    idx_b = {tok: i for i, tok in enumerate(b)}
    pairs = list(combinations(range(len(a)), 2))
    if not pairs:
        return 0.0
    ok = 0
    for i, j in pairs:
        t1, t2 = a[i], a[j]
        if t1 in idx_b and t2 in idx_b and idx_b[t1] < idx_b[t2]:
            ok += 1
    return ok / len(pairs)

def confidence(up, down):
    total = (up or 0) + (down or 0)
    if total == 0:
        return 0.0
    score = ((up or 0) - (down or 0)) / total
    return max(0.0, score)
