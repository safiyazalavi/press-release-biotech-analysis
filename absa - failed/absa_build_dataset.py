#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
# Ensure NLTK data exists (works on 3.8+)
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import sent_tokenize

# -------- CONFIG --------
INPUT_CSV     = "cleaned_catalysts_cleaned_metadata.csv"
OUT_SENTENCES = "absa_sentences.jsonl"
OUT_TRIPLETS  = "absa_triplets_weak.jsonl"
OUT_FEATURES  = "absa_features.csv"
OUT_DEBUG     = "absa_debug.csv"

ID_COL   = "row_id"
TEXT_COL = "Catalyst"

EXPECTED_COLS = [
    "Ticker","Drug","Indication","Stage","Date","Catalyst","Shares_Millions","Market_Cap","Volatility",
    "Share_Price_D-1","Share_Price_D1","Share_Price_D5","Perc_Return_D1","Perc_Return_D5",
    "XBI_Return_D1","XBI_Return_D5","Short_Interest_Pct","Year_Beta","Daily_Volum_Traded"
]

# ----- Aspect regex (compiled) -----
ASPECT_PATTERNS = {
    "endpoint":   r"\b(primary|secondary)\s+endpoint(s)?\b|\bBCVA\b|\bDRSS\b|\bPFS\b|\bOS\b|\bORR\b|\bDOR\b|\bCR\b|\bPR\b",
    "pvalue":     r"\bp\s*[<=>]\s*\d*\.?\d+\b",
    "effect":     r"\b(HR|OR|IRR|RR)\s*[:=]\s*\d*\.?\d+|\b(?:increase|decrease|reduction|improvement)\s+of\s+\d+(?:\.\d+)?\b",
    "safety":     r"\b(safety|adverse events|AEs|SAEs|DLTs?|well[- ]tolerated|tolerability)\b",
    "sample":     r"\bn\s*=\s*\d{2,5}\b|\b\d{2,5}\s+(?:patients?|subjects?|participants?)\b",
    "phase":      r"\bPhase\s*(?:1\/2|2\/3|1|2|3)\b|\bpivotal\b|\bregistration[- ]enabling\b",
    "partner":    r"\b(partner(?:ed|ship)?|collaborat(?:e|ion)|license(?:d|ing)?|acquir(?:e|ed|ing)|alliance|co-?development|agreement)\b",
    "regulatory": r"\b(approval|BLA|NDA|MAA|priority review|fast[- ]track|breakthrough therapy|CRL|complete response letter|clinical hold)\b",
    "financial":  r"\b(upfront|milestone|royalt(?:y|ies)|offering|secondary|ATM|debt|equity)\b",
    "enrollment": r"\b(?:enrollment\s+(?:complete(?:d)?|completed|closed|finished|has\s+(?:been\s+)?completed|has\s+finished|completion)\b|"
                  r"fully\s+enrolled\b|"
                  r"last\s+patient\s+(?:in|enrolled)\b|"
                  r"patient\s+enrollment\s+(?:completed|finished|closed)\b)"
}

ASPECT_RE = {k: re.compile(v, re.I) for k, v in ASPECT_PATTERNS.items()}

# ----- Opinion cues → polarity -----
OPINION_POLARITY = [
    (r"\b(met|achieved|attained|statistically significant|p\s*<\s*0\.05|positive topline|successful)\b", "POS"),
    (r"\b(did not meet|failed to meet|missed|not statistically significant|non[- ]significant|p\s*>=\s*0\.05|negative topline|unsuccessful)\b", "NEG"),
    (r"\b(well[- ]tolerated|no\s+(DLTs?|SAEs?)|favorable safety|manageable safety)\b", "POS"),
    (r"\b(DLTs?|serious adverse events|safety concern|black box)\b", "NEG"),
    (r"\b(approval|BLA accepted|priority review|breakthrough therapy|fast[- ]track|orphan designation|top[- ]tier partner)\b", "POS"),
    (r"\b(CRL|complete response letter|clinical hold|terminated|withdrawn|partial clinical hold)\b", "NEG"),
]
OPN_RE = [(re.compile(p, re.I), pol) for p, pol in OPINION_POLARITY]

# Map aspect → feature flag
ASPECT_FLAG = {
    "endpoint":"has_endpoint","pvalue":"has_pvalue","effect":"has_effect","safety":"has_safety","sample":"has_sample",
    "phase":"has_phase","partner":"has_partner","regulatory":"has_regulatory","financial":"has_financial","enrollment":"has_enrollment",
    "catalyst":"has_catalyst"
}

def require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def extract_aspects(text: str):
    return [a for a, cre in ASPECT_RE.items() if cre.search(text)]

def vote_polarity_and_opinion(text: str):
    pos = any(cre.search(text) for cre, pol in OPN_RE if pol == "POS")
    neg = any(cre.search(text) for cre, pol in OPN_RE if pol == "NEG")
    if pos and not neg:
        pol = "POS"
    elif neg and not pos:
        pol = "NEG"
    elif pos and neg:
        pol = "NEU"
    else:
        return None, None
    # return a sample cue for debug
    for cre, p in OPN_RE:
        m = cre.search(text)
        if m:
            return pol, m.group(0)
    return pol, None

def main():
    df = pd.read_csv(INPUT_CSV)
    require_columns(df, EXPECTED_COLS)

    # Add row_id
    df[ID_COL] = np.arange(len(df))
    total_rows = len(df)

    # -------- 1) Sentence records --------
    sent_rows = []
    with open(OUT_SENTENCES, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            text = (str(row[TEXT_COL]).replace("…", "...").strip()
                    if pd.notna(row[TEXT_COL]) else "")
            if not text:
                continue
            sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
            # Catalyst is often just one sentence — still fine
            for j, s in enumerate(sents):
                rec = {
                    "row_id": int(row[ID_COL]),
                    "sent_id": f"{row[ID_COL]}_{j}",
                    "Ticker": row["Ticker"], "Drug": row["Drug"], "Indication": row["Indication"],
                    "Stage": row["Stage"], "Date": row["Date"], "text": s
                }
                sent_rows.append(rec)
                fout.write(json.dumps(rec) + "\n")

    # -------- 2) Weak ABSA triplets --------
    triplet_rows = []
    debug_rows = []
    with open(OUT_TRIPLETS, "w", encoding="utf-8") as fout:
        for rec in sent_rows:
            text = rec["text"]
            aspects = extract_aspects(text)
            pol, opn = vote_polarity_and_opinion(text)

            # Fallback: if sentiment exists but no aspect, use generic "catalyst"
            if pol and not aspects:
                aspects = ["catalyst"]

            if not aspects or not pol:
                continue

            triplets = [[a, opn, pol] for a in aspects]
            out = {"row_id": rec["row_id"], "sent_id": rec["sent_id"], "text": text, "triplets": triplets}
            fout.write(json.dumps(out) + "\n")
            triplet_rows.append(out)

            debug_rows.append({
                "row_id": rec["row_id"],
                "sent_id": rec["sent_id"],
                "text": text,
                "aspects": ",".join(aspects),
                "polarity": pol,
                "opinion_sample": opn
            })

    if debug_rows:
        pd.DataFrame(debug_rows).to_csv(OUT_DEBUG, index=False)

    # -------- 3) Aggregate to row-level features --------
    base_flags = {v:0 for v in ASPECT_FLAG.values()}
    absa_df = pd.DataFrame({"row_id": df[ID_COL].values})
    for k in base_flags: absa_df[k] = 0
    absa_df["pos_count"] = 0
    absa_df["neg_count"] = 0
    absa_df["neu_count"] = 0
    absa_df["headline_sentiment"] = 0

    strongest = {}  # row_id -> -1/0/+1

    for r in triplet_rows:
        rid = r["row_id"]
        aspects = {t[0] for t in r["triplets"]}
        pols    = [t[2] for t in r["triplets"]]

        # flags
        for a in aspects:
            if a in ASPECT_FLAG:
                absa_df.loc[absa_df["row_id"]==rid, ASPECT_FLAG[a]] = 1

        # counts
        absa_df.loc[absa_df["row_id"]==rid, "pos_count"] += pols.count("POS")
        absa_df.loc[absa_df["row_id"]==rid, "neg_count"] += pols.count("NEG")
        absa_df.loc[absa_df["row_id"]==rid, "neu_count"] += pols.count("NEU")

        # vote
        vote = 0
        if pols.count("NEG")>0 and pols.count("POS")==0:
            vote = -1
        elif pols.count("POS")>0 and pols.count("NEG")==0:
            vote = +1
        strongest[rid] = max(strongest.get(rid, 0), vote, key=abs)

    for rid, v in strongest.items():
        absa_df.loc[absa_df["row_id"]==rid, "headline_sentiment"] = v

    # Merge with original columns
    keep_cols = [c for c in EXPECTED_COLS if c in df.columns] + [ID_COL]
    merged = df[keep_cols].merge(absa_df, on="row_id", how="left").fillna(0)

    # numeric coercion
    for c in [
        "Shares_Millions","Market_Cap","Volatility","Share_Price_D-1","Share_Price_D1","Share_Price_D5",
        "Perc_Return_D1","Perc_Return_D5","XBI_Return_D1","XBI_Return_D5",
        "Short_Interest_Pct","Year_Beta","Daily_Volum_Traded",
        "pos_count","neg_count","neu_count","headline_sentiment"
    ]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged.to_csv(OUT_FEATURES, index=False)

    # -------- 4) Coverage stats --------
    n_sent = len(sent_rows)
    n_trip = len(triplet_rows)
    n_rows_with_any = merged[(merged.filter(like="has_").sum(axis=1) > 0) | (merged[["pos_count","neg_count","neu_count"]].sum(axis=1) > 0)].shape[0]

    print(f"✅ Wrote: {OUT_SENTENCES}, {OUT_TRIPLETS}, {OUT_FEATURES}")
    print(f"Sentences processed: {n_sent:,}")
    print(f"Sentences with triplets: {n_trip:,} ({(n_trip/max(1,n_sent))*100:.1f}%)")
    print(f"Rows with any ABSA signal: {n_rows_with_any}/{len(merged)} ({(n_rows_with_any/max(1,len(merged)))*100:.1f}%)")
    print(f"Debug CSV with matches: {OUT_DEBUG}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        print("Tip: ensure NLTK punkt & punkt_tab are available.")
        sys.exit(1)
