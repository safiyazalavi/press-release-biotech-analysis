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
INPUT_CSV     = "data/cleaned_catalysts_cleaned_metadata.csv"
OUT_SENTENCES = "absa/absa_sentences.jsonl"
OUT_TRIPLETS  = "absa/absa_triplets_weak.jsonl"
OUT_FEATURES  = "data/absa_features.csv"
OUT_DEBUG     = "data/absa_debug.csv"

ID_COL   = "row_id"
TEXT_COL = "Catalyst"

EXPECTED_COLS = [
    "Ticker","Drug","Indication","Stage","Date","Catalyst","Shares_Millions","Market_Cap","Volatility",
    "Share_Price_D-1","Share_Price_D1","Share_Price_D5","Perc_Return_D1","Perc_Return_D5",
    "XBI_Return_D1","XBI_Return_D5","Short_Interest_Pct","Year_Beta","Daily_Volum_Traded"
]

# --- light text normalization to fix common typos & artifacts ---
NORMALIZE_REPLACEMENTS = [
    (r"\bsubmitteed\b", "submitted"),           # typo seen in your CSV
    (r"â\xa0read less", ""),                    # scrape artifact
    (r"\s+", " "),                              # collapse whitespace
]

def preprocess_text(s: str) -> str:
    s = s or ""
    for pat, rep in NORMALIZE_REPLACEMENTS:
        s = re.sub(pat, rep, s, flags=re.I)
    return s.strip()


# ----- Aspect regex (compiled) -----
# ----- Aspect regex (compiled) -----
ASPECT_PATTERNS = {
    # endpoints / p-values (kept)
    "endpoint":   r"\b(primary|secondary)\s+endpoint(s)?\b|\bBCVA\b|\bDRSS\b|\bPFS\b|\bOS\b|\bORR\b|\bDOR\b|\bCR\b|\bPR\b",
    "pvalue":     r"\bp\s*[<=>]\s*\d*\.?\d+\b",

    # effect / comparator language (NEW: 'compared to')
    "effect": (
        r"\b(HR|OR|IRR|RR)\s*[:=]\s*\d*\.?\d+"
        r"|\b(?:increase|decrease|reduction|improvement)\s+of\s+\d+(?:\.\d+)?\b"
        r"|\b(?:greater|superior)\s+(?:reduction|increase|improvement)\s+(?:vs\.?|versus|to|compared\s+to)\s+(?:placebo|control)\b"
        r"|\b(?:increase|decrease|reduction|improvement)\s+(?:vs\.?|versus|over|compared\s+to)\s+(?:placebo|control)\b"
        r"|\b(?:least[-\s]*squares?|ls)\s+mean\b"  # LS mean phrasing seen in your file
    ),

    "safety":     r"\b(safety|adverse events|AEs|SAEs|DLTs?|well[- ]tolerated|tolerability|boxed warning)\b",

    "sample":     r"\bn\s*=\s*\d{2,5}\b|\b\d{2,5}\s+(?:patients?|subjects?|participants?)\b",

    # phases (NEW: allow 1b/2a/2b etc.)
    "phase":      r"\bPhase\s*(?:1b\/?2a|1b|2a|2b|3b|1\/2|2\/3|1|2|3)\b|\bpivotal\b|\bregistration[- ]enabling\b",

    "partner":    r"\b(partner(?:ed|ship)?|collaborat(?:e|ion)|license(?:d|ing)?|acquir(?:e|ed|ing)|alliance|co-?development|agreement)\b",

    # regulatory (NEW: 'cleared by the FDA'; tolerant 'submi(t|tt)ed')
    "regulatory": (
        r"\b(approval|BLA|NDA|MAA|priority review|fast[- ]track|breakthrough therapy|CRL|complete response letter|clinical hold)\b"
        r"|\b(?:BLA|NDA|MAA)\s+(?:submi(?:t|tt)ed|submission|filed|filing|resubmitted|accepted|acceptance|validated|approved)\b"
        r"|\b(?:FDA|EMA)\s+(?:acceptance|acknowledg(?:e|ment)\s+letter|acceptance\s+letter|granted|approved|cleared)\b"
        r"|\bcleared\s+by\s+the\s+FDA\b"  # common in your data
        r"|\bFDA\s+(?:approved|approval)\b"
    ),

    # financial (kept from your last update; can tune later)
    "financial": (
        r"\b(upfront|milestone|royalt(?:y|ies)|offering|secondary|ATM|at-the-market|debt|equity|convertible(?:\s+notes)?|PIPE|registered\s+direct|follow[- ]on)\b"
        r"|\b(?:public|underwritten)\s+offering\b"
        r"|\b(raise|raised|raising)\s+\$?\d+(?:\.\d+)?\s*(?:m|million|b|billion)?\b"
        r"|\b(?:cash\s+runway|going\s+concern|liquidity\s+constraints?)\b"
        r"|\b(resource\s+allocation|strategic\s+reprioritization)\b"
        r"|\b(discontinue(?:d|s|ing)?|wind\s+down|halt)\s+(?:program|trial|development)\b"
    ),

    # enrollment (NEW: LSO/subject variants; passive 'was completed')
    "enrollment": (
        r"\b(?:enrollment\s+(?:complete(?:d)?|closed|finish(?:ed)?|has\s+(?:been\s+)?completed|was\s+completed|has\s+finished|completion)\b|"
        r"fully\s+enrolled\b|"
        r"(?:first|last)\s+(?:patient|subject)\s+(?:in|enrolled|out|visit)\b|"
        r"\bLSO\b|"                                   # last subject out
        r"(?:FPI|LSI|LPI|LPO|LPD)\b)"
    ),

    # clinical ops & comms (NEW: cohort #s; dose-escalation; topline/readout; abstract)
    "clinical": (
        r"\b(dosing|dose)\s+(?:initiated|has\s+begun|began|started|underway)\b"
        r"|\bfirst\s+(?:patient|subject)\s+(?:dosed|treated)\b|\bFPI\b"
        r"|\b(?:first|second|third|initial)\s+cohort\s+(?:completed|complete|finished)\b|\bcohort\s*\d+\s+(?:complete|completed)\b"
        r"|\b(?:dose[- ]?escalation|dose[- ]?expansion)\s+(?:complete(?:d)?|initiated|started|underway|begun)\b"
        r"|\b(?:last\s+(?:patient|subject)\s+(?:dosed|treated)|LPD)\b"
        r"|\bIND[- ]enabling\s+studies?\s+(?:complete|completed|finished|concluded)\b"
        r"|\b(?:significant|statistically\s+significant)\s+improvement\b"
        r"|\b(?:top[- ]?line|topline)\s+(?:data|results)\b|\b(read[- ]?out|readout)\b"
        r"|\b(?:data|results|analysis|abstract)\s+(?:to\s+be\s+presented|will\s+be\s+presented|presented|accepted\s+for\s+presentation)"
        r"\s+at\s+(?:ASCO|AACR|ESMO|AASLD|AAO|ARVO|ASH|EULAR|ADA|SABCS|CTAD|AAIC|NANS|BBSW|DIA|WCC|WCLC|SITC|SID|AAO-HNS)\b"
        r"|\b(?:data|results|analysis|abstract)\s+(?:to\s+be\s+presented|will\s+be\s+presented|presented|accepted\s+for\s+presentation)"
        r"\s+at\s+(?:a|an)?\s*(?:scientific|medical)\s+(?:conference|meeting|congress)\b"
    ),
}

ASPECT_RE = {k: re.compile(v, re.I) for k, v in ASPECT_PATTERNS.items()}


# Map aspect → feature flag
ASPECT_FLAG = {
    "endpoint":"has_endpoint","pvalue":"has_pvalue","effect":"has_effect","safety":"has_safety","sample":"has_sample",
    "phase":"has_phase","partner":"has_partner","regulatory":"has_regulatory","financial":"has_financial",
    "enrollment":"has_enrollment","clinical":"has_clinical","catalyst":"has_catalyst"
}


# ----- Opinion cues → polarity -----
# ----- Opinion cues → polarity -----
OPINION_POLARITY = [
    # Endpoints / stats
    (r"\b(met|achieved|attained|statistically significant|p\s*<\s*0\.05|positive topline|successful)\b", "POS"),
    (r"\b(did not meet|failed to meet|missed|not statistically significant|non[- ]significant|p\s*>=\s*0\.05|negative topline|unsuccessful)\b", "NEG"),

    # Safety
    (r"\b(well[- ]tolerated|no\s+(DLTs?|SAEs?)|favorable safety|manageable safety)\b", "POS"),
    (r"\b(DLTs?|serious adverse events|safety concern|black box|boxed warning)\b", "NEG"),

    # Regulatory (NEW: “cleared by the FDA”, tolerant submitted)
    (r"\b(?:FDA)\s+(?:approval|approved)\b", "POS"),
    (r"\bcleared\s+by\s+the\s+FDA\b", "POS"),
    (r"\b(approval|approv(?:ed|al)|BLA\s+accepted|priority review|breakthrough therapy|fast[- ]track|orphan designation)\b", "POS"),
    (r"\b(?:BLA|NDA|MAA)\s+(?:submi(?:t|tt)ed|submission|filed|filing|resubmitted|accepted|acceptance|validated|approved)\b", "POS"),
    (r"\b(?:FDA|EMA)\s+(?:acceptance|acknowledg(?:e|ment)\s+letter|acceptance\s+letter|granted|approved|cleared)\b", "POS"),
    (r"\b(CRL|complete\s+response\s+letter|refusal\s+to\s+file|RTF|clinical\s+hold|partial\s+clinical\s+hold|withdrawn|withdrawal|terminated|suspended|deficiency\s+letter)\b", "NEG"),

    # Enrollment (NEW: LSO/subject phrasing)
    (r"\b(enrollment\s+(?:complete(?:d)?|closed|finish(?:ed)?|has\s+(?:been\s+)?completed|was\s+completed|has\s+finished|completion)|fully\s+enrolled)\b", "POS"),
    (r"\b(?:first|last)\s+(?:patient|subject)\s+(?:in|enrolled|out|visit)\b|\b(?:FPI|LSI|LPI|LPO|LPD|LSO)\b", "POS"),
    (r"\b(slow\s+enrollment|enrollment\s+(?:paused|suspended|halted|slowed|delayed|terminated|stopped)|enrollment\s+challenges)\b", "NEG"),

    # Clinical ops (NEW: cohorts, dose-escalation, topline/readout)
    (r"\b(dosing|dose)\s+(?:initiated|has\s+begun|began|started|underway)\b", "POS"),
    (r"\b(?:first|second|third|initial)\s+cohort\s+(?:completed|complete|finished)\b|\bcohort\s*\d+\s+(?:complete|completed)\b", "POS"),
    (r"\b(?:dose[- ]?escalation|dose[- ]?expansion)\s+(?:complete(?:d)?|initiated|started|underway|begun)\b", "POS"),
    (r"\b(?:last\s+(?:patient|subject)\s+(?:dosed|treated)|LPD)\b", "POS"),
    (r"\bIND[- ]enabling\s+studies?\s+(?:complete|completed|finished|concluded)\b", "POS"),
    (r"\b(?:significant|statistically\s+significant)\s+improvement\b", "POS"),
    (r"\b(dosing|dose)\s+(?:paused|suspended|halted|stopped|on\s+hold)\b", "NEG"),
    (r"\bstopped\s+early\s+due\s+to\s+efficacy\b", "POS"),
    (r"\bfutilit(y|ies)\b|\bstopped\s+early\s+for\s+futility\b", "NEG"),

    # Biomarkers / efficacy vs control (NEW: 'compared to')
    (r"\b(?:biomarker[s]?)\s+(?:show(?:ed)?|demonstrat(?:e|ed)|exhibit(?:s|ed))\s+"
     r"(?:greater|significant|higher|lower)\s+(?:increase|reduction|decrease|improvement)\s+"
     r"(?:vs\.?|versus|over|compared\s+to)\s+(?:placebo|control)\b", "POS"),
    (r"\b(?:significant|greater|superior)\s+(?:reduction|increase|improvement)\s+"
     r"(?:vs\.?|versus|over|compared\s+to)\s+(?:placebo|control)\b", "POS"),
    (r"\b(?:no|not)\s+(?:difference|improvement|reduction|increase)\s+"
     r"(?:vs\.?|versus|over|compared\s+to)\s+(?:placebo|control)\b", "NEG"),
    (r"\b(?:similar|comparable)\s+(?:to|vs\.?)\s+(?:placebo|control)\b", "NEG"),
    (r"\b(?:wors(?:e|ened)\s+(?:than|vs\.?)|inferior\s+(?:to|vs\.?))\s+(?:placebo|control)\b", "NEG"),

    # Conferences / abstracts (operationally positive)
    (r"\b(?:data|results|analysis|abstract)\s+(?:to\s+be\s+presented|will\s+be\s+presented|presented|accepted\s+for\s+presentation)\s+"
     r"at\s+(?:ASCO|AACR|ESMO|AASLD|AAO|ARVO|ASH|EULAR|ADA|SABCS|CTAD|AAIC|NANS|BBSW|DIA|WCC|WCLC|SITC|SID|AAO-HNS)\b", "POS"),
    (r"\b(?:data|results|analysis|abstract)\s+(?:to\s+be\s+presented|will\s+be\s+presented|presented|accepted\s+for\s+presentation)\s+"
     r"at\s+(?:a|an)?\s*(?:scientific|medical)\s+(?:conference|meeting|congress)\b", "POS"),
]

OPN_RE = [(re.compile(p, re.I), pol) for p, pol in OPINION_POLARITY]


# -------- Functions --------

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
            text = preprocess_text((str(rec["text"]).replace("…", "...").strip() if pd.notna(rec["text"]) else ""))
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
