#!/usr/bin/env python3
import re
import math
import pandas as pd
from typing import Optional

INPUT_FILE = "data/cleaned_catalysts_metadata.csv"
OUTPUT_FILE = "data/cleaned_catalysts_cleaned_metadata.csv"

# --- helpers ---
_SUFFIX_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}

def parse_market_cap(val) -> Optional[float]:
    """
    Parse Market_Cap values that may include $, commas, and K/M/B/T suffixes.
    Returns a float (USD) or None if invalid/unparseable.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None

    # remove currency symbols/commas/spaces
    s = s.replace("$", "").replace(",", "").strip()

    # If it looks like a pure number, parse directly
    try:
        return float(s)
    except ValueError:
        pass

    # Try suffix form like "1.2B", "350M", "850k", etc.
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([KMBTkm bt])", s)
    if not m:
        # also accept things like "1.2 B" with space
        m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([KMBTkmbt])", s)

    if m:
        num = float(m.group(1))
        suf = m.group(2).upper()
        mult = _SUFFIX_MULTIPLIERS.get(suf, None)
        if mult is None:
            return None
        return num * mult


    # Not a recognized numeric pattern
    return None

def column_like_share_price(col_name: str) -> bool:
    return "share_price" in col_name.lower()

def is_div0(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().upper()
    return "DIV/0" in s  # catches #DIV/0! and variants

# --- main cleaning ---
def main():
    df = pd.read_csv(INPUT_FILE)

    initial_rows = len(df)

    # Identify Share_Price columns (any column name containing "Share_Price", case-insensitive)
    share_price_cols = [c for c in df.columns if column_like_share_price(c)]

    # 1) Clean Market_Cap: coerce to float dollars; drop invalid/<=0
    if "Market_Cap" in df.columns:
        df["Market_Cap_parsed"] = df["Market_Cap"].apply(parse_market_cap)
        # Drop rows where parsed is NaN/None or <= 0
        before_mc = len(df)
        df = df[df["Market_Cap_parsed"].notna() & (df["Market_Cap_parsed"] > 0)]
        after_mc = len(df)
        # If you want to overwrite Market_Cap with parsed numeric:
        df["Market_Cap"] = df["Market_Cap_parsed"]
        df.drop(columns=["Market_Cap_parsed"], inplace=True)
    else:
        before_mc = after_mc = len(df)

    # 2) Drop rows with DIV/0! (or any 'DIV/0') in any Share_Price column
    if share_price_cols:
        # mask for any DIV/0 error-like string
        mask_div0 = df[share_price_cols].applymap(is_div0).any(axis=1)

        # mask for numeric == 0 in any Share_Price column (coerce non-numeric to NaN)
        numeric_sp = df[share_price_cols].apply(pd.to_numeric, errors="coerce")
        mask_zero = numeric_sp.eq(0).any(axis=1)

        # rows to drop if either condition holds
        to_drop_mask = mask_div0 | mask_zero

        before_sp = len(df)
        df = df[~to_drop_mask].copy()
        after_sp = len(df)
    else:
        before_sp = after_sp = len(df)

    # 3) Drop the "Price" column if present
    if "Price" in df.columns:
        df.drop(columns=["Price"], inplace=True)

    # 4) Save
    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print(f"Input rows: {initial_rows}")
    print(f"After Market_Cap filter: {after_mc} (removed {before_mc - after_mc})")
    if share_price_cols:
        print(f"Share_Price columns found: {share_price_cols}")
        print(f"After Share_Price filters: {after_sp} (removed {before_sp - after_sp})")
    else:
        print("No Share_Price columns found.")
    print(f"Output saved → {OUTPUT_FILE}")
    print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()
