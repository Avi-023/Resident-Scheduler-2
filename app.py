python -c "import sys, streamlit; print('PY:', sys.executable); print('ST:', streamlit.__version__)"
# app.py
import io, json, re, math, random, time
from datetime import date, timedelta
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import numpy as np

# --------------------------
# Optional libraries
# --------------------------
try:
    import pyarrow as _pa  # noqa: F401
    HAS_ARROW = True
except Exception:
    HAS_ARROW = False

try:
    import pulp  # Lightweight ILP backend if available
    HAS_PULP = True
except Exception:
    HAS_PULP = False

# --------------------------
# Colors & styling
# --------------------------
PALETTE = {
    "gold":       "#FFD966",
    "red_green":  "#008000",  # Red/Green merged team shown in GREEN
    "vascular":   "#8B0000",  # dark red
    "breast":     "#FFB6C1",  # light pink
    "chief":      "#000080",  # navy
    "icu":        "#FFFF00",  # yellow
    "floor":      "#D2B48C",
    "nights":     "#000000",  # black
    "elective":   "#D9B3FF",
    "pittsburgh": "#FFE599",
    "vacation":   "#B6D7A8",
}
WHITE_TEXT = {"nights", "chief", "vascular", "red_green"}
BOLD_TEXT  = {"icu", "vascular", "red_green"}

def norm_label(v: str) -> str:
    if not isinstance(v, str) or not v.strip():
        return ""
    t = v.strip().lower()
    if t == "gold": return "gold"
    if t in ("red/green","redgreen","rg","red-green"): return "red_green"
    if t.startswith("vasc"): return "vascular"
    if t.startswith("breast"): return "breast"
    if t in ("chief","trauma chief"): return "chief"
    if t in ("icu","trauma icu"): return "icu"
    if t in ("floor","trauma floor"): return "floor"
    if t == "nf" or t.startswith("night"): return "nights"
    if t.startswith("elective"): return "elective"
    if t.startswith("pitts"): return "pittsburgh"
    if t in ("liver","transplant","pancreas","pediatrics","peds"): return "pittsburgh"
    if t in ("vac","vacation","v"): return "vacation"
    return t

def style_cell(label: str) -> dict:
    key = norm_label(label)
    if not key or key not in PALETTE:
        return {}
    css = {"background-color": PALETTE[key]}
    css["color"] = "#FFFFFF" if key in WHITE_TEXT else "#000000"
    css["font-weight"] = "bold" if key in BOLD_TEXT else "normal"
    return css

def style_each_cell(v):
    css = style_cell(v)
    if not css: return ""
    return ";".join(f"{k}: {val}" for k, val in css.items())

# --------------------------
# Helpers
# --------------------------
WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def build_blocks(start: date, n_blocks: int = 13):
    return [(start + timedelta(days=28*i), start + timedelta(days=27 + 28*i)) for i in range(n_blocks)]

def hdr_for_block(s: date, e: date) -> str:
    return f"{s.month}/{s.day} - {e.month}/{e.day}"

def tab_name(s: date, e: date) -> str:
    return hdr_for_block(s, e)

def pgy_of(name, roster_map):
    return roster_map.get(name, {}).get("PGY")

def role_of(name, roster_map):
    p = pgy_of(name, roster_map)
    if p in ("PGY-5","PGY-4"): return "Senior"
    if p in ("PGY-3","PGY-2"): return "Junior"
    return "Intern"

def gini(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[arr>=0]
    if arr.size == 0: return 0.0
    if np.all(arr==0): return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return (n+1 - 2*np.sum(cum)/cum[-1]) / n

# Excel sheet-name sanitizer
_INVALID_EXCEL_CHARS = r'[:\\/?*\[\]]'
def safe_sheet_name(name: str, taken: set) -> str:
    n = re.sub(_INVALID_EXCEL_CHARS, '-', str(name)).strip().strip("'") or "Sheet"
    n = n[:31]
    base, i = n, 1
    while n in taken:
        suf = f" ({i})"
        n = base[:31-len(suf)] + suf
        i += 1
    taken.add(n)
    return n

# --------------------------
# Lite-mode friendly table
# --------------------------
def show_table(df: pd.DataFrame, key: str, *, editable: bool, hide_index: bool=False,
               styler_subset=None, daily=False, index_name_hint="Resident") -> pd.DataFrame:
    if HAS_ARROW:
        if editable:
            return st.data_editor(
                df,
                use_container_width=True,
                key=key,
                hide_index=hide_index,
                num_rows="dynamic"  # enable add/remove rows
            )
        else:
            if styler_subset is not None:
                st.dataframe(df.style.applymap(style_each_cell, subset=styler_subset),
                             use_container_width=True, hide_index=hide_index)
            else:
                st.dataframe(df, use_container_width=True, hide_index=hide_index)
            return df

    # Lite mode CSV fallback
    if editable:
        if daily:
            flat = df.copy()
            flat.columns = [f"{a}|{b}" for (a,b) in flat.columns]
            default_csv = flat.reset_index().to_csv(index=False)
            csv_text = st.text_area(f"CSV editor – {key}", default_csv, height=260, key=f"csv_{key}")
            try:
                tmp = pd.read_csv(io.StringIO(csv_text))
                if index_name_hint in tmp.columns:
                    tmp = tmp.set_index(index_name_hint)
                tuples = []
                for c in tmp.columns:
                    if isinstance(c, str) and "|" in c:
                        a, b = c.split("|", 1)
                        tuples.append((a, b))
                    else:
                        tuples.append((c, ""))
                tmp.columns = pd.MultiIndex.from_tuples(tuples, names=["Weekday","Date"])
                st.caption("Preview"); st.markdown(tmp.to_html(escape=False), unsafe_allow_html=True)
                return tmp
            except Exception as e:
                st.error(f"CSV parse error: {e}")
                st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
                return df
        else:
            default_csv = df.reset_index().to_csv(index=False)
            csv_text = st.text_area(f"CSV editor – {key}", default_csv, height=240, key="csv_"+key)
            try:
                tmp = pd.read_csv(io.StringIO(csv_text))
                if index_name_hint in tmp.columns and (df.index.name in [None, index_name_hint]):
                    tmp = tmp.set_index(index_name_hint)
                st.caption("Preview"); st.markdown(tmp.to_html(escape=False), unsafe_allow_html=True)
                return tmp
            except Exception as e:
                st.error(f"CSV parse error: {e}")
                st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
                return df
    else:
        if styler_subset is not None:
            st.markdown(df.style.applymap(style_each_cell, subset=styler_subset).to_html(), unsafe_allow_html=True)
        else:
            st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
        return df

# --------------------------
# Roster normalization & utils
# --------------------------
def normalize_roster_input(df: pd.DataFrame) -> pd.DataFrame:
    import re as _re
    if df is None or df.empty:
        raise ValueError("Roster is empty.")
    df = df.copy()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Auto-clean++: trim, drop blanks/"none"
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)

    col_norm = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in col_norm.items()}

    def find_col(cands):
        for a in cands:
            if a in inv: return inv[a]
        return None

    resident_col = find_col(["resident","name","resident name"])
    if resident_col is None:
        if df.index.name and str(df.index.name).strip().lower() in {"resident","name","resident name"}:
            df = df.reset_index().rename(columns={df.columns[0]: "Resident"})
        elif df.index.dtype == "object" and df.index.notna().any():
            df = df.reset_index().rename(columns={df.columns[0]: "Resident"})
        else:
            obj_cols = [c for c in df.columns if df[c].dtype == "object"]
            if obj_cols:
                df = df.rename(columns={obj_cols[0]: "Resident"})
    else:
        if resident_col != "Resident":
            df = df.rename(columns={resident_col: "Resident"})

    # Standard columns
    pgy_col = find_col(["pgy","level"])
    if pgy_col and pgy_col != "PGY": df = df.rename(columns={pgy_col: "PGY"})
    prelim_col = find_col(["prelim","preliminary"])
    if prelim_col and prelim_col != "Prelim": df = df.rename(columns={prelim_col: "Prelim"})
    include_col = find_col(["include?","include","active"])
    if include_col and include_col != "Include?": df = df.rename(columns={include_col: "Include?"})

    # Optional attribute columns respected by engine
    for c in ["Max Nights","Avoid Services","PTO Windows","Notes","Lock"]:
        if c not in df.columns: df[c] = ""

    # Ensure required cols
    if "Resident" not in df.columns:
        raise KeyError("Could not locate a 'Resident' column.")
    if "PGY" not in df.columns:
        raise KeyError("Could not locate a 'PGY' column.")
    if "Prelim" not in df.columns: df["Prelim"] = "N"
    if "Include?" not in df.columns: df["Include?"] = "Y"

    # Normalize
    def _norm_pgy(x: str) -> str:
        s = str(x).strip().upper()
        s = _re.sub(r"\(.*?\)", "", s)
        s = s.replace("_"," ").replace("-"," ")
        s = s.replace("YEAR","").replace("PGY","PGY ")
        s = _re.sub(r"\s+"," ", s).strip()
        m = _re.search(r"\b([1-5])\b", s)
        if m: return f"PGY-{m.group(1)}"
        return s

    def _yn(x):
        s = str(x).strip().lower()
        return "Y" if s in {"y","yes","true","1"} else ("N" if s in {"n","no","false","0"} else "Y")

    df["Resident"] = df["Resident"].astype(str)
    df["PGY"]      = df["PGY"].astype(str).map(_norm_pgy)
    df["Prelim"]   = df["Prelim"].astype(str).map(_yn)
    df["Include?"] = df["Include?"].astype(str).map(_yn)

    # Drop blanks/"none"
    bad = df["Resident"].str.strip().str.lower().isin(["", "none", "na", "n/a"])
    df = df[~bad].reset_index(drop=True)

    cols = ["Resident","PGY","Prelim","Include?","Max Nights","Avoid Services","PTO Windows","Notes","Lock"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

def validate_roster(df: pd.DataFrame) -> None:
    bad = []
    if df["Resident"].isna().any() or (df["Resident"].str.strip()=="").any():
        bad.append("Blank resident names")
    if df["Resident"].str.strip().str.lower().eq("none").any():
        bad.append("A row named 'None' (remove it)")
    dups = df["Resident"].str.strip().value_counts()
    dups = dups[dups > 1]
    if not dups.empty:
        bad.append(f"Duplicate names: {', '.join(dups.index.tolist())}")
    bad_pgy = ~df["PGY"].str.match(r"^PGY-[1-5]$")
    if bad_pgy.any():
        bad.append("Non-standard PGY values present (expect PGY-1..PGY-5)")
    if bad:
        raise ValueError("; ".join(bad))

def ensure_roster_size(df: pd.DataFrame, size: int) -> pd.DataFrame:
    df = df.copy()
    size = max(int(size), 1)
    if len(df) < size:
        add = size - len(df)
        add_df = pd.DataFrame({
            "Resident": ["" for _ in range(add)],
            "PGY": ["PGY-1"] * add,
            "Prelim": ["N"] * add,
            "Include?": ["Y"] * add,
            "Max Nights": ["" for _ in range(add)],
            "Avoid Services": ["" for _ in range(add)],
            "PTO Windows": ["" for _ in range(add)],
            "Notes": ["" for _ in range(add)],
            "Lock": ["" for _ in range(add)],
        })
        df = pd.concat([df, add_df], ignore_index=True)
    elif len(df) > size:
        df = df.iloc[:size].reset_index(drop=True)
    return df

def parse_csv_list(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]

# --------------------------
# Reason logging (explainability)
# --------------------------
def log_reason(reasons, res_name, block_hdr, why):
    reasons.append({"Resident": res_name, "Block": block_hdr, "Reason": why})

# --------------------------
# Auto-generate Yearly grid (Greedy)
# --------------------------
def auto_generate_yearly(roster_df: pd.DataFrame, start_date: date, constraints: dict, base_fixed: pd.DataFrame = None):
    roster_df = normalize_roster_input(roster_df)
    roster_df = roster_df[roster_df["Include?"].str.upper().eq("Y")].reset_index(drop=True)

    order_keys = {"PGY-5":0,"PGY-4":1,"PGY-3":2,"PGY-2":3,"PGY-1":4}
    roster_df["ord"] = roster_df["PGY"].map(order_keys)
    roster_df = roster_df.sort_values(["ord","Resident"]).drop(columns=["ord"]).reset_index(drop=True)

    names = list(roster_df["Resident"])
    roster_map = {r["Resident"]: {"PGY": r["PGY"], "Prelim": r.get("Prelim","N")} for _, r in roster_df.iterrows()}

    blocks = build_blocks(start_date, 13)
    headers = [hdr_for_block(s,e) for (s,e) in blocks]
    schedule_df = pd.DataFrame(index=names, columns=headers, dtype=object)

    # carry forward user-fixed entries
    if base_fixed is not None and not base_fixed.empty:
        base_fixed = base_fixed.reindex(index=names, columns=headers)
        mask = base_fixed.notna() & (base_fixed.astype(str).str.strip() != "")
        schedule_df[mask] = base_fixed[mask]

    # Assignment reasons
    reasons = []

    # Helper: previous assignment in prior blocks
    def prev_assignment(name: str, bi: int):
        if bi <= 0: return None
        return schedule_df.loc[name, headers[bi-1]]

    def prev2_assignment(name: str, bi: int):
        if bi <= 1: return None
        return schedule_df.loc[name, headers[bi-2]]

    # Prevent >2 consecutive same rotation
    def exceeds_consecutive_limit(name: str, bi: int, label: str, limit:int=2) -> bool:
        if bi < limit: 
            return False
        last1 = prev_assignment(name, bi)
        last2 = prev2_assignment(name, bi)
        if last1 is None or last2 is None: 
            return False
        return norm_label(last1) == norm_label(label) == norm_label(last2)

    # Helper: count current seniors on a team in this block
    def seniors_on(bi: int, label: str) -> int:
        lbl = label
        return sum(1 for n in names if schedule_df.loc[n, headers[bi]] == lbl and role_of(n, roster_map) == "Senior")

    # PGY-4 Electives (avoid 1/7/13 + holidays)
    if constraints.get("pg4_elective", True):
        avoid = {0,6,12}
        xmas = date(start_date.year,12,21); newyr = date(start_date.year,12,28)
        for i,(s,e) in enumerate(blocks):
            if s<=xmas<=e or s<=newyr<=e: avoid.add(i)
        pgy4s = [n for n in names if roster_map[n]["PGY"]=="PGY-4"]
        slots = [i for i in range(13) if i not in avoid]
        for k,n in enumerate(pgy4s):
            if slots:
                bi = slots[(k*4) % len(slots)]
                if pd.isna(schedule_df.loc[n, headers[bi]]):
                    schedule_df.loc[n, headers[bi]] = "Elective"
                    log_reason(reasons, n, headers[bi], "PGY-4 elective placement avoiding 1/7/13 & holidays")

    # ---------- PGY-3 Pittsburgh (3 consecutive, NON-OVERLAP, Jan 1–15 allowed; Jan 16–31 avoided) ----------
    if constraints.get("pg3_pittsburgh", True):
        avoid = {0,12}  # avoid Block 1 & 13
        january_year = start_date.year + 1

        def block_has_late_january(s: date, e: date) -> bool:
            d = s
            while d <= e:
                if d.year == january_year and d.month == 1 and d.day >= 16:
                    return True
                d += timedelta(days=1)
            return False

        for i,(s,e) in enumerate(blocks):
            if block_has_late_january(s,e):
                avoid.add(i)

        pgy3s = [n for n in names if roster_map[n]["PGY"]=="PGY-3"]
        taken = set()  # blocks already assigned to Pittsburgh (to avoid overlap strictly)

        windows = [(i, i+1, i+2) for i in range(0, 11)]

        def window_ok(w):
            return all(0 <= b < 13 and b not in avoid and b not in taken for b in w)

        for n in pgy3s:
            candidates = [w for w in windows if window_ok(w)]
            if not candidates:
                st.session_state.setdefault("pgh_assign_issues", []).append(
                    ("No non-overlap PGH window available", n)
                )
                continue
            chosen = candidates[0]  # earliest-fit
            for b in chosen:
                if pd.isna(schedule_df.loc[n, headers[b]]):
                    schedule_df.loc[n, headers[b]] = "Pittsburgh"
                    log_reason(reasons, n, headers[b], "PGY-3 Pittsburgh (3× consecutive, non-overlap; Jan 1–15 allowed)")
                taken.add(b)

    # ICU fair counters
    icu_counts = Counter()

    GOLD_MAX = int(constraints.get("gold_cap", 4))
    RG_MAX   = int(constraints.get("rg_cap", 4))
    VASC_MAX = int(constraints.get("vascular_cap", 3))

    # Vascular 3rd-member per-resident cap
    vasc_third_cap = int(constraints.get("vascular_third_cap_per_resident", 1))
    vasc_third_used = Counter()
    vasc_junior_counts = Counter()  # track PGY-2/3 used as vascular juniors

    # Night float tracking
    st.session_state.setdefault("_nf_counts", Counter())
    st.session_state.setdefault("_nf_role_counts", defaultdict(Counter))  # name -> {"Senior":x,"Junior":y,"Intern":z}
    st.session_state.setdefault("_last_nf", {})
    st.session_state.setdefault("nf_overrides", [])
    nf_counts = st.session_state["_nf_counts"]
    nf_role_counts = st.session_state["_nf_role_counts"]
    last_nf   = st.session_state["_last_nf"]
    nf_log    = st.session_state["nf_overrides"]

    MIN_NF_GAP_BLOCKS = 1

    # Service minima
    svc_min = constraints.get("service_minima", {"Gold":3, "Red/Green":3, "ICU":1})

    # Eligibility calendars (JSON spec)
    svc_unavail = constraints.get("eligibility_calendars", {})  # {"ICU":{"PGY-5":[0,1,...], "names": {"Jane Doe":[3]}} ...}

    def is_unavailable(name, pgy, service_label, bi):
        key = service_label
        spec = svc_unavail.get(key, {})
        blist_pgy = spec.get(pgy, [])
        if bi in set(blist_pgy):
            return True
        byname = spec.get("names", {})
        if name in byname and bi in set(byname[name]):
            return True
        # PTO windows
        try:
            pto = next((x for x in roster_df.itertuples(index=False) if x.Resident==name), None)
            if pto and getattr(pto, "PTO Windows", ""):
                blocks_list = []
                for tok in parse_csv_list(getattr(pto, "PTO Windows")):
                    try: blocks_list.append(int(tok))
                    except Exception: pass
                if bi in set(blocks_list) and key not in {"Nights","Weekend Call"}:
                    return True
        except Exception:
            pass
        # Avoid Services (free-text list)
        try:
            row = next((x for x in roster_df.itertuples(index=False) if x.Resident==name), None)
            if row and getattr(row, "Avoid Services", ""):
                avoid_set = set(norm_label(s) for s in parse_csv_list(getattr(row, "Avoid Services")))
                if norm_label(key) in avoid_set:
                    return True
        except Exception:
            pass
        return False

    # Seniority helpers
    def is_senior(n): return role_of(n, roster_map) == "Senior"
    def is_junior(n): return role_of(n, roster_map) == "Junior"
    def is_intern(n): return roster_map[n]["PGY"] == "PGY-1"

    # prefer candidates who do NOT repeat last block on same service
    def repeat_penalty(name: str, bi: int, label: str) -> int:
        last = prev_assignment(name, bi)
        return 1 if (isinstance(last, str) and norm_label(last) == norm_label(label)) else 0

    def would_violate_senior_cap(name: str, bi: int, label: str) -> bool:
        return is_senior(name) and seniors_on(bi, label) >= 2

    def would_exceed_run(name: str, bi: int, label: str) -> bool:
        return exceeds_consecutive_limit(name, bi, label, limit=2)

    def assign_first(pool, bi, label):
        """Fewest prior on that label; avoid consecutive repeat; forbid >2-run; respect senior<=2; returns pick or None."""
        hdr = headers[bi]
        cand=[n for n in pool if pd.isna(schedule_df.loc[n, hdr])
              and not is_unavailable(n, roster_map[n]["PGY"], label, bi)
              and not would_violate_senior_cap(n, bi, label)
              and not would_exceed_run(n, bi, label)]
        if not cand: return None
        counts={n:int((schedule_df.loc[n]==label).sum()) for n in cand}
        pick=sorted(cand, key=lambda n:(counts[n], repeat_penalty(n, bi, label), n))[0]
        schedule_df.loc[pick, hdr]=label
        log_reason(reasons, pick, hdr, f"Assigned to {label} (min prior; avoid consecutive; no >2-run)")
        return pick

    def team(bi,label): return [n for n in names if schedule_df.loc[n, headers[bi]]==label]

    def _nf_gap_ok(name: str, bi_: int, min_gap=MIN_NF_GAP_BLOCKS) -> bool:
        last = last_nf.get(name, None)
        return (last is None) or ((bi_ - last) >= (min_gap + 1))

    def _is_pgy5(name: str) -> int:
        return 1 if roster_map.get(name, {}).get("PGY") == "PGY-5" else 0

    rng = random.Random(int(constraints.get("random_seed", 0) or 0))

    # ---------------- Per-block assignment ----------------
    for bi in range(13):
        hdr = headers[bi]

        # Locked rows: skip assigning any locked rows (keep base_fixed or prior assignment)
        locked_rows = set()
        for r in roster_df.itertuples(index=False):
            if getattr(r, "Lock", "").strip().lower() in {"row","true","y","yes"}:
                locked_rows.add(r.Resident)

        # Chief: H1 PGY-3; H2 PGY-2
        if bi<6:
            chief=assign_first([n for n in names if roster_map[n]["PGY"]=="PGY-3" and n not in locked_rows], bi,"Chief")
            if chief is None:
                assign_first([n for n in names if roster_map[n]["PGY"]=="PGY-2" and n not in locked_rows], bi,"Chief")
        else:
            chief=assign_first([n for n in names if roster_map[n]["PGY"]=="PGY-2" and n not in locked_rows], bi,"Chief")
            if chief is None:
                assign_first([n for n in names if roster_map[n]["PGY"]=="PGY-3" and n not in locked_rows], bi,"Chief")

        # ICU (HARD): H1 PGY-2 only; H2 PGY-1 only; fair distribution; must be covered
        if bi < 6:
            elig = [n for n in names if roster_map[n]["PGY"]=="PGY-2" and pd.isna(schedule_df.loc[n, hdr])
                    and not is_unavailable(n, "PGY-2", "ICU", bi) and n not in locked_rows
                    and not would_exceed_run(n, bi, "ICU")]
            if elig:
                pick = sorted(elig, key=lambda n:(icu_counts[n], repeat_penalty(n, bi, "ICU"), n))[0]
                schedule_df.loc[pick, hdr] = "ICU"; icu_counts[pick] += 1
                log_reason(reasons, pick, hdr, "ICU (H1 PGY-2 only) fair distribution (lowest ICU count; avoid consecutive)")
        else:
            elig = [n for n in names if roster_map[n]["PGY"]=="PGY-1" and pd.isna(schedule_df.loc[n, hdr])
                    and not is_unavailable(n, "PGY-1", "ICU", bi) and n not in locked_rows
                    and not would_exceed_run(n, bi, "ICU")]
            if elig:
                pick = sorted(elig, key=lambda n:(icu_counts[n], repeat_penalty(n, bi, "ICU"), n))[0]
                schedule_df.loc[pick, hdr] = "ICU"; icu_counts[pick] += 1
                log_reason(reasons, pick, hdr, "ICU (H2 PGY-1 only) fair distribution (lowest ICU count; avoid consecutive)")

        # Night Float (HARD): must exist every block; Sr + Jr + Intern; ≥1-block gap preferred; avoid PGY-5 senior in last block
        def _pick_night_role(role: str, relax_gap: bool, avoid_pgy5_last: bool=False):
            pool = [n for n in names if pd.isna(schedule_df.loc[n, hdr]) and n not in locked_rows]
            pool = [n for n in pool if not is_unavailable(n, roster_map[n]["PGY"], "Nights", bi)
                    and not would_exceed_run(n, bi, "Nights")]
            if role == "Senior":
                pool = [n for n in pool if is_senior(n)]
            elif role == "Junior":
                pool = [n for n in pool if is_junior(n)]
            else:
                pool = [n for n in pool if is_intern(n)]
            if not relax_gap:
                pool = [n for n in pool if _nf_gap_ok(n, bi)]
            if not pool:
                return None
            if role == "Senior":
                keyf = lambda n: (nf_role_counts[n]["Senior"], nf_counts[n],
                                  repeat_penalty(n, bi, "Nights"),
                                  (_is_pgy5(n) if (avoid_pgy5_last and bi==12) else 0), n)
            else:
                keyf = lambda n: (nf_role_counts[n][role], nf_counts[n], repeat_penalty(n, bi, "Nights"), n)
            pool.sort(key=keyf)
            return pool[0]

        senior = _pick_night_role("Senior", relax_gap=False, avoid_pgy5_last=True)
        junior = _pick_night_role("Junior", relax_gap=False)
        intern = _pick_night_role("Intern", relax_gap=False)

        if senior is None:
            senior = _pick_night_role("Senior", relax_gap=True, avoid_pgy5_last=True)
            if senior and not _nf_gap_ok(senior, bi):
                st.session_state["nf_overrides"].append(("NF gap override (Senior)", hdr, f"{senior} assigned despite <{MIN_NF_GAP_BLOCKS+1}-block gap"))
        if junior is None:
            junior = _pick_night_role("Junior", relax_gap=True)
            if junior and not _nf_gap_ok(junior, bi):
                st.session_state["nf_overrides"].append(("NF gap override (Junior)", hdr, f"{junior} assigned despite <{MIN_NF_GAP_BLOCKS+1}-block gap"))
        if intern is None:
            intern = _pick_night_role("Intern", relax_gap=True)
            if intern and not _nf_gap_ok(intern, bi):
                st.session_state["nf_overrides"].append(("NF gap override (Intern)", hdr, f"{intern} assigned despite <{MIN_NF_GAP_BLOCKS+1}-block gap"))

        if senior and junior and intern:
            for role, pick in (("Senior", senior), ("Junior", junior), ("Intern", intern)):
                schedule_df.loc[pick, hdr] = "Nights"
                nf_counts[pick] += 1; last_nf[pick] = bi; nf_role_counts[pick][role] += 1
                log_reason(reasons, pick, hdr, f"Nights as {role} (role-fair; spacing respected; avoid consecutive)")
            if bi == 12 and _is_pgy5(senior):
                st.session_state["nf_overrides"].append(("NF last-block PGY-5 (avoidance failed)", hdr, f"{senior} is PGY-5 on Nights"))
        else:
            missing = []
            if not senior: missing.append("Senior")
            if not junior: missing.append("Junior")
            if not intern: missing.append("Intern")
            st.session_state["nf_overrides"].append(("NF hard failure", hdr, f"Missing roles: {', '.join(missing)}"))

        # Floor (HARD): PGY-1 only
        assign_first([n for n in names if roster_map[n]["PGY"]=="PGY-1" and n not in locked_rows], bi, "Floor")

        # --------- Vascular baseline & junior inclusion BEFORE RG seeding ---------
        # Hard: must include an Intern; Also ensure Senior present; then try to add a PGY-2/3 as 3rd if cap allows.
        if constraints.get("vascular_sr_jr", True):
            # Senior
            assign_first([n for n in names if is_senior(n) and n not in locked_rows], bi, "Vascular")
            # Intern (hard)
            assigned_intern = assign_first([n for n in names if is_intern(n) and n not in locked_rows], bi, "Vascular")

        # Try adding a PGY-2/3 as 3rd on Vascular (fairly) if cap allows
        def count_team(lbl): return int((schedule_df[hdr]==lbl).sum())
        if count_team("Vascular") >= 2 and count_team("Vascular") < VASC_MAX:
            avail_for_vasc = [n for n in names if pd.isna(schedule_df.loc[n, hdr])
                              and n not in locked_rows
                              and is_junior(n)  # PGY-2/3
                              and not is_unavailable(n, roster_map[n]["PGY"], "Vascular", bi)
                              and not would_violate_senior_cap(n, bi, "Vascular")
                              and not would_exceed_run(n, bi, "Vascular")]
            if avail_for_vasc:
                pick = sorted(avail_for_vasc, key=lambda n:(vasc_junior_counts[n],
                                                            int((schedule_df.loc[n]=="Vascular").sum()),
                                                            repeat_penalty(n, bi, "Vascular"),
                                                            n))[0]
                schedule_df.loc[pick, hdr] = "Vascular"
                vasc_junior_counts[pick] += 1
                log_reason(reasons, pick, hdr, "Vascular 3rd member (PGY-2/3 fairness; before RG seeding)")

        # Breast (PGY-2+)
        if constraints.get("breast_pgy2plus", True):
            assign_first([n for n in names if roster_map[n]["PGY"] in {"PGY-2","PGY-3","PGY-4","PGY-5"} and n not in locked_rows], bi, "Breast")

        # Pre-seed Gold Senior
        if constraints.get("senior_on_gold", True):
            assign_first([n for n in names if is_senior(n) and n not in locked_rows], bi, "Gold")

        # Red/Green composition minima AFTER vascular junior attempt
        if constraints.get("rg_require_sji", True):
            assign_first([n for n in names if is_senior(n) and n not in locked_rows], bi, "Red/Green")
            assign_first([n for n in names if is_junior(n) and n not in locked_rows], bi, "Red/Green")
            assign_first([n for n in names if is_intern(n) and n not in locked_rows], bi, "Red/Green")

        # Priority fill with caps + service minima
        avail=[n for n in names if pd.isna(schedule_df.loc[n, hdr]) and n not in locked_rows]

        def fill_to(lbl,tgt,cap=None):
            nonlocal avail
            while count_team(lbl) < tgt and (cap is None or count_team(lbl) < cap):
                cand = [x for x in avail
                        if not is_unavailable(x, roster_map[x]["PGY"], lbl, bi)
                        and not would_violate_senior_cap(x, bi, lbl)
                        and not would_exceed_run(x, bi, lbl)]
                if not cand: break
                pick=sorted(cand, key=lambda n:(int((schedule_df.loc[n]==lbl).sum()),
                                                repeat_penalty(n, bi, lbl),
                                                rng.random(), n))[0]
                schedule_df.loc[pick, hdr]=lbl; avail.remove(pick)
                log_reason(reasons, pick, hdr, f"Filled {lbl} to meet target/min/cap (avoid consecutive; no >2-run)")

        # Minima first
        fill_to("Gold", max(svc_min.get("Gold",3), 0), cap=GOLD_MAX)
        fill_to("Red/Green", max(svc_min.get("Red/Green",3), 0), cap=RG_MAX)
        fill_to("ICU", max(svc_min.get("ICU",1), 0), cap=1)

        # Remaining priority
        if constraints.get("priority_gold_rg", True):
            fill_to("Gold",4,cap=GOLD_MAX)
            if constraints.get("rg_allow_fourth", True):
                fill_to("Red/Green",4,cap=RG_MAX)
        else:
            fill_to("Red/Green",4 if constraints.get("rg_allow_fourth", True) else 3, cap=RG_MAX)
            fill_to("Gold",4,cap=GOLD_MAX)

        # If Vascular still has capacity, attempt another PGY-2/3 (rare) respecting per-resident 3rd-cap
        if 2 <= count_team("Vascular") < VASC_MAX:
            avail=[n for n in names if pd.isna(schedule_df.loc[n, hdr])
                   and n not in locked_rows
                   and not is_unavailable(n, roster_map[n]["PGY"], "Vascular", bi)
                   and not would_violate_senior_cap(n, bi, "Vascular")
                   and not would_exceed_run(n, bi, "Vascular")]
            if avail:
                def vasc_third_rank(name: str):
                    p = roster_map.get(name, {}).get("PGY")
                    tier = 0 if p in {"PGY-2","PGY-3"} else (1 if p=="PGY-1" else 2)  # still prefer 2/3, then 1, then 4/5
                    prior = int((schedule_df.loc[name] == "Vascular").sum())
                    cap_ok = 0 if vasc_third_used[name] < vasc_third_cap else 1
                    rep = repeat_penalty(name, bi, "Vascular")
                    return (cap_ok, tier, rep, prior, name)
                cand = sorted(avail, key=vasc_third_rank)[0]
                schedule_df.loc[cand, hdr] = "Vascular"
                vasc_third_used[cand] += 1
                log_reason(reasons, cand, hdr, "Vascular extra capacity (prefer PGY-2/3, then PGY-1; avoid 4/5)")

        # HARD: everyone must have assignment — safety fill
        avail=[n for n in names if pd.isna(schedule_df.loc[n, hdr])]
        tries = 0
        while avail and tries < 200:
            n = avail.pop(0)
            # pick eligible destination prioritizing caps and avoiding consecutive & >2-run
            choices = []
            for lbl, cap in (("Vascular", VASC_MAX), ("Gold", GOLD_MAX), ("Red/Green", RG_MAX), ("Breast", None), ("Floor", None)):
                if is_unavailable(n, roster_map[n]["PGY"], lbl, bi): continue
                if lbl=="Floor" and roster_map[n]["PGY"]!="PGY-1": continue
                if cap is not None and count_team(lbl) >= cap: continue
                if would_violate_senior_cap(n, bi, lbl): continue
                if would_exceed_run(n, bi, lbl): continue
                choices.append(lbl)
            if not choices:
                # fallback: allow breaking >2-run rule as last resort but still avoid illegal Floor
                choices = ["Breast","Gold","Red/Green","Vascular"] if roster_map[n]["PGY"]!="PGY-1" else ["Floor","Vascular","Red/Green","Gold"]
            choices_sorted = sorted(choices, key=lambda lbl: (repeat_penalty(n, bi, lbl), lbl))
            lbl = choices_sorted[0]
            schedule_df.loc[n, hdr] = lbl
            log_reason(reasons, n, hdr, f"Safety fill → {lbl} (hard: all residents assigned; minimal repeat)")

            # refresh avail list
            avail=[x for x in names if pd.isna(schedule_df.loc[x, hdr])]
            tries += 1

        # Vascular intern hard check with swap if needed
        def vascular_has_intern() -> bool:
            return any(is_intern(n) for n in team(bi,"Vascular"))
        if not vascular_has_intern():
            vas = team(bi,"Vascular")
            jr_on_vas = [n for n in vas if is_junior(n)]
            if jr_on_vas:
                intern_pool = [n for n in names if is_intern(n) and schedule_df.loc[n, hdr] in {"Gold","Red/Green","Breast","Floor"}]
                if intern_pool:
                    give = intern_pool[0]
                    take = jr_on_vas[0]
                    old_lbl = schedule_df.loc[give, hdr]
                    # avoid >2-run for both sides
                    if not exceeds_consecutive_limit(give, bi, "Vascular") and not exceeds_consecutive_limit(take, bi, old_lbl):
                        schedule_df.loc[give, hdr] = "Vascular"
                        schedule_df.loc[take, hdr] = old_lbl
                        log_reason(reasons, give, hdr, f"Swap to ensure Vascular has an intern (from {old_lbl})")
                        log_reason(reasons, take, hdr, f"Swap off Vascular to {old_lbl} (make room for intern)")

    # persist trackers
    st.session_state["_nf_counts"] = nf_counts
    st.session_state["_nf_role_counts"] = nf_role_counts
    st.session_state["_last_nf"]   = last_nf
    st.session_state["assign_reasons"] = reasons

    return schedule_df

# --------------------------
# Optional Optimizer (ILP polish) using PuLP
# --------------------------
def polish_with_optimizer(schedule_df: pd.DataFrame, roster_df: pd.DataFrame, start_date: date, constraints: dict, time_limit_s: int = 8):
    if not HAS_PULP:
        st.info("Optimizer not available (PuLP not installed). Using greedy schedule.")
        return schedule_df

    names = list(schedule_df.index)
    headers = list(schedule_df.columns)
    roster = {r["Resident"]: r for _, r in normalize_roster_input(roster_df).iterrows()}

    GOLD_MAX = int(constraints.get("gold_cap", 4))
    RG_MAX   = int(constraints.get("rg_cap", 4))
    VASC_MAX = int(constraints.get("vascular_cap", 3))

    # penalties
    w_nf_gap      = float(constraints.get("penalty_nf_gap", 3.0))
    w_pgy5_last   = float(constraints.get("penalty_pgy5_last", 5.0))
    w_gold_sen    = float(constraints.get("penalty_gold_senior", 3.0))
    w_repeat_svc  = float(constraints.get("penalty_consecutive_repeat", 2.0))

    SVC = ["Gold","Red/Green","Vascular","Breast","Chief","ICU","Floor","Nights","Elective","Pittsburgh","Vacation"]

    prob = pulp.LpProblem("Polish", pulp.LpMinimize)

    x = {}
    movable = {"Gold","Red/Green","Vascular","Breast","Floor"}
    for n in names:
        for bi, hdr in enumerate(headers):
            orig = str(schedule_df.loc[n, hdr])
            for s in SVC:
                if orig in {"Chief","ICU","Nights","Pittsburgh","Elective","Vacation"}:
                    val = 1 if s == orig else 0
                    x[(n,bi,s)] = pulp.LpVariable(f"x_{n}_{bi}_{s}", lowBound=val, upBound=val, cat="Binary")
                else:
                    if s in movable:
                        x[(n,bi,s)] = pulp.LpVariable(f"x_{n}_{bi}_{s}", lowBound=0, upBound=1, cat="Binary")
                    else:
                        val = 1 if s == orig else 0
                        x[(n,bi,s)] = pulp.LpVariable(f"x_{n}_{bi}_{s}", lowBound=val, upBound=val, cat="Binary")

    # Each (n,b) exactly one service
    for n in names:
        for bi, hdr in enumerate(headers):
            prob += pulp.lpSum(x[(n,bi,s)] for s in SVC) == 1

    # Team caps within blocks
    for bi, hdr in enumerate(headers):
        prob += pulp.lpSum(x[(n,bi,"Gold")] for n in names) <= GOLD_MAX
        prob += pulp.lpSum(x[(n,bi,"Red/Green")] for n in names) <= RG_MAX
        prob += pulp.lpSum(x[(n,bi,"Vascular")] for n in names) <= VASC_MAX
        prob += pulp.lpSum(x[(n,bi,"Red/Green")] for n in names) >= 3

        # Senior cap ≤2 per service (across all services)
        seniors = [n for n in names if role_of(n, roster)=="Senior"]
        for svc in ["Gold","Red/Green","Vascular","Breast","ICU","Floor","Nights","Pittsburgh","Elective","Vacation"]:
            prob += pulp.lpSum(x[(n,bi,svc)] for n in seniors) <= 2

        # Vascular must include at least one intern
        interns = [n for n in names if roster[n]["PGY"]=="PGY-1"]
        if interns:
            prob += pulp.lpSum(x[(n,bi,"Vascular")] for n in interns) >= 1

    # Objective: soft penalties
    obj = 0

    # Penalty if Gold lacks a Senior
    for bi, hdr in enumerate(headers):
        seniors = [n for n in names if role_of(n, roster) == "Senior"]
        gold_sen_count = pulp.lpSum(x[(n,bi,"Gold")] for n in seniors)
        b_gs = pulp.LpVariable(f"miss_golds_{bi}", lowBound=0, upBound=1, cat="Binary")
        prob += gold_sen_count + b_gs >= 1
        obj += w_gold_sen * b_gs

    # Penalty: PGY-5 on nights in last block
    bi_last = len(headers)-1
    pgy5s = [n for n in names if roster[n]["PGY"]=="PGY-5"]
    if pgy5s:
        obj += w_pgy5_last * pulp.lpSum(x[(n,bi_last,"Nights")] for n in pgy5s)

    # Penalty: consecutive repeats (including >2 runs)
    y = {}
    for n in names:
        for bi in range(1, len(headers)):
            for s in SVC:
                y[(n,bi,s)] = pulp.LpVariable(f"y_rep_{n}_{bi}_{s}", lowBound=0, upBound=1, cat="Binary")
                prob += y[(n,bi,s)] <= x[(n,bi-1,s)]
                prob += y[(n,bi,s)] <= x[(n,bi,s)]
                prob += y[(n,bi,s)] >= x[(n,bi-1,s)] + x[(n,bi,s)] - 1
                obj += w_repeat_svc * y[(n,bi,s)]

    prob += obj

    try:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=max(1,int(time_limit_s)))
        prob.solve(solver)
    except Exception as e:
        st.warning(f"Optimizer error: {e}")
        return schedule_df

    out = schedule_df.copy()
    for n in names:
        for bi, hdr in enumerate(headers):
            for s in SVC:
                if pulp.value(x[(n,bi,s)]) >= 0.5:
                    out.loc[n, hdr] = s
                    break
    return out

# --------------------------
# Build Daily tabs
# --------------------------
def build_dailies_from_yearly(schedule_df: pd.DataFrame, start_date: date):
    blocks = build_blocks(start_date, 13)
    headers = [hdr_for_block(s,e) for (s,e) in blocks]
    names = list(schedule_df.index)
    dailies={}
    for bi,(s,e) in enumerate(blocks):
        cols=[]; d=s
        for _ in range(4):
            for wd in WEEKDAYS:
                cols.append((wd, d.strftime("%m/%d/%Y"))); d += timedelta(days=1)
        cols=pd.MultiIndex.from_tuples(cols, names=["Weekday","Date"])
        daily=pd.DataFrame(index=names, columns=cols, dtype=object)
        for n in names:
            base=schedule_df.loc[n, headers[bi]]
            daily.loc[n,:]=base
        wknd = daily.columns.get_level_values(0).isin(["Saturday","Sunday"])
        daily.loc[:, wknd] = ""  # weekends uncolored/empty except call text
        dailies[tab_name(s,e)] = daily
    return dailies

# --------------------------
# Weekend call assignment
# --------------------------
def auto_assign_weekend_call(dailies: dict, schedule_df: pd.DataFrame, start_date: date, constraints: dict, roster_df: pd.DataFrame):
    r = normalize_roster_input(roster_df)
    r = r[r["Include?"].str.upper().eq("Y")].reset_index(drop=True)
    roster_map = {row["Resident"]: {"PGY": row["PGY"]} for _,row in r.iterrows()}
    names = [n for n in schedule_df.index if n in roster_map]
    def _role(n): return role_of(n, roster_map)

    blocks = build_blocks(start_date, 13)
    headers=[hdr_for_block(s,e) for (s,e) in blocks]
    excluded_base=set(map(norm_label, constraints.get("exclude_from_call", [])))

    call_year=Counter()
    issues = st.session_state.setdefault("forced_call_issues", [])

    last_global_wknd = defaultdict(lambda: -999)
    def global_week_index(bi, wk): return bi*4 + wk

    def pick(pool, need, block_ctr, bi, wk):
        pool = [n for n in pool if block_ctr[n]<2]
        pool.sort(key=lambda n:(call_year[n], -(global_week_index(bi,wk) - last_global_wknd[n]), n))
        for n in pool:
            if need == "Senior" and _role(n)=="Senior": return n
            if need == "Junior" and _role(n) in {"Junior","Senior"}: return n
            if need == "Intern" and _role(n) in {"Intern","Junior","Senior"}: return n
        return None

    for bi,(s,e) in enumerate(blocks):
        hdr=headers[bi]; daily=dailies[tab_name(s,e)]
        sat_cols=[c for c in daily.columns if c[0]=="Saturday"]
        sun_cols=[c for c in daily.columns if c[0]=="Sunday"]
        block_ctr=Counter()

        for wk in range(4):
            scol, sucol = sat_cols[wk], sun_cols[wk]
            base_pool=[n for n in names if norm_label(schedule_df.loc[n, hdr]) not in excluded_base]
            used=set()

            # F/Su
            team=[]
            for need in ("Senior","Junior","Intern"):
                p=pick([n for n in base_pool if n not in used], need, block_ctr, bi, wk)
                if p: team.append(p); used.add(p)
            if len(team)<3:
                pool_any=[n for n in names if n not in used and block_ctr[n]<2]
                while len(team)<3 and pool_any:
                    need=("Senior","Junior","Intern")[len(team)]
                    p=pick(pool_any, need, block_ctr, bi, wk)
                    if not p: break
                    team.append(p); used.add(p); pool_any.remove(p)
                    issues.append(("Forced override", hdr, f"Week {wk+1}: {p} → F/Su to ensure coverage"))
            for n in team:
                daily.loc[n, [sucol]]="F/Su"; block_ctr[n]+=1; call_year[n]+=1; last_global_wknd[n]=global_week_index(bi,wk)
            if len(team)<3:
                issues.append(("Call coverage shortfall", hdr, f"Week {wk+1}: F/Su {len(team)}/3"))

            # Sa
            team=[]
            for need in ("Senior","Junior","Intern"):
                p=pick([n for n in base_pool if n not in used], need, block_ctr, bi, wk)
                if p: team.append(p); used.add(p)
            if len(team)<3:
                pool_any=[n for n in names if n not in used and block_ctr[n]<2]
                while len(team)<3 and pool_any:
                    need=("Senior","Junior","Intern")[len(team)]
                    p=pick(pool_any, need, block_ctr, bi, wk)
                    if not p: break
                    team.append(p); used.add(p); pool_any.remove(p)
                    issues.append(("Forced override", hdr, f"Week {wk+1}: {p} → Sa to ensure coverage"))
            for n in team:
                daily.loc[n, [scol]]="Sa"; block_ctr[n]+=1; call_year[n]+=1; last_global_wknd[n]=global_week_index(bi,wk)
            if len(team)<3:
                issues.append(("Call coverage shortfall", hdr, f"Week {wk+1}: Sa {len(team)}/3"))

# --------------------------
# Effective yearly from dailies
# --------------------------
def effective_yearly_from_dailies(dailies: dict, start_date: date, index_order: list):
    blocks = build_blocks(start_date, 13)
    headers=[hdr_for_block(s,e) for (s,e) in blocks]
    eff=pd.DataFrame(index=index_order, columns=headers, dtype=object)
    for bi,(s,e) in enumerate(blocks):
        name=tab_name(s,e); df=dailies[name]
        wk_cols=[c for c in df.columns if c[0] in ["Monday","Tuesday","Wednesday","Thursday","Friday"]]
        for n in df.index:
            vals=[v for v in df.loc[n, wk_cols].values if isinstance(v,str) and v!=""]
            eff.loc[n, headers[bi]] = (pd.Series(vals).mode().iat[0] if vals else "")
    return eff

# --------------------------
# Checks with severity, codes, heatmap data
# --------------------------
def compute_checks(schedule_df: pd.DataFrame, dailies: dict, roster_df: pd.DataFrame, constraints: dict, start_date: date):
    if schedule_df is None or schedule_df.empty:
        empty = pd.DataFrame()
        return {"issues": empty, "summary": empty, "coverage": empty,
                "call": empty, "rot_counts": empty, "pgy_coverage": empty,
                "special_counts": empty, "svc_variance": empty}

    r = normalize_roster_input(roster_df)
    r = r[r["Include?"].str.upper().eq("Y")].reset_index(drop=True)
    order_keys={"PGY-5":0,"PGY-4":1,"PGY-3":2,"PGY-2":3,"PGY-1":4}
    r["ord"]=r["PGY"].map(order_keys); r=r.sort_values(["ord","Resident"]).drop(columns=["ord"]).reset_index(drop=True)
    roster_map={row["Resident"]: {"PGY": row["PGY"]} for _,row in r.iterrows()}

    names=[n for n in schedule_df.index if n in roster_map]
    headers=list(schedule_df.columns)
    blocks=build_blocks(start_date, 13)
    hdr_to_bi={hdr_for_block(s,e):i for i,(s,e) in enumerate(blocks)}
    svc_labels=["Gold","Red/Green","Vascular","Breast","Chief","ICU","Floor","Nights","Elective","Pittsburgh","Vacation"]

    issues=[]

    def add_issue(code, severity, hdr, msg):
        issues.append({"Code": code, "Severity": severity, "Block": hdr, "Details": msg})

    # Unassigned (ERROR)
    for hdr in headers:
        unassigned = [n for n in names if (pd.isna(schedule_df.loc[n, hdr]) or str(schedule_df.loc[n, hdr]).strip()=="")]
        for n in unassigned:
            add_issue("UNASSIGNED", "ERROR", hdr, f"{n} has no assignment")

    # Pittsburgh overlap & January (late) exclusion
    january_year = start_date.year + 1
    def block_has_late_january(s: date, e: date) -> bool:
        d = s
        while d <= e:
            if d.year == january_year and d.month == 1 and d.day >= 16:
                return True
            d += timedelta(days=1)
        return False

    for idx,(s,e) in enumerate(blocks):
        hdr = headers[idx]
        pgh_count = int((schedule_df[hdr]=="Pittsburgh").sum())
        if pgh_count > 1:
            add_issue("PGH_OVERLAP", "ERROR", hdr, f"{pgh_count} residents in Pittsburgh (should be ≤ 1)")
        if block_has_late_january(s,e) and pgh_count>0:
            add_issue("PGH_JAN_LATE", "ERROR", hdr, "Pittsburgh scheduled in late January (Jan 16–31) not allowed")

    # Per-block composition / caps / senior cap and vascular-intern rules
    for hdr in headers:
        bi=hdr_to_bi.get(hdr,0)
        def svc(lbl): return [n for n in names if norm_label(schedule_df.loc[n, hdr])==norm_label(lbl)]
        gold      = svc("Gold")
        rg        = svc("Red/Green")
        breast    = svc("Breast")
        vas       = svc("Vascular")
        chief     = svc("Chief")
        icu       = svc("ICU")
        floor     = svc("Floor")
        nights    = svc("Nights")

        # Nights existence & composition (ERROR)
        roles = [role_of(n, roster_map) for n in nights]
        if len(nights) < 3:
            add_issue("NF_COVERAGE", "ERROR", hdr, f"Nights team size {len(nights)} (<3)")
        else:
            if Counter(roles) != Counter({"Senior":1,"Junior":1,"Intern":1}):
                add_issue("NF_COMPOSITION", "ERROR", hdr, f"Expected 1 Senior, 1 Junior, 1 Intern; got {Counter(roles)}")
        if bi == 12 and any(roster_map[n]["PGY"]=="PGY-5" for n in nights):
            add_issue("NF_LASTBLOCK_PGY5", "WARN", hdr, "PGY-5 on Nights in last block (avoid)")

        # Chief eligibility
        if bi<6:
            for n in chief:
                if roster_map[n]["PGY"]=="PGY-2":
                    add_issue("CHIEF_H1_PGY2", "WARN", hdr, f"PGY-2 Chief: {n}")
        else:
            for n in chief:
                if roster_map[n]["PGY"]=="PGY-3":
                    add_issue("CHIEF_H2_PGY3", "WARN", hdr, f"PGY-3 Chief (fallback only): {n}")

        # ICU (ERROR)
        if bi<6:
            if not icu:
                add_issue("ICU_H1_NOCOVER", "ERROR", hdr, "ICU not covered by PGY-2 (required)")
            for n in icu:
                if roster_map[n]["PGY"]!="PGY-2":
                    add_issue("ICU_H1_BADPGY", "ERROR", hdr, f"{n} ({roster_map[n]['PGY']}) on ICU (PGY-2 only)")
        else:
            if not icu:
                add_issue("ICU_H2_NOCOVER", "ERROR", hdr, "ICU not covered by PGY-1 (required)")
            for n in icu:
                if roster_map[n]["PGY"]!="PGY-1":
                    add_issue("ICU_H2_BADPGY", "ERROR", hdr, f"{n} ({roster_map[n]['PGY']}) on ICU (PGY-1 only)")

        # Floor (ERROR if non-intern)
        for n in floor:
            if roster_map[n]["PGY"]!="PGY-1":
                add_issue("FLOOR_BADPGY", "ERROR", hdr, f"{n} ({roster_map[n]['PGY']}) on Floor (PGY-1 only)")

        # Senior-per-service cap (ERROR if >2)
        for svc_name, team_ in (("Gold",gold),("Red/Green",rg),("Vascular",vas),("Breast",breast),("ICU",icu),("Floor",floor),("Nights",nights),("Pittsburgh",svc("Pittsburgh")),("Elective",svc("Elective")),("Vacation",svc("Vacation"))):
            senior_ct = sum(1 for n in team_ if role_of(n, roster_map)=="Senior")
            if senior_ct > 2:
                add_issue("SR_CAP", "ERROR", hdr, f"{svc_name} has {senior_ct} seniors (>2)")

        # Gold cap & senior presence
        if len(gold) > int(constraints.get("gold_cap",4)):
            add_issue("GOLD_CAP", "WARN", hdr, f"Gold has {len(gold)} (> cap)")
        if gold and not any(role_of(n, roster_map)=="Senior" for n in gold):
            add_issue("GOLD_NEEDS_SENIOR", "WARN", hdr, "Gold has no Senior")

        # Red/Green size & composition + cap
        if len(rg) < 3 or len(rg) > int(constraints.get("rg_cap",4)):
            add_issue("RG_SIZE", "WARN", hdr, f"Red/Green has {len(rg)} (must be 3–4)")
        need_sr = not any(role_of(n, roster_map)=="Senior" for n in rg)
        need_jr = not any(role_of(n, roster_map)=="Junior" for n in rg)
        need_in = not any(role_of(n, roster_map)=="Intern" for n in rg)
        if need_sr or need_jr or need_in:
            miss = ", ".join(x for x,flag in [("Senior",need_sr),("Junior",need_jr),("Intern",need_in)] if flag)
            add_issue("RG_COMPOSITION", "WARN", hdr, f"RG missing: {miss}")

        # Breast eligibility
        if any(roster_map[n]["PGY"]=="PGY-1" for n in breast):
            add_issue("BREAST_PGY1", "WARN", hdr, "PGY-1 on Breast")

        # Vascular rules (MUST include an intern)
        has_sr = any(role_of(n, roster_map)=="Senior" for n in vas)
        has_jrint = any(role_of(n, roster_map) in {"Junior","Intern"} for n in vas)
        has_intern = any(role_of(n, roster_map)=="Intern" for n in vas)
        if not has_intern:
            add_issue("VASC_NEEDS_INTERN", "ERROR", hdr, "Vascular missing Intern (hard rule)")
        if not (2 <= len(vas) <= int(constraints.get("vascular_cap",3)) and has_sr and has_jrint):
            add_issue("VASC_RULE", "WARN", hdr, f"Expect Sr + Jr/Int, size 2–{int(constraints.get('vascular_cap',3))}; got {len(vas)}")

    # Night spacing (WARN)
    for n in names:
        last = None
        for bi, hdr in enumerate(headers):
            if schedule_df.loc[n, hdr] == "Nights":
                if last is not None and (bi - last) < 2:
                    add_issue("NF_SPACING", "WARN", hdr, f"{n} on NF < 1 block after {headers[last]}")
                last = bi

    # Consecutive >2 same rotation (ERROR)
    for n in names:
        run = 1
        prev = None
        for bi, hdr in enumerate(headers):
            cur = norm_label(schedule_df.loc[n, hdr])
            if cur and prev == cur:
                run += 1
            else:
                run = 1
            if run >= 3:
                add_issue("REPEAT_GT2", "ERROR", hdr, f"{n} has {run} consecutive {cur}")
            prev = cur

    issues_df = pd.DataFrame(issues).sort_values(["Severity","Code","Block","Details"], ignore_index=True)
    summary_issue = (issues_df.value_counts(["Severity","Code"]).rename("Count").reset_index()
                     if not issues_df.empty else pd.DataFrame(columns=["Severity","Code","Count"]))

    # Coverage
    cov_rows=[]
    for hdr in headers:
        row={"Block": hdr}
        for lbl in svc_labels: row[lbl]=int((schedule_df[hdr]==lbl).sum())
        cov_rows.append(row)
    coverage_df=pd.DataFrame(cov_rows)

    # Call-load
    call_rows=[]
    for n in names:
        FSu=Sa=0
        for daily in dailies.values():
            vals = daily.loc[n].to_numpy()
            FSu += (vals=="F/Su").sum()
            Sa  += (vals=="Sa").sum()
        call_rows.append({"Resident": n, "PGY": roster_map[n]["PGY"], "F/Su": FSu, "Sa": Sa, "Total": FSu+Sa})
    order={"PGY-5":0,"PGY-4":1,"PGY-3":2,"PGY-2":3,"PGY-1":4}
    call_df=(pd.DataFrame(call_rows)
             .assign(_ord=lambda d:d["PGY"].map(order))
             .sort_values(["_ord","Total","Resident"], ascending=[True,False,True])
             .drop(columns=["_ord"]).reset_index(drop=True))

    # Rotation counts
    rot=[]
    for n in names:
        rrow={"Resident": n, "PGY": roster_map[n]["PGY"]}
        for lbl in svc_labels: rrow[lbl]=int((schedule_df==lbl).loc[n].sum())
        rot.append(rrow)
    rot_counts_df=(pd.DataFrame(rot)
                   .assign(_ord=lambda d:d["PGY"].map(order))
                   .sort_values(["_ord","Resident"])
                   .drop(columns=["_ord"]).reset_index(drop=True))

    # Per-PGY per block
    pgy_rows=[]
    for hdr in headers:
        for pgy in ["PGY-5","PGY-4","PGY-3","PGY-2","PGY-1"]:
            row={"Block":hdr,"PGY":pgy}
            for lbl in svc_labels:
                row[lbl]=sum(1 for n in names if roster_map[n]["PGY"]==pgy and schedule_df.loc[n,hdr]==lbl)
            pgy_rows.append(row)
    pgy_coverage_df=pd.DataFrame(pgy_rows)

    # Special counts
    spec=[]
    for n in names:
        spec.append({
            "Resident": n, "PGY": roster_map[n]["PGY"],
            "Night Float blocks": int((schedule_df=="Nights").loc[n].sum()),
            "Pittsburgh blocks": int((schedule_df=="Pittsburgh").loc[n].sum()),
            "Elective blocks": int((schedule_df=="Elective").loc[n].sum()),
        })
    special_counts_df=(pd.DataFrame(spec)
                       .assign(_ord=lambda d:d["PGY"].map(order))
                       .sort_values(["_ord","Resident"])
                       .drop(columns=["_ord"]).reset_index(drop=True))

    svc_totals={lbl:int((schedule_df==lbl).sum().sum()) for lbl in svc_labels}
    svc_variance_df=(pd.DataFrame([{"Service":k,"Total blocks":v} for k,v in svc_totals.items()])
                     .sort_values("Service").reset_index(drop=True))

    # Reasons & logs
    reasons = st.session_state.get("assign_reasons", [])
    reasons_df = pd.DataFrame(reasons) if reasons else pd.DataFrame(columns=["Resident","Block","Reason"])
    call_overrides = st.session_state.get("forced_call_issues", [])
    nf_overrides   = st.session_state.get("nf_overrides", [])
    out = {
        "issues": issues_df, "summary": summary_issue, "coverage": coverage_df,
        "call": call_df, "rot_counts": rot_counts_df, "pgy_coverage": pgy_coverage_df,
        "special_counts": special_counts_df, "svc_variance": svc_variance_df,
        "reasons": reasons_df
    }
    if call_overrides:
        out["call_overrides"] = pd.DataFrame(call_overrides, columns=["Type","Block","Details"])
    if nf_overrides:
        out["nf_overrides"] = pd.DataFrame(nf_overrides, columns=["Type","Block","Details"])

    out["heat_rotations"] = rot_counts_df
    out["heat_calls"] = call_df
    return out

# --------------------------
# Amion CSV lint
# --------------------------
def lint_amion(dailies: dict):
    rows=[]
    for tab_name,df in dailies.items():
        for resident in df.index:
            for (weekday,dstr),val in df.loc[resident].items():
                if not isinstance(resident,str) or not resident.strip():
                    rows.append(("ERROR","Blank name", tab_name, dstr, resident, val))
                if not isinstance(dstr,str) or not re.match(r"^\d{2}/\d{2}/\d{4}$", dstr):
                    rows.append(("ERROR","Bad date", tab_name, dstr, resident, val))
                if isinstance(val,str) and ("," in val):
                    rows.append(("WARN","Comma in assignment", tab_name, dstr, resident, val))
    return pd.DataFrame(rows, columns=["Severity","Issue","BlockTab","Date","Resident","Assignment"])

# --------------------------
# Export helpers
# --------------------------
def _build_format_maps_xlsxwriter(workbook):
    """Create and cache XlsxWriter formats for each service color."""
    fmts = {}
    for key, hexcolor in PALETTE.items():
        fmts[key] = workbook.add_format({
            "bg_color": hexcolor,
            "font_color": "#FFFFFF" if key in WHITE_TEXT else "#000000",
            "bold": True if key in BOLD_TEXT else False,
        })
    return fmts

def _apply_coloring_xlsxwriter(ws, df: pd.DataFrame, start_row: int, start_col: int, fmts: dict, weekdays_only=False):
    """Write values back with formatting. start_row/col are 0-based in XlsxWriter."""
    nrows, ncols = df.shape
    # Iterate through data frame
    for r in range(nrows):
        for c in range(ncols):
            val = df.iat[r, c]
            disp = "" if (pd.isna(val) or val == "") else str(val)
            fmt = None
            # If daily and weekdays_only, check weekday level
            if weekdays_only and isinstance(df.columns, pd.MultiIndex):
                wd = df.columns[c][0]
                if wd in ("Saturday","Sunday"):
                    ws.write(start_row + r, start_col + c, disp)  # no color
                    continue
            key = norm_label(disp)
            if key in fmts:
                fmt = fmts[key]
            if fmt is not None:
                ws.write(start_row + r, start_col + c, disp, fmt)
            else:
                ws.write(start_row + r, start_col + c, disp)

def _apply_coloring_openpyxl(ws, df: pd.DataFrame, start_row: int, start_col: int, weekdays_only=False):
    """Fallback coloring with openpyxl. start_row/col are 1-based in openpyxl."""
    try:
        from openpyxl.styles import PatternFill, Font
    except Exception:
        return
    nrows, ncols = df.shape
    for r in range(nrows):
        for c in range(ncols):
            val = df.iat[r, c]
            disp = "" if (pd.isna(val) or val == "") else str(val)
            # Weekday filter for daily
            if weekdays_only and isinstance(df.columns, pd.MultiIndex):
                wd = df.columns[c][0]
                if wd in ("Saturday","Sunday"):
                    ws.cell(row=start_row + r, column=start_col + c, value=disp)
                    continue
            key = norm_label(disp)
            cell = ws.cell(row=start_row + r, column=start_col + c, value=disp)
            if key in PALETTE:
                hexcolor = PALETTE[key].lstrip("#")
                cell.fill = PatternFill(start_color=hexcolor, end_color=hexcolor, fill_type="solid")
                font_color = "FFFFFFFF" if key in WHITE_TEXT else "FF000000"
                cell.font = Font(bold=(key in BOLD_TEXT), color=font_color)

def to_excel(schedule_df, dailies, checks) -> bytes:
    """Export with colors applied to Yearly and Daily sheets."""
    output=io.BytesIO()
    engine="xlsxwriter"
    try:
        import xlsxwriter as _xw  # noqa
    except Exception:
        engine="openpyxl"
    taken=set()
    with pd.ExcelWriter(output, engine=engine) as xw:
        # === Yearly (Schedule) ===
        schedule_name=safe_sheet_name("Schedule", taken)
        schedule_df.to_excel(xw, sheet_name=schedule_name)
        ws = xw.sheets[schedule_name]

        # Freeze and column widths
        try:
            if engine == "xlsxwriter":
                ws.freeze_panes(1,1)
                ws.set_column(0, 0, 22)
                ws.set_column(1, schedule_df.shape[1], 14)
            else:
                ws.freeze_panes = "B2"
                # openpyxl width
                ws.column_dimensions["A"].width = 22
        except Exception:
            pass

        # Apply colors for the Yearly sheet
        if engine == "xlsxwriter":
            fmts = _build_format_maps_xlsxwriter(xw.book)
            # Data region starts at row=1, col=1 (0-based) because index & headers written
            _apply_coloring_xlsxwriter(ws, schedule_df, start_row=1, start_col=1, fmts=fmts, weekdays_only=False)
        else:
            # openpyxl uses 1-based; data starts at row=2, col=2
            _apply_coloring_openpyxl(ws, schedule_df, start_row=2, start_col=2, weekdays_only=False)

        # === Daily sheets ===
        for name,df in dailies.items():
            # Drop 'none' rows for export (as before)
            df2=df.loc[[i for i in df.index if isinstance(i,str) and i.strip().lower()!="none"]]
            sname=safe_sheet_name(name, taken)
            df2.to_excel(xw, sheet_name=sname)
            ws2 = xw.sheets[sname]
            try:
                if engine == "xlsxwriter":
                    ws2.freeze_panes(1,1)
                    ws2.set_column(0, 0, 22)
                    ws2.set_column(1, df2.shape[1], 12)
                else:
                    ws2.freeze_panes = "B2"
                    ws2.column_dimensions["A"].width = 22
            except Exception:
                pass

            # Apply weekday-only colors to Daily (Mon–Fri), leave Sa/Su uncolored
            if engine == "xlsxwriter":
                fmts = _build_format_maps_xlsxwriter(xw.book)
                # MultiIndex columns → header depth = df2.columns.nlevels
                header_depth = getattr(df2.columns, "nlevels", 1)
                start_row = header_depth
                start_col = 1  # index written in col 0
                _apply_coloring_xlsxwriter(ws2, df2, start_row=start_row, start_col=start_col, fmts=fmts, weekdays_only=True)
            else:
                header_depth = getattr(df2.columns, "nlevels", 1)
                # openpyxl 1-based: data start row = 1 (header row) + header_depth
                start_row = 1 + header_depth
                start_col = 2
                _apply_coloring_openpyxl(ws2, df2, start_row=start_row, start_col=start_col, weekdays_only=True)

        # === Checks sheet ===
        checks_sheet=safe_sheet_name("Checks", taken)
        row=0
        for key in ["summary","issues","coverage","call","rot_counts","pgy_coverage","special_counts","svc_variance","reasons","call_overrides","nf_overrides"]:
            df=checks.get(key, pd.DataFrame())
            if df is not None and not df.empty:
                df.to_excel(xw, sheet_name=checks_sheet, index=False, startrow=row)
                try:
                    ws = xw.sheets[checks_sheet]
                    nrows = len(df)
                    if engine == "xlsxwriter" and "Severity" in df.columns:
                        col_idx = list(df.columns).index("Severity")
                        ws.conditional_format(row+1, col_idx, row+nrows, col_idx, {
                            "type": "text",
                            "criteria": "containing",
                            "value": "ERROR",
                            "format": xw.book.add_format({"bg_color":"#FFC7CE"})
                        })
                except Exception:
                    pass
                row += len(df) + 3
    return output.getvalue()

def to_amion_csv(dailies) -> bytes:
    rows=[]
    for tab_name,df in dailies.items():
        show=df.loc[[i for i in df.index if isinstance(i,str) and i.strip().lower()!="none"]]
        for resident in show.index:
            for (weekday,dstr),val in show.loc[resident].items():
                rows.append([resident,dstr,val])
    out=io.StringIO()
    pd.DataFrame(rows, columns=["Resident","Date","Assignment"]).to_csv(out, index=False)
    return out.getvalue().encode("utf-8")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Surgery Scheduler – Editable", layout="wide")
st.title("Surgery Scheduler – Editable Prototype")

if not HAS_ARROW:
    st.warning("Running in Lite mode (no pyarrow). Editable tables use CSV text areas; exports still work.")
if not HAS_PULP:
    st.info("Optional optimizer (PuLP) not detected. Install 'pulp' to enable ILP polish.")

with st.sidebar:
    st.markdown("### Academic Year")
    default_start=date(2026,6,29)
    ay_start=st.date_input("Start date (Block 1 begins Monday)", value=default_start)

    st.markdown("---"); st.markdown("### Constraints")
    st.checkbox("Require Senior on Gold", True, key="senior_on_gold")
    st.checkbox("Red/Green must include Sr + Jr + Intern", True, key="rg_require_sji")
    st.checkbox("Allow a 4th resident on Red/Green", True, key="rg_allow_fourth")
    st.checkbox("Prioritize fill: Gold → Red/Green", True, key="priority_gold_rg")
    st.checkbox("Breast only PGY-2+", True, key="breast_pgy2plus")
    st.checkbox("Vascular = Senior + Junior/Intern", True, key="vascular_sr_jr")

    st.number_input("Gold team cap", min_value=1, max_value=10, value=4, key="gold_cap")
    st.number_input("Red/Green team cap", min_value=3, max_value=10, value=4, key="rg_cap")
    st.number_input("Vascular team cap", min_value=2, max_value=5, value=3, key="vascular_cap")

    st.markdown("**Service minima (per block)**")
    gold_min = st.number_input("Gold min", 0, 6, 3, key="gold_min")
    rg_min   = st.number_input("Red/Green min", 0, 6, 3, key="rg_min")
    icu_min  = st.number_input("ICU min", 0, 2, 1, key="icu_min")

    st.markdown("**Night Float**")
    st.checkbox("Enable Night Float (Sr+Jr+Intern every block)", True, key="nightfloat")
    st.caption("NF rules: Sr+Jr+Intern each block; ≥1-block gap preferred; avoid PGY-5 last block.")

    st.markdown("**Fairness/Soft penalties**")
    st.slider("Penalty: NF gap violations", 0.0, 10.0, 3.0, 0.5, key="penalty_nf_gap")
    st.slider("Penalty: PGY-5 on Nights (last block)", 0.0, 10.0, 5.0, 0.5, key="penalty_pgy5_last")
    st.slider("Penalty: Gold lacks a Senior", 0.0, 10.0, 3.0, 0.5, key="penalty_gold_sen")
    st.slider("Penalty: consecutive same rotation", 0.0, 10.0, 2.0, 0.5, key="penalty_consecutive_repeat")

    st.markdown("**Elig. calendars JSON (per service)**")
    st.text_area("Example: {\"Vascular\": {\"PGY-1\": [0,1], \"names\": {\"Jane Doe\": [3]}}}",
                 value="", key="eligibility_calendars_json")

    st.number_input("Max times a resident can be the 3rd on Vascular", 0, 5, 1, key="vascular_third_cap")

    st.markdown("**Weekend Call – Exclude these rotations**")
    st.multiselect(
        "Excluded from weekend call (may override as last resort)",
        ["Nights","Elective","Pittsburgh","Vacation","Breast","Vascular","Chief","ICU","Floor","Gold","Red/Green"],
        default=["Nights","Elective","Pittsburgh","Vacation"], key="exclude_from_call"
    )

    st.markdown("---")
    # Roster management controls
    st.number_input("Target roster size", min_value=1, max_value=120, value=18, step=1, key="roster_size")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Add blank row"):
            row = pd.DataFrame([{"Resident":"", "PGY":"PGY-1", "Prelim":"N", "Include?":"Y",
                                 "Max Nights":"","Avoid Services":"","PTO Windows":"","Notes":"","Lock":""}])
            st.session_state.roster_table = pd.concat([st.session_state.get("roster_table", pd.DataFrame(columns=["Resident","PGY","Prelim","Include?","Max Nights","Avoid Services","PTO Windows","Notes","Lock"])), row], ignore_index=True)
    with c2:
        if st.button("🧹 Remove empty/None rows"):
            if "roster_table" in st.session_state:
                rt = st.session_state.roster_table.copy()
                rt["Resident"] = rt["Resident"].astype(str)
                bad = rt["Resident"].str.strip().str.lower().isin(["", "none", "na", "n/a"])
                st.session_state.roster_table = rt[~bad].reset_index(drop=True)
    with c3:
        if st.button("↔ Resize to N"):
            if "roster_table" in st.session_state:
                st.session_state.roster_table = ensure_roster_size(st.session_state.roster_table, st.session_state.roster_size)
            else:
                st.session_state.roster_table = ensure_roster_size(pd.DataFrame(columns=["Resident","PGY","Prelim","Include?","Max Nights","Avoid Services","PTO Windows","Notes","Lock"]), st.session_state.roster_size)

    st.checkbox("Auto-assign weekend call after edits", True, key="auto_call")

    st.markdown("---")
    st.checkbox("Use optimizer (polish with ILP if available)", False, key="use_optimizer")
    st.number_input("Optimizer time limit (sec)", 1, 30, 8, key="opt_time_limit")
    st.number_input("Random seed (tie-breaks)", 0, 10_000, 0, key="random_seed")
    scenario_name = st.text_input("Scenario name (save snapshot)", value="", key="scenario_name")

# Editable roster (seed)
st.subheader("Roster (editable)")
_default_roster=pd.DataFrame([
    ["Avi Robinson","PGY-5","N","Y","","","","",""],
    ["Kathleen Koesarie","PGY-5","N","Y","","","","",""],
    ["Arruj Hassan","PGY-5","N","Y","","","","",""],
    ["Shirin Siddiqi","PGY-5","N","Y","","","","",""],
    ["Kayla Orr","PGY-4","N","Y","","","","",""],
    ["Makayla Gologram","PGY-4","N","Y","","","","",""],
    ["Adrianne Pellegrini","PGY-4","N","Y","","","","",""],
    ["Zane Hamden","PGY-3","N","Y","","","","",""],
    ["Lauren Delong","PGY-3","N","Y","","","","",""],
    ["Brittany Steffens","PGY-3","N","Y","","","","",""],
    ["Zoe Wecht","PGY-2","N","Y","","","","",""],
    ["Jessica Marks","PGY-2","N","Y","","","","",""],
    ["Jacob Allenabaugh","PGY-2","N","Y","","","","",""],
    ["Intern 1","PGY-1","N","Y","","","","",""],
    ["Intern 2","PGY-1","N","Y","","","","",""],
    ["Intern 3","PGY-1","N","Y","","","","",""],
    ["Intern 4 (Prelim)","PGY-1","Y","Y","","","","",""],
    ["Intern 5 (Prelim)","PGY-1","Y","Y","","","","",""],
], columns=["Resident","PGY","Prelim","Include?","Max Nights","Avoid Services","PTO Windows","Notes","Lock"])

if "roster_table" not in st.session_state:
    st.session_state.roster_table = _default_roster.copy()

roster = show_table(st.session_state.roster_table, "roster_editor", editable=True, hide_index=True, index_name_hint="Resident")
st.session_state.roster_table = roster.copy()

# Side: quick fairness indicator (call Gini)
if "dailies" in st.session_state:
    call_totals = []
    for n in st.session_state.schedule_df.index:
        FSu=Sa=0
        for d in st.session_state.dailies.values():
            vals=d.loc[n].to_numpy(); FSu += (vals=="F/Su").sum(); Sa += (vals=="Sa").sum()
        call_totals.append(FSu+Sa)
    st.sidebar.markdown(f"**Call balance (Gini):** {gini(call_totals):.3f}")

st.markdown("---")
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button("Generate schedule from roster"):
        try:
            roster_raw = st.session_state.roster_table.copy()
            roster_raw["Resident"] = roster_raw["Resident"].astype(str)
            bad_names = roster_raw["Resident"].str.strip().str.lower().isin(["", "none", "na", "n/a"])
            roster_raw = roster_raw[~bad_names].reset_index(drop=True)
            roster_raw["PGY"] = roster_raw["PGY"].astype(str)
            roster_raw["PGY"] = roster_raw["PGY"].str.replace(r"\(.*?\)", "", regex=True)
            roster_raw["PGY"] = roster_raw["PGY"].str.upper().str.replace(r"PGY\s*([1-5])", r"PGY-\1", regex=True)
            roster_raw["PGY"] = roster_raw["PGY"].str.replace(r"^\s*([1-5])\s*$", r"PGY-\1", regex=True)
            roster_norm = normalize_roster_input(roster_raw)
            validate_roster(roster_norm)
        except Exception as e:
            st.error(f"Roster problem: {e}"); st.stop()

        # Constraints pack
        try:
            elig_json = st.session_state.eligibility_calendars_json.strip()
            eligibility_calendars = json.loads(elig_json) if elig_json else {}
        except Exception as e:
            st.warning(f"Eligibility calendars JSON invalid: {e}")
            eligibility_calendars = {}

        constraints = {
            "senior_on_gold": st.session_state.senior_on_gold,
            "rg_require_sji": st.session_state.rg_require_sji,
            "rg_allow_fourth": st.session_state.rg_allow_fourth,
            "priority_gold_rg": st.session_state.priority_gold_rg,
            "breast_pgy2plus": st.session_state.breast_pgy2plus,
            "vascular_sr_jr": st.session_state.vascular_sr_jr,
            "nightfloat": st.session_state.nightfloat,

            "gold_cap": st.session_state.gold_cap,
            "rg_cap": st.session_state.rg_cap,
            "vascular_cap": st.session_state.vascular_cap,

            "service_minima": {"Gold":st.session_state.gold_min, "Red/Green":st.session_state.rg_min, "ICU":st.session_state.icu_min},

            "exclude_from_call": st.session_state.exclude_from_call,
            "pg4_elective": True,
            "pg3_pittsburgh": True,

            "penalty_nf_gap": st.session_state.penalty_nf_gap,
            "penalty_pgy5_last": st.session_state.penalty_pgy5_last,
            "penalty_gold_senior": st.session_state.penalty_gold_sen,
            "penalty_consecutive_repeat": st.session_state.penalty_consecutive_repeat,

            "vascular_third_cap_per_resident": st.session_state.vascular_third_cap,

            "eligibility_calendars": eligibility_calendars,

            "random_seed": st.session_state.random_seed,
        }
        st.session_state.roster_df = roster_norm.copy()
        base_fixed = st.session_state.get("schedule_df", None)

        yearly = auto_generate_yearly(roster_norm.copy(), ay_start, constraints, base_fixed=base_fixed)
        if st.session_state.use_optimizer:
            yearly = polish_with_optimizer(yearly, roster_norm.copy(), ay_start, constraints, time_limit_s=st.session_state.opt_time_limit)

        dailies = build_dailies_from_yearly(yearly, ay_start)
        if st.session_state.auto_call:
            auto_assign_weekend_call(dailies, yearly, ay_start, constraints, roster_norm)

        st.session_state.schedule_df = yearly
        st.session_state.dailies = dailies
        st.session_state.constraints = constraints
        st.session_state.schedule_df_effective = None
        try:
            st.toast("Generated. See tabs below.")
        except Exception:
            st.success("Generated. See tabs below.")
with colB:
    edit_mode = st.toggle("Edit mode", value=True, help="ON = tables editable; OFF = preview (weekday colors only)")
with colC:
    if st.button("💾 Save scenario"):
        sc = st.session_state.get("scenarios", {})
        sc = {} if sc is None else dict(sc)
        name = st.session_state.scenario_name.strip() or f"Scenario {len(sc)+1}"
        sc[name] = {
            "schedule": st.session_state.get("schedule_df", pd.DataFrame()).copy(),
            "dailies": st.session_state.get("dailies", {}).copy(),
            "constraints": st.session_state.get("constraints", {}).copy(),
            "roster": st.session_state.get("roster_df", pd.DataFrame()).copy(),
        }
        st.session_state.scenarios = sc
        st.success(f"Saved: {name}")

# Utility: get a valid Yearly for checks/export
def get_effective_yearly_for_checks():
    sched = st.session_state.get("schedule_df", None)
    dailies = st.session_state.get("dailies", None)
    if sched is not None and not sched.empty:
        return sched
    if dailies:
        any_df = next(iter(dailies.values()))
        idx_order = list(any_df.index)
        return effective_yearly_from_dailies(dailies, ay_start, idx_order)
    return None

if "dailies" in st.session_state and "schedule_df" in st.session_state:
    schedule_df = st.session_state.schedule_df
    dailies     = st.session_state.dailies
    constraints = st.session_state.constraints
    roster_df_ss= st.session_state.get("roster_df", roster)

    tabs = st.tabs(["Yearly (editable)","Daily Blocks (editable)","Checks","Export"])

    # Yearly
    with tabs[0]:
        st.markdown("#### Yearly Schedule")
        target = st.selectbox("Jump to resident", ["(all)"] + list(schedule_df.index), index=0, key="jump_resident")
        view_df = schedule_df
        if target != "(all)":
            view_df = schedule_df.loc[[target]]
        if edit_mode:
            yearly_edit = view_df.reset_index().rename(columns={"index":"Resident"})
            edited = show_table(yearly_edit, "yearly_editor", editable=True, hide_index=True, index_name_hint="Resident")
            if st.button("Apply Yearly edits → rebuild Daily / Checks / Export"):
                new_yearly = edited.set_index("Resident") if "Resident" in edited.columns else edited.copy()
                if target != "(all)":
                    full = schedule_df.copy()
                    full.loc[new_yearly.index, :] = new_yearly.values
                    new_yearly = full
                st.session_state.schedule_df = new_yearly
                new_dailies = build_dailies_from_yearly(new_yearly, ay_start)
                if st.session_state.auto_call:
                    auto_assign_weekend_call(new_dailies, new_yearly, ay_start, constraints, roster_df_ss)
                st.session_state.dailies = new_dailies
                st.session_state.schedule_df_effective = None
                st.success("Applied Yearly edits.")
        else:
            vis = view_df.reset_index().rename(columns={"index":"Resident"})
            cols = vis.columns[1:]
            st.dataframe(vis.style.applymap(style_each_cell, subset=cols),
                         use_container_width=True, hide_index=True)

    # Daily
    with tabs[1]:
        st.markdown("#### Daily block tabs (Weekends uncolored; Sa / F/Su text only)")
        new_dailies={}
        for name,df in dailies.items():
            st.markdown(f"**{name}**")
            if edit_mode:
                edited = show_table(df, f"edit_{name}", editable=True, daily=True, index_name_hint="Resident")
                new_dailies[name]=edited.copy()
            else:
                weekday_cols=[c for c in df.columns if c[0] in ["Monday","Tuesday","Wednesday","Thursday","Friday"]]
                st.dataframe(df.style.applymap(style_each_cell, subset=weekday_cols),
                             use_container_width=True)
        if edit_mode and st.button("Apply Daily edits → recompute Checks & Export"):
            st.session_state.dailies = new_dailies
            eff_yearly = effective_yearly_from_dailies(new_dailies, ay_start, list(schedule_df.index))
            st.session_state.schedule_df_effective = eff_yearly
            st.success("Applied Daily edits and derived effective Yearly.")

    # Checks
    with tabs[2]:
        st.markdown("#### Checks & Balance Tables")
        base_yearly = st.session_state.get("schedule_df_effective", None) or get_effective_yearly_for_checks()
        if base_yearly is None or base_yearly.empty:
            st.info("No schedule available yet. Generate a schedule first.")
        else:
            checks = compute_checks(base_yearly, st.session_state.dailies, roster_df_ss, constraints, ay_start)
            c1,c2,c3 = st.columns([1.1,1.25,1.25])
            with c1:
                st.markdown("**Summary (Severity × Code)**")
                st.dataframe(checks["summary"], use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Per-block coverage**")
                st.dataframe(checks["coverage"], use_container_width=True, hide_index=True)
            with c3:
                st.markdown("**Call-load by resident**")
                st.dataframe(checks["call"], use_container_width=True, hide_index=True)

            st.markdown("**Issues (detailed)**"); st.dataframe(checks["issues"], use_container_width=True, hide_index=True)
            st.markdown("**Per-resident rotation counts (PGY-sorted)**"); st.dataframe(checks["rot_counts"], use_container_width=True, hide_index=True)
            st.markdown("**Per-PGY coverage per block**"); st.dataframe(checks["pgy_coverage"], use_container_width=True, hide_index=True)
            st.markdown("**Special blocks per resident**"); st.dataframe(checks["special_counts"], use_container_width=True, hide_index=True)
            st.markdown("**Service totals across the year**"); st.dataframe(checks["svc_variance"], use_container_width=True, hide_index=True)
            if "reasons" in checks and not checks["reasons"].empty:
                st.markdown("**Why these assignments?**"); st.dataframe(checks["reasons"], use_container_width=True, hide_index=True)
            if "call_overrides" in checks:
                st.markdown("**Weekend call overrides (forced for coverage)**")
                st.dataframe(checks["call_overrides"], use_container_width=True, hide_index=True)
            if "nf_overrides" in checks:
                st.markdown("**Night Float overrides & warnings**")
                st.dataframe(checks["nf_overrides"], use_container_width=True, hide_index=True)

    # Export
    with tabs[3]:
        base_yearly = st.session_state.get("schedule_df_effective", None) or get_effective_yearly_for_checks()
        if base_yearly is None or base_yearly.empty:
            st.info("No schedule available to export yet.")
        else:
            lint = lint_amion(st.session_state.dailies)
            if not lint.empty:
                st.markdown("**Amion CSV Lint**"); st.dataframe(lint, use_container_width=True, hide_index=True)

            checks_now = compute_checks(base_yearly, st.session_state.dailies, roster_df_ss, constraints, ay_start)
            xlsx_bytes = to_excel(base_yearly, st.session_state.dailies, checks_now)
            st.download_button("⬇️ Download Excel (.xlsx)", data=xlsx_bytes,
                               file_name="ResidentSchedule_26-27_Editable.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            csv_bytes = to_amion_csv(st.session_state.dailies)
            st.download_button("⬇️ Download Amion-style CSV (.csv)", data=csv_bytes,
                               file_name="ResidentSchedule_26-27_Amion.csv", mime="text/csv")
else:
    st.info("Edit the roster and click **Generate schedule from roster** to begin.")
