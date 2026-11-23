"""Map current HGNC symbols back to the names used in the archived matrices.

The archived matrices (5 Aug 2026) were built when MyGene.info still returned
the five old symbols below. HGNC has since renamed these genes. A re-run today
gets the new names for cohorts whose identifiers are mapped through
MyGene.info (GSE130970, GSE135251, GSE162694, GSE207310, GSE240729), while
the cohorts whose platform files carry symbols directly keep the old names
(GSE193066, GSE213621, GSE276114, GSE31803, GSE48452, GSE84044, GSE89632).
Without this map the shared gene set loses all five genes (10,730 instead of
10,735). With it, a re-run from GEO reproduces the archived gene set and order.
"""
import pandas as pd

# current symbol -> symbol in the archived matrices
NEW_TO_ARCHIVE = {
    "LYLDT1": "C12orf54",
    "KLHL27": "IPP",
    "FYRD1": "TBRG1",
    "FASTKD4": "TBRG4",
    "FAABTP": "TMEM19",
}


def to_archive_symbols(df: pd.DataFrame, cohort: str = "") -> pd.DataFrame:
    """Rename current symbols in a genes x samples matrix to the archived ones.

    Raises if a cohort holds both the old and the new name of one gene, since
    renaming would then create a duplicate row.
    """
    present = [g for g in NEW_TO_ARCHIVE if g in df.index]
    clash = [g for g in present if NEW_TO_ARCHIVE[g] in df.index]
    if clash:
        raise ValueError(f"{cohort}: both old and new symbol present for {clash}")
    return df.rename(index=NEW_TO_ARCHIVE) if present else df


def apply_to_all(expr: dict) -> dict:
    """Apply `to_archive_symbols` to every cohort matrix; return a new dict."""
    return {g: to_archive_symbols(m, g) for g, m in expr.items()}
