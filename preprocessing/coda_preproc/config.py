"""Constants shared by every preprocessing stage.

Values are copied from preprocessing.ipynb (Cells 2, 28, 29 and 32 in the
notebook's own numbering). Nothing here was retuned. Changing any value
changes the processed data used to train the archived models.
"""
from pathlib import Path

RANDOM_SEED = 42
STRING_THRESHOLD = 700          # STRING v12.0 combined_score
N_PANEL_GENES = 72
TASK = "y_T1_significant"       # F2+ vs F0/F1
CAL_MODE = "V1"                 # one whole T1 training cohort calibrates each fold

# Label resolution codes (notebook Cell 7).
RES_EXACT = "exact"
RES_G_F0F1 = "grouped_F0_F1"
RES_G_F3F4 = "grouped_F3_F4"
RES_G_F0F2 = "grouped_F0_F2"
RES_HEALTHY = "healthy"
RES_UNKNOWN = "unknown"

# Final tiers after the data-driven promotions of notebook Cell 13.
T1_COHORTS = ["GSE130970", "GSE135251", "GSE162694", "GSE193066",
              "GSE213621", "GSE240729", "GSE31803", "GSE84044"]
T2_COHORTS = ["GSE207310", "GSE276114", "GSE48452", "GSE89632"]
PLATFORM = {
    "GSE130970": "rnaseq", "GSE135251": "rnaseq", "GSE162694": "rnaseq",
    "GSE193066": "rnaseq", "GSE207310": "rnaseq", "GSE213621": "rnaseq",
    "GSE240729": "rnaseq", "GSE276114": "rnaseq",
    "GSE31803": "microarray", "GSE48452": "microarray",
    "GSE84044": "microarray", "GSE89632": "microarray",
}

# Panel pathways (notebook Cell 29). Order matters: it fixes the order of
# the six pathway-activity features.
PATHWAYS = {
    "ECM_remodeling": ["COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1", "COL6A1",
                       "FN1", "LUM", "DCN", "VCAN", "BGN", "SPARC", "MMP2", "MMP14",
                       "TIMP1", "TIMP2", "LOX", "LOXL2", "POSTN", "THBS2"],
    "TGFB_signaling": ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4",
                       "SMAD7", "SERPINE1", "CCN2", "THBS1", "LTBP1"],
    "HSC_activation": ["ACTA2", "TAGLN", "PDGFRB", "PDGFB", "DES", "VIM", "COL1A1",
                       "TIMP1", "MYH11", "CALD1", "NOTCH3", "RGS5"],
    "Inflammation": ["CD68", "CCL2", "CCL5", "CXCL9", "CXCL10", "IL6", "TNF", "IL1B",
                     "CD163", "MARCO", "ITGAX", "TLR4", "NFKB1", "SPP1", "CSF1R"],
    "Apoptosis_stress": ["CASP3", "CASP8", "BAX", "BCL2", "TP53", "FAS", "XBP1", "DDIT3",
                         "ATF4", "HSPA5", "GADD45A", "CDKN1A"],
    "Hepatocyte_function": ["CYP2E1", "CYP1A2", "CYP2C9", "CPS1", "ASS1", "ARG1",
                            "SERPINA1", "TTR", "TF", "HNF4A", "ALDOB", "GLUL", "APOH",
                            "AHSG", "ASGR1", "NR1H4"],
}

# A-priori gene sets for the directional deviation target (notebook Cell 28).
ECM_UP = ["COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL5A1", "COL5A2",
          "COL6A1", "COL6A2", "COL6A3", "COL15A1", "COL18A1", "FN1", "LUM", "DCN",
          "VCAN", "BGN", "FBN1", "SPARC", "THBS1", "THBS2", "TIMP1", "TIMP2", "MMP2",
          "MMP14", "LOX", "LOXL1", "LOXL2", "ELN", "POSTN", "TNC", "SERPINE1", "ACTA2",
          "TAGLN", "PDGFRB", "TGFB1", "CCN2", "SPP1", "VIM", "ITGB1", "FBLN1", "FBLN2",
          "NID1", "LAMB1", "LAMC1", "EMILIN1", "MFAP4", "OLFML3", "CTHRC1", "SULF1"]
HEP_DOWN = [
    "CYP2E1", "CYP3A4", "CYP1A2", "CYP2C9", "CYP2C8", "CYP2C19", "CYP2A6", "CYP2B6",
    "CYP2D6", "CYP8B1", "CYP4A11", "CYP7A1", "CYP27A1",
    "UGT1A1", "UGT2B7", "SULT2A1", "GSTA1", "NAT2", "CES1",
    "CPS1", "OTC", "ASS1", "ASL", "ARG1", "TAT", "HAL", "SDS", "CTH", "AGXT", "HPD", "FAH",
    "GLS2", "BHMT", "GNMT", "MAT1A", "TDO2",
    "PCK1", "ALDOB", "FBP1", "G6PC1", "SLC2A2", "ADH1B", "ADH1C", "ADH4", "ACSM2B",
    "SERPINA1", "SERPINC1", "SERPIND1", "TTR", "TF", "FGG", "AHSG", "APOH", "APOC3",
    "APOA2", "APOC1", "HPX", "ITIH4", "AMBP", "F2", "F7", "F9", "PROC", "AGT", "C8A", "C9",
    "ABCB11", "SLCO1B1", "SLC10A1", "SLC22A1", "NR1H4", "NR1I2", "NR1I3",
    "HNF4A", "HNF1A", "ONECUT1", "FOXA2", "GLUL",
]
MIN_ECM, MIN_HEP = 25, 12

FEATURES_A = ["expression_z", "deviation_rank_z"] + [f"pw_{p}" for p in PATHWAYS]
FEATURES_B = ["pathway_membership_count", "degree_centrality",
              "betweenness_centrality", "closeness_centrality",
              "clustering_coefficient", "gtex_expression_level",
              "gtex_expression_std", "expression_variability",
              "cross_platform_consistency"]
FEATURES_C = ["disease_specificity", "pathway_coherence", "cross_etiology_consistency",
              "fibrosis_progression", "cohens_d", "t_statistic", "log_fold_change",
              "per_gene_auc", "rank_score_composite"]

# Columns written to every fold's y.csv in the archived package.
# The notebook's later "v2" export adds exact_stage and eligible_T* columns;
# set ARCHIVE_COMPATIBLE=False in export.write_package to get those as well.
META_COLS_ARCHIVE = ["cohort", "stage", "resolution", "tier", "etiology", "platform",
                     "y_T1_significant", "y_T2_ordinal", "y_T3_early_adv"]

STRING_LINKS_URL = ("https://stringdb-downloads.org/download/protein.links.v12.0/"
                    "9606.protein.links.v12.0.txt.gz")
STRING_INFO_URL = ("https://stringdb-downloads.org/download/protein.info.v12.0/"
                   "9606.protein.info.v12.0.txt.gz")
GTEX_TPM_URL = ("https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
                "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz")
GTEX_ATTR_URL = ("https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/"
                 "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt")


def paths(root):
    root = Path(root)
    p = {k: root / k for k in ("raw", "interim", "processed", "reports", "manifests")}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p
