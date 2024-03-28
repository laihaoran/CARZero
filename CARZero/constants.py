from pathlib import Path

# MIMIC constants
MIMIC_DATA_DIR = Path("./Dataset/MIMIC/")

MIMIC_MASTER_CSV_XH =  "mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv"

MIMIC_VALID_NUM = 5000
MIMIC_VIEW_COL = "Frontal/Lateral"
MIMIC_PATH_COL = "Path"
MIMIC_SPLIT_COL = "Split"
MIMIC_REPORT_COL = "Report Impression"
MIMIC_LLM_REPORT_COL = "LLM Report Impression"
MIMIC_XH_REPORT_COL = "xinhuo"
MIMIC_LLM_REPORT_V1_COL = "LLM Report v1 Impression"
MIMIC_DataFlag_COL = "Data Flag"
MIMIC_RAMINDEX_COL = "Index"
MIMIC_Original_VIEW_COL = "OriginalView"
