"""IA-MixMatch: Imbalance-aware MixMatch for multi-label chest X-ray classification."""
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
HEAD_T, TAIL_T = 0.05, 0.02
