from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hospital_data.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";").copy()

    df["DateAdmission"] = pd.to_datetime(df["DateAdmission"], format="%d/%m/%Y", errors="coerce")
    df["DateSortie"] = pd.to_datetime(df["DateSortie"], format="%d/%m/%Y", errors="coerce")

    df["DureeCalculee"] = (df["DateSortie"] - df["DateAdmission"]).dt.days
    df["Ecart"] = df["DureeSejour"] - df["DureeCalculee"]
    df["CoutParJour"] = df["Cout"] / df["DureeSejour"].replace(0, np.nan)
    df["MoisAdmission"] = df["DateAdmission"].dt.month_name()
    df["SejourLong"] = (df["DureeSejour"] > df["DureeSejour"].mean()).astype(int)

    df["TrancheAge"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 50, 65, 80, 100],
        labels=["0-18", "19-35", "36-50", "51-65", "66-80", "81+"],
        include_lowest=True,
    )

    iso_data = df[["Age", "DureeSejour", "Cout"]].copy()
    iso_data = iso_data.fillna(iso_data.median(numeric_only=True))

    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Anomalie"] = iso.fit_predict(iso_data)
    df["AnomalieLabel"] = df["Anomalie"].map({1: "Normal", -1: "Anomalie"})

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    return {
        "patients": len(df),
        "age_moyen": round(df["Age"].mean(), 2),
        "duree_moyenne": round(df["DureeSejour"].mean(), 2),
        "cout_moyen": round(df["Cout"].mean(), 2),
        "departement_dominant": df["Departement"].mode().iloc[0] if not df.empty else "N/A",
        "anomalies": int((df["Anomalie"] == -1).sum())
    }