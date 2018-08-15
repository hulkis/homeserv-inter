"""Console script for homeserv_inter."""
import fire
import pandas as pd

from homeserv_inter.constants import DATA_DIR, RAW_DATA_DIR, NUMERIC_COLS, STR_COLS
from homeserv_inter.datahandler import generate_cleaned_sets
from homeserv_inter.model import LgbHomeService
from wax_toolbox import Timer


def convert_csv_to_parquet(engine="pyarrow"):
    """Convert csv files into train.parquet.gzip & test.parquet.gzip.
    Args:
        engine (str): engine for parquet format among ['pyarrow', 'fastparquet']

    Assumption:

        ..code :: bash
            ❯ tree data
            data
            ├── raw
            │   ├── contract_history.csv
            │   ├── data_description.xlsx
            │   ├── equipment.csv
            │   ├── intervention_history.csv
            │   ├── intervention-life-cycle
            │   │   ├── PROCESS_BI_CONTRATS_EN.ppt
            │   │   └── PROCESS_BI_CONTRATS_FR.ppt
            │   ├── intervention_test.csv
            │   ├── intervention_train.csv
            │   ├── nature-code
            │   │   ├── nature_code_eau_chaude.csv
            │   │   ├── nature_code_energie.csv
            │   │   ├── nature_code_fonction.csv
            │   │   ├── nature_code_installation.csv
            │   │   └── nature_code_specification.csv
            │   ├── organisation.csv
            │   └── sample_submission.csv
            └── read_and_join.ipynb

            3 directories, 16 files
"""

    def to_datetimes(dt_cols, df):
        for col in dt_cols:
            df[col] = pd.to_datetime(df[col])
        return df

    with Timer("Reading organisation.csv", report_func=print):
        orga_df = pd.read_csv(
            RAW_DATA_DIR / "organisation.csv", sep="|", encoding="Latin-1"
        )

    with Timer("Reading equipment.csv", report_func=print):
        equipement_df = pd.read_csv(
            RAW_DATA_DIR / "equipment.csv", sep="|", encoding="Latin-1"
        )
        dt_cols = ["INSTALL_DATE", "RACHAT_DATE"]
        equipement_df = to_datetimes(dt_cols, equipement_df)

    with Timer("Reading contract_history.csv", report_func=print):
        contrat_history_df = pd.read_csv(
            RAW_DATA_DIR / "contract_history.csv", sep="|", encoding="Latin-1"
        )
        dt_cols = ["CRE_DATE", "UPD_DATE", "DATE_RESILIATION", "DATE_DEBUT", "DATE_FIN"]
        contrat_history_df = to_datetimes(dt_cols, contrat_history_df)

    with Timer("Reading intervention_test.csv", report_func=print):
        intervention_test_df = pd.read_csv(
            RAW_DATA_DIR / "intervention_test.csv", sep="|", encoding="Latin-1"
        )
        dt_cols = ["SCHEDULED_START_DATE", "SCHEDULED_END_DATE", "CRE_DATE_GZL"]
        intervention_test_df = to_datetimes(dt_cols, intervention_test_df)

    with Timer("Reading intervention_train.csv", report_func=print):
        intervention_train_df = pd.read_csv(
            RAW_DATA_DIR / "intervention_train.csv", sep="|", encoding="Latin-1"
        )
        dt_cols = ["SCHEDULED_START_DATE", "SCHEDULED_END_DATE", "CRE_DATE_GZL"]
        intervention_train_df = to_datetimes(dt_cols, intervention_train_df)

    with Timer("Reading intervention_test.csv", report_func=print):
        intervention_history_df = pd.read_csv(
            RAW_DATA_DIR / "intervention_history.csv", sep="|", encoding="Latin-1"
        )
        dt_cols = [
            "DATE_SAISIE_RETOUR",
            "SCHEDULED_START_DATE",
            "SCHEDULED_END_DATE",
            "ACTUAL_START_DATE",
            "ACTUAL_END_DATE",
            "CRE_DATE_GZL",
        ]
        intervention_history_df = to_datetimes(dt_cols, intervention_history_df)

    with Timer("Reading nature-code csv", report_func=print):
        nature_code_eau_chaude = pd.read_csv(
            RAW_DATA_DIR / "nature-code/nature_code_eau_chaude.csv",
            sep="|",
            encoding="Latin-1",
        )
        nature_code_energie = pd.read_csv(
            RAW_DATA_DIR / "nature-code/nature_code_energie.csv",
            sep="|",
            encoding="Latin-1",
        )
        nature_code_fonction = pd.read_csv(
            RAW_DATA_DIR / "nature-code/nature_code_fonction.csv",
            sep="|",
            encoding="Latin-1",
        )
        nature_code_installation = pd.read_csv(
            RAW_DATA_DIR / "nature-code/nature_code_installation.csv",
            sep="|",
            encoding="Latin-1",
        )
        nature_code_specification = pd.read_csv(
            RAW_DATA_DIR / "nature-code/nature_code_specification.csv",
            sep="|",
            encoding="Latin-1",
        )

    def prepare_data(data_):
        data = data_.merge(equipement_df, how="left", on="INSTANCE_ID").merge(
            orga_df,
            how="left",
            left_on="ORGANISATION_ID",
            right_on="L2_ORGANISATION_ID",
        )
        contrat_history_s = (
            data[["INCIDENT_NUMBER", "INSTANCE_ID", "CRE_DATE_GZL"]]
            .merge(contrat_history_df)
            .query("CRE_DATE_GZL>=UPD_DATE")
        )
        contrat_history_s = contrat_history_s.sort_values(
            ["INCIDENT_NUMBER", "UPD_DATE"], ascending=[True, False]
        ).drop_duplicates(keep="first", subset=["INCIDENT_NUMBER"])
        data = data.merge(contrat_history_s, how="left").merge(
            nature_code_eau_chaude, how="left"
        )
        data = data.merge(nature_code_energie, how="left").merge(
            nature_code_fonction, how="left"
        )
        data = data.merge(nature_code_installation, how="left").merge(
            nature_code_specification, how="left"
        )
        return data

    with Timer("Merging all train set", report_func=print):
        train_data = prepare_data(intervention_train_df)
        for col in STR_COLS:
            train_data[col] = train_data[col].astype(str)
        for col in NUMERIC_COLS + ["target"]:
            train_data[col] = pd.to_numeric(train_data[col])
        train_data.to_parquet(
            DATA_DIR / "train.parquet.gzip", compression="gzip", engine=engine
        )

    with Timer("Merging all test set", report_func=print):
        test_data = prepare_data(intervention_test_df)
        for col in STR_COLS:
            test_data[col] = test_data[col].astype(str)
        for col in NUMERIC_COLS:
            test_data[col] = pd.to_numeric(test_data[col])
        test_data.to_parquet(
            DATA_DIR / "test.parquet.gzip", compression="gzip", engine=engine
        )


def main():
    return fire.Fire(
        {
            "convert-raw": convert_csv_to_parquet,
            "convert-cleaned": generate_cleaned_sets,
            "model": LgbHomeService,
        }
    )
