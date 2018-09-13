import pandas as pd

from homeserv_inter.constants import DATA_DIR, NUMERIC_COLS, RAW_DATA_DIR, STR_COLS
from wax_toolbox import Timer


def to_datetimes(dt_cols, df):
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col])
    return df


class HomeServiceRaw:
    def __init__(self, use_full_history=False, engine="pyarrow"):
        self.use_full_history = use_full_history
        self.engine = engine

    def _merge_them_all(self, data, df_orga, df_equipement,
                        df_contract_history, *df_nature_codes):
        data = data.merge(df_equipement, how="left", on="INSTANCE_ID")
        data = data.merge(
            df_orga,
            how="left",
            left_on="ORGANISATION_ID",
            right_on="L2_ORGANISATION_ID",
        )

        # Dropping some values:
        contrat_history_s = (data[[
            "INCIDENT_NUMBER", "INSTANCE_ID", "CRE_DATE_GZL"
        ]].merge(df_contract_history).query("CRE_DATE_GZL>=UPD_DATE"))
        contrat_history_s = contrat_history_s.sort_values(
            ["INCIDENT_NUMBER", "UPD_DATE"],
            ascending=[True, False]).drop_duplicates(
                keep="first", subset=["INCIDENT_NUMBER"])

        data = data.merge(contrat_history_s, how="left")

        # Merging with all nature codes:
        for df in df_nature_codes:
            data = data.merge(df, how="left")

        return data

    def read_nature_codes(self):
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

        return (nature_code_eau_chaude, nature_code_energie,
                nature_code_fonction, nature_code_installation,
                nature_code_specification)

    def read_organisation(self):
        return pd.read_csv(
            RAW_DATA_DIR / "organisation.csv", sep="|", encoding="Latin-1")

    def read_equipement(self):
        equipement_df = pd.read_csv(
            RAW_DATA_DIR / "equipment.csv", sep="|", encoding="Latin-1")
        dt_cols = ["INSTALL_DATE", "RACHAT_DATE"]
        return to_datetimes(dt_cols, equipement_df)

    def read_contract_history(self):
        contrat_history_df = pd.read_csv(
            RAW_DATA_DIR / "contract_history.csv", sep="|", encoding="Latin-1")
        dt_cols = [
            "CRE_DATE", "UPD_DATE", "DATE_RESILIATION", "DATE_DEBUT",
            "DATE_FIN"
        ]
        return to_datetimes(dt_cols, contrat_history_df)

    def read_intervention_test(self):
        intervention_test_df = pd.read_csv(
            RAW_DATA_DIR / "intervention_test.csv",
            sep="|",
            encoding="Latin-1")
        dt_cols = [
            "SCHEDULED_START_DATE", "SCHEDULED_END_DATE", "CRE_DATE_GZL"
        ]
        return to_datetimes(dt_cols, intervention_test_df)

    def read_intervention_train(self):
        intervention_train_df = pd.read_csv(
            RAW_DATA_DIR / "intervention_train.csv",
            sep="|",
            encoding="Latin-1")
        dt_cols = [
            "SCHEDULED_START_DATE", "SCHEDULED_END_DATE", "CRE_DATE_GZL"
        ]
        return to_datetimes(dt_cols, intervention_train_df)

    def read_intervention_history(self):
        intervention_history_df = pd.read_csv(
            RAW_DATA_DIR / "intervention_history.csv",
            sep="|",
            encoding="Latin-1")
        dt_cols = [
            "DATE_SAISIE_RETOUR",
            "SCHEDULED_START_DATE",
            "SCHEDULED_END_DATE",
            "ACTUAL_START_DATE",
            "ACTUAL_END_DATE",
            "CRE_DATE_GZL",
        ]
        return to_datetimes(dt_cols, intervention_history_df)

    def read_all(self):
        with Timer("Reading organisation.csv"):
            df_orga = self.read_organisation()

        with Timer("Reading equipment.csv"):
            df_equipement = self.read_equipement()

        with Timer("Reading contract_history.csv"):
            df_contract_history = self.read_contract_history()

        with Timer("Reading intervention_test.csv"):
            df_intervention_test = self.read_intervention_test()

        if not self.use_full_history:
            with Timer("Reading intervention_train.csv"):
                df_interv = self.read_intervention_train()
        else:
            with Timer("Reading intervention_history.csv"):
                df_interv = self.read_intervention_history()

        df_nature_codes = self.read_nature_codes()
        return (df_orga, df_equipement, df_contract_history,
                df_intervention_test, df_interv, df_nature_codes)

    def convert(self):

        with Timer('-----Reading all csv-----', at_enter=True):
            (df_orga, df_equipement, df_contract_history, df_intervention_test,
             df_interv, df_nature_codes) = self.read_all()

        with Timer("Merging all train set"):
            train_data = self._merge_them_all(
                df_interv, df_orga, df_equipement, df_contract_history,
                *df_nature_codes)
            for col in STR_COLS:
                train_data[col] = train_data[col].astype(str)
            for col in NUMERIC_COLS:
                train_data[col] = pd.to_numeric(
                    train_data[col], downcast='signed')
            if 'target' in train_data:
                train_data[col] = pd.to_numeric(
                    train_data[col], downcast='signed')

        if self.use_full_history:
            pathfile = DATA_DIR / "fullhist_train.parquet.gzip"
        else:
            pathfile = DATA_DIR / "train.parquet.gzip"

        with Timer('Saving into {}'.format(pathfile)):
            train_data.to_parquet(
                pathfile, compression="gzip", engine=self.engine)

        with Timer("Merging all test set"):
            test_data = self._merge_them_all(
                df_intervention_test, df_orga, df_equipement,
                df_contract_history, *df_nature_codes)
            for col in STR_COLS:
                test_data[col] = test_data[col].astype(str)
            for col in NUMERIC_COLS:
                test_data[col] = pd.to_numeric(
                    test_data[col], downcast='signed')
            pathfile = DATA_DIR / "test.parquet.gzip"

        with Timer('Saving into {}'.format(pathfile)):
            test_data.to_parquet(
                pathfile, compression="gzip", engine=self.engine)
