from pathlib import Path

PKG_DIR = Path(__file__).parent.parent.parent
SRC_DIR = Path(__file__).parent
DATA_DIR = PKG_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
MODEL_DIR = PKG_DIR / "model_bank"
TUNING_DIR = PKG_DIR / "tuning_hist"
RESULT_DIR = PKG_DIR / "results"

for d in [DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, MODEL_DIR, TUNING_DIR, RESULT_DIR]:
    if not d.exists():
        d.mkdir(parents=True)

ALL_COLS = ["INSTANCE_ID", "INCIDENT_NUMBER", "INCIDENT_TYPE_ID", "INCIDENT_TYPE_NAME", "TYPE_BI", "NB_PASSAGE", "MILLESIME", "PROBLEM_CODE", "PROBLEM_DESC", "AUTEUR_INCIDENT", "ORIGINE_INCIDENT", "COMMENTAIRE_BI", "SS_TR_FLAG", "TYPE_UT", "GRAVITE", "RESOURCE_ID", "SCHEDULED_START_DATE", "SCHEDULED_END_DATE", "CRE_DATE_GZL", "target", "LOCATION_ID", "ORGANISATION_ID", "PARTY_ID_OCC", "TYPE_OCC", "INSTALL_DATE", "RACHAT_CODE", "RACHAT_LIB", "RACHAT_DATE", "NATURE_CODE", "MARQUE_CODE", "MARQUE_LIB", "MODELE_CODE", "MODELE_LIB", "USAGE_LOCAL", "LOCALISATION_ORGANISME", "COMPLEMENT_RUE", "CODE_POSTAL", "ESCALIER", "ETAGE", "NUMERO", "RUE", "PAYS", "TYPE_VOIE", "VILLE", "POINTS_FIDEL", "STOP_PHONING", "CODE_GEN_EQUIPEMENT", "CODE_FONCTION", "CODE_ENERGIE", "CODE_INSTALLATION", "CODE_SPECIFICATION", "CODE_EAU_CHAUDE", "L1_ORGANISATION_ID", "L1_NAME", "L2_ORGANISATION_ID", "L2_NAME", "ADRESSE", "L2_ORGA_CODE_POSTAL", "L2_ORGA_VILLE", "CIA", "ORGANISATION_CODE", "STS_CODE", "CONTRACT_NUMBER", "CONTRACT_MODIFICATEUR", "CRE_DATE", "UPD_DATE", "DATE_RESILIATION", "DATE_DEBUT", "DATE_FIN", "FORMULE", "OPTION", "CONTRAT_TARIF", "PRIX_FACTURE", "CONDITION_REGLEMENT", "MOTIF_RESILIATION", "RENOUVELLEMENT_AGENCE", "PRIX_FORMULE", "PRIX_OPTION", "NUM_CAMPAGNE", "EAU_CHAUDE", "ENERGIE", "FONCTION", "INSTALLATION", "SPECIFICATION"]
NUMERIC_COLS = ["INSTANCE_ID", "INCIDENT_NUMBER", "INCIDENT_TYPE_ID", "NB_PASSAGE", "MILLESIME", "AUTEUR_INCIDENT", "RESOURCE_ID", "LOCATION_ID", "ORGANISATION_ID", "PARTY_ID_OCC", "MARQUE_CODE", "CODE_POSTAL", "POINTS_FIDEL", "CODE_ENERGIE", "L1_ORGANISATION_ID", "L2_ORGANISATION_ID", "L2_ORGA_CODE_POSTAL", "CIA", "ORGANISATION_CODE", "CONTRACT_NUMBER", "CONTRAT_TARIF", "PRIX_FACTURE", "PRIX_FORMULE", "PRIX_OPTION", "NUM_CAMPAGNE"]
FLOAT_COLS = []
STR_COLS = ["INCIDENT_TYPE_NAME", "TYPE_BI", "PROBLEM_CODE", "PROBLEM_DESC", "ORIGINE_INCIDENT", "COMMENTAIRE_BI", "SS_TR_FLAG", "TYPE_UT", "GRAVITE", "TYPE_OCC", "RACHAT_CODE", "RACHAT_LIB", "NATURE_CODE", "MARQUE_LIB", "MODELE_CODE", "MODELE_LIB", "USAGE_LOCAL", "LOCALISATION_ORGANISME", "COMPLEMENT_RUE", "ESCALIER", "ETAGE", "NUMERO", "RUE", "PAYS", "TYPE_VOIE", "VILLE", "STOP_PHONING", "CODE_GEN_EQUIPEMENT", "CODE_FONCTION", "CODE_INSTALLATION", "CODE_SPECIFICATION", "CODE_EAU_CHAUDE", "L1_NAME", "L2_NAME", "ADRESSE", "L2_ORGA_VILLE", "STS_CODE", "CONTRACT_MODIFICATEUR", "FORMULE", "OPTION", "CONDITION_REGLEMENT", "MOTIF_RESILIATION", "RENOUVELLEMENT_AGENCE", "EAU_CHAUDE", "ENERGIE", "FONCTION", "INSTALLATION", "SPECIFICATION"]
TIMESTAMP_COLS = ["CRE_DATE", "CRE_DATE_GZL", "DATE_DEBUT", "DATE_FIN", "DATE_RESILIATION", "INSTALL_DATE", "RACHAT_DATE", "SCHEDULED_END_DATE", "SCHEDULED_START_DATE", "UPD_DATE"]

NLP_COLS = ["COMMENTAIRE_BI"]

SEED = 42


DROPCOLS = [
    # 'organisation' table:
    "L1_ORGANISATION_ID", # Keep L1_NAME label encoded
    "L2_ORGANISATION_ID", # Keep L2_NAME label encoded
    "ORGANISATION_ID",    # IS equal to L2_ORGANISATION_ID

    # 'contract_history' table:
    # seems all important

    # 'equipment' table:
    # "INSTANCE_ID",  # primary key
    "MARQUE_CODE",  # keep MARQUE_LIB label encoded
    "MODELE_CODE",  # keep MODELE_LIB label encoded

    # 'intervetion' table:
    "INCIDENT_NUMBER",  # primary key
    "INCIDENT_TYPE_ID", # keep INCIDENT_TYPE_NAME label encoded
    "TYPE_BI",          # seems like a repetition of INCIDENT_TYPE_NAME
    "PROBLEM_CODE",     # keep PROBLEM_DESC label encoded

    # 'nature code' table:
    "CODE_EAU_CHAUDE",    # keep EAU_CHAUDE label encoded
    "CODE_ENERGIE",       # keep ENERGIE label encoded
    "CODE_FONCTION",      # keep FONCTION label encoded
    "CODE_INSTALLATION",  # keep INSTALLATION label encoded
    "CODE_SPECIFICATION", # keep SPECIFICATION label encoded
]

# What should not be considered as categorical feature due to too large number:
_not_categorical = [
    'COMMENTAIRE_BI',
    'MODELE_CODE',
    'NUMERO',
    'MODELE_LIB',
    'LOCALISATION_ORGANISME',
    'RUE',
    'VILLE',
    'CONTRACT_MODIFICATEUR',
]

# Categorical features:
NUM_COLS_CATEGORICAL = [  # Numeric columns that should be considered categorical
    'INSTANCE_ID', 'INCIDENT_TYPE_ID', 'RESOURCE_ID', 'LOCATION_ID', 'ORGANISATION_ID', 'PARTY_ID_OCC', 'L1_ORGANISATION_ID', 'L2_ORGANISATION_ID',
    'INCIDENT_NUMBER', 'CONTRACT_NUMBER',
]

LABEL_COLS = list(STR_COLS)
for col in NLP_COLS:
    LABEL_COLS.remove(col)
LABEL_COLS += NUM_COLS_CATEGORICAL


HIGH_NUM_CAT = ['RUE',
                'RESOURCE_ID',
                # 'INSTANCE_ID',
                'LOCATION_ID',
                # 'PARTY_ID_OCC',
                'VILLE',
                # 'CONTRACT_NUMBER',
                'LOCATION_ID',
                ]

MEDIUM_NUM_CAT = ['MARQUE_LIB', 'TYPE_VOIE', 'OPTION', 'CODE_GEN_EQUIPEMENT']

SMALL_NUM_CAT = [
    'FORMULE',
    'ORIGINE_INCIDENT',
    'TYPE_OCC',
    'USAGE_LOCAL',
    'CONDITION_REGLEMENT',
]


CATBOOST_FEATURES = NUM_COLS_CATEGORICAL + STR_COLS

LOW_IMPORTANCE_FEATURES = [
    'USAGE_LOCAL',
    'nbdays_duration_of_contract',

    'SCHEDULED_START_DATE_weekday',
    'SCHEDULED_START_DATE_week',
    'SCHEDULED_START_DATE_year',
    'SCHEDULED_START_DATE_month',
    'SCHEDULED_END_DATE_year',

    'UPD_DATE_year',

    'CRE_DATE_year',

    'DATE_DEBUT_year',

    'DATE_FIN_month',
    'DATE_FIN_year',

    'RACHAT_DATE_day',
    'RACHAT_DATE_dayofyear',
    'RACHAT_DATE_weekday',
    'RACHAT_DATE_week',
    'RACHAT_DATE_month',
    'RACHAT_DATE_year',

    'DATE_RESILIATION_week',
    'DATE_RESILIATION_weekday',
    'DATE_RESILIATION_year',
    'DATE_RESILIATION_month',

    'TYPE_UT',
    'RENOUVELLEMENT_AGENCE',
    'SS_TR_FLAG',
    'PROBLEM_DESC',
    'PAYS',
    'nbdays_delta_intervention_contract_start',
    'nbdays_duration_of_intervention',

    'INCIDENT_TYPE_NAME_label0', 'FORMULE_INTEGRAL***', 'FORMULE_SECURITE',
    'INCIDENT_TYPE_NAME_label1', 'TYPE_OCC_PROPRIETAIRE',
    'CONDITION_REGLEMENT_PREL01', 'FORMULE_ESSENTIEL', 'FORMULE_SERENITE',
    'USAGE_LOCAL_HABI', 'CONDITION_REGLEMENT_PREL04',
    'ORIGINE_INCIDENT_nan', 'USAGE_LOCAL_PRO',
    'ORIGINE_INCIDENT_indirect_contact', 'FORMULE_Sécurité_Pack',
    'ORIGINE_INCIDENT_Venue_client', 'ORIGINE_INCIDENT_Courrier',
    'FORMULE_MULTISERVICES', 'ORIGINE_INCIDENT_Télé-alarme',
    'USAGE_LOCAL_nan', 'CONDITION_REGLEMENT_nan', 'FORMULE_nan',
]
