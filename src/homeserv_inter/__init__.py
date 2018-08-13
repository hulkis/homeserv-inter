import os
from pathlib import Path

from pkg_resources import DistributionNotFound, get_distribution

try:
    _dist = get_distribution("homeserv_inter")

    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)

    if not here.startswith(os.path.join(dist_loc, "homeserv_inter")):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = "Version not found."
else:
    __version__ = _dist.version


PKG_DIR = Path(__file__).parent.parent.parent
SRC_DIR = Path(__file__).parent
DATA_DIR = PKG_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

numeric_cols = [
    "INSTANCE_ID",
    "INCIDENT_NUMBER",
    "INCIDENT_TYPE_ID",
    "NB_PASSAGE",
    "MILLESIME",
    "AUTEUR_INCIDENT",
    "RESOURCE_ID",
    "LOCATION_ID",
    "ORGANISATION_ID",
    "PARTY_ID_OCC",
    "MARQUE_CODE",
    "CODE_POSTAL",
    "POINTS_FIDEL",
    "CODE_ENERGIE",
    "L1_ORGANISATION_ID",
    "L2_ORGANISATION_ID",
    "L2_ORGA_CODE_POSTAL",
    "CIA",
    "ORGANISATION_CODE",
    "CONTRACT_NUMBER",
    "CONTRAT_TARIF",
    "PRIX_FACTURE",
    "PRIX_FORMULE",
    "PRIX_OPTION",
    "NUM_CAMPAGNE",
]
str_cols = [
    "INCIDENT_TYPE_NAME",
    "TYPE_BI",
    "PROBLEM_CODE",
    "PROBLEM_DESC",
    "ORIGINE_INCIDENT",
    "COMMENTAIRE_BI",
    "SS_TR_FLAG",
    "TYPE_UT",
    "GRAVITE",
    "TYPE_OCC",
    "RACHAT_CODE",
    "RACHAT_LIB",
    "NATURE_CODE",
    "MARQUE_LIB",
    "MODELE_CODE",
    "MODELE_LIB",
    "USAGE_LOCAL",
    "LOCALISATION_ORGANISME",
    "COMPLEMENT_RUE",
    "ESCALIER",
    "ETAGE",
    "NUMERO",
    "RUE",
    "PAYS",
    "TYPE_VOIE",
    "VILLE",
    "STOP_PHONING",
    "CODE_GEN_EQUIPEMENT",
    "CODE_FONCTION",
    "CODE_INSTALLATION",
    "CODE_SPECIFICATION",
    "CODE_EAU_CHAUDE",
    "L1_NAME",
    "L2_NAME",
    "ADRESSE",
    "L2_ORGA_VILLE",
    "STS_CODE",
    "CONTRACT_MODIFICATEUR",
    "FORMULE",
    "OPTION",
    "CONDITION_REGLEMENT",
    "MOTIF_RESILIATION",
    "RENOUVELLEMENT_AGENCE",
    "EAU_CHAUDE",
    "ENERGIE",
    "FONCTION",
    "INSTALLATION",
    "SPECIFICATION",
]

all_cols = [
    "INSTANCE_ID",
    "INCIDENT_NUMBER",
    "INCIDENT_TYPE_ID",
    "INCIDENT_TYPE_NAME",
    "TYPE_BI",
    "NB_PASSAGE",
    "MILLESIME",
    "PROBLEM_CODE",
    "PROBLEM_DESC",
    "AUTEUR_INCIDENT",
    "ORIGINE_INCIDENT",
    "COMMENTAIRE_BI",
    "SS_TR_FLAG",
    "TYPE_UT",
    "GRAVITE",
    "RESOURCE_ID",
    "SCHEDULED_START_DATE",
    "SCHEDULED_END_DATE",
    "CRE_DATE_GZL",
    "target",
    "LOCATION_ID",
    "ORGANISATION_ID",
    "PARTY_ID_OCC",
    "TYPE_OCC",
    "INSTALL_DATE",
    "RACHAT_CODE",
    "RACHAT_LIB",
    "RACHAT_DATE",
    "NATURE_CODE",
    "MARQUE_CODE",
    "MARQUE_LIB",
    "MODELE_CODE",
    "MODELE_LIB",
    "USAGE_LOCAL",
    "LOCALISATION_ORGANISME",
    "COMPLEMENT_RUE",
    "CODE_POSTAL",
    "ESCALIER",
    "ETAGE",
    "NUMERO",
    "RUE",
    "PAYS",
    "TYPE_VOIE",
    "VILLE",
    "POINTS_FIDEL",
    "STOP_PHONING",
    "CODE_GEN_EQUIPEMENT",
    "CODE_FONCTION",
    "CODE_ENERGIE",
    "CODE_INSTALLATION",
    "CODE_SPECIFICATION",
    "CODE_EAU_CHAUDE",
    "L1_ORGANISATION_ID",
    "L1_NAME",
    "L2_ORGANISATION_ID",
    "L2_NAME",
    "ADRESSE",
    "L2_ORGA_CODE_POSTAL",
    "L2_ORGA_VILLE",
    "CIA",
    "ORGANISATION_CODE",
    "STS_CODE",
    "CONTRACT_NUMBER",
    "CONTRACT_MODIFICATEUR",
    "CRE_DATE",
    "UPD_DATE",
    "DATE_RESILIATION",
    "DATE_DEBUT",
    "DATE_FIN",
    "FORMULE",
    "OPTION",
    "CONTRAT_TARIF",
    "PRIX_FACTURE",
    "CONDITION_REGLEMENT",
    "MOTIF_RESILIATION",
    "RENOUVELLEMENT_AGENCE",
    "PRIX_FORMULE",
    "PRIX_OPTION",
    "NUM_CAMPAGNE",
    "EAU_CHAUDE",
    "ENERGIE",
    "FONCTION",
    "INSTALLATION",
    "SPECIFICATION",
]

nlp_cols = ["COMMENTAIRE_BI"]

label_cols = list(str_cols)
for col in nlp_cols:
    label_cols.remove(col)

timestamp_cols = [
    "CRE_DATE",
    "CRE_DATE_GZL",
    "DATE_DEBUT",
    "DATE_FIN",
    "DATE_RESILIATION",
    "INSTALL_DATE",
    "RACHAT_DATE",
    "SCHEDULED_END_DATE",
    "SCHEDULED_START_DATE",
    "UPD_DATE",
]
