"""Console script for homeserv_inter."""
import fire

from homeserv_inter.datahandler import generate_cleaned_sets
from homeserv_inter.geo import GeocoderHomeserv
from homeserv_inter.model import CatBoostHomService, LgbHomeService, XgbHomeService
from homeserv_inter.raw import convert_csv_to_parquet


def main():
    return fire.Fire(
        {
            "convert-raw": convert_csv_to_parquet,
            "convert-cleaned": generate_cleaned_sets,
            "geo": GeocoderHomeserv,
            "lgb": LgbHomeService,
            "xgb": XgbHomeService,
            "cgb": CatBoostHomService,
        }
    )
