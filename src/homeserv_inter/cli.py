"""Console script for homeserv_inter."""
import fire

from homeserv_inter.datahandler import generate_cleaned_sets
from homeserv_inter.geo import GeocoderHomeserv
from homeserv_inter.model import CatBoostHomService, LgbHomeService, XgbHomeService
from homeserv_inter.raw import HomeServiceRaw


def main():
    return fire.Fire(
        {
            "raw": HomeServiceRaw,
            "convert-cleaned": generate_cleaned_sets,
            "geo": GeocoderHomeserv,
            "lgb": LgbHomeService,
            "xgb": XgbHomeService,
            "cgb": CatBoostHomService,
        }
    )
