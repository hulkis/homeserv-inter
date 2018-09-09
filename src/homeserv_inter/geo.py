import logging
import os

import geopy
import googlemaps
import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

from homeserv_inter.constants import DATA_DIR
from wax_toolbox import Timer

tqdm.pandas()

logger = logging.getLogger(__name__)


class GeocoderHomeserv:
    @property
    def geolocator(self):
        if self.source == 'openstreet':
            geolocator = geopy.geocoders.Nominatim(user_agent='homeservice')
        elif self.source == 'gmaps':
            geolocator = geopy.GoogleV3(api_key=self.token)
        else:
            raise ValueError

        return geolocator

    def __init__(self,
                 dataset,
                 source='openstreet',
                 token=os.getenv('GOOGLE_MAPS_TOKEN')):
        """
        Args:
            dataset (str): 'test' or 'train'
            source (str): where to get the datas, in ['gmaps', 'openstreet']
        """
        self.source = source
        self.dataset = dataset

        if source == 'gmaps' and token is None:
            raise Exception('You choosed gmaps API but have no token.')

        self.token = token

    @staticmethod
    def compute_address(df):
        cols_of_interest = [
            'NUMERO', 'RUE', 'VILLE', 'PAYS', 'ADRESSE', 'L2_ORGA_VILLE'
        ]
        df = df[cols_of_interest]
        df = df.fillna('').replace('nan', '')

        df['address_particulier'] = (df['NUMERO'] + ' rue ' + df['RUE'] + ', '
                                     + df['VILLE'] + ' ' + df['PAYS'])
        df['address_l2_orga'] = (
            df['ADRESSE'] + ', ' +
            # df['L2_ORGA_CODE_POSTAL'].astype(int).astype(str) + ' ' +  # Buggy due to NaNs
            df['L2_ORGA_VILLE'])
        return df

    def request_geocode_datas(self, address_column):

        with Timer("Reading {} set".format(self.dataset)):
            df = pd.read_parquet(
                DATA_DIR / "{}.parquet.gzip".format(self.dataset))
            df = self.compute_address(df)

        geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)

        location_column = 'location_{}'.format(address_column)
        point_column = 'point_{}'.format(address_column)
        msg = 'Requesting geocode with {} API'.format(self.source)

        df = df[[address_column]].drop_duplicates()
        with Timer(msg):
            df[location_column] = df[address_column].progress_apply(geocode)
            df[point_column] = df[location_column].apply(
                lambda loc: tuple(loc.point) if loc else None)

        fcache_name = '{}_{}set_geocodes_{}.pkl'.format(
            address_column, self.dataset, self.source)
        with Timer('Stored in {}'.format(fcache_name)):
            df.to_pickle(DATA_DIR / fcache_name)


def generate_distance_matrix(dataset, token=os.getenv('GOOGLE_MAPS_TOKEN')):
    if token is None:
        raise Exception('I need a gmap token!')

    gmaps = googlemaps.Client(key=token)

    with Timer("Reading {} set".format(dataset)):
        df = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))

    df = compute_address(df)

    total_size = df.shape[0]
    cols = [
        'distance_l2_particulier_meters', 'duration_l2_particulier_seconds'
    ]

    with Timer('Requesting Google API', at_enter=True):
        for i, row in df.iterrows():
            addpart = row['address_particulier']
            addl2 = row['address_l2_orga']
            msg = '{}% --> from {} to {}'.format(i // total_size, addpart,
                                                 addl2)
            with Timer(msg):
                try:
                    res = gmaps.distance_matrix(
                        origins=addl2, destinations=addpart)
                    if len(res['rows']) != 1 and len(
                            res['rows'][0]['elements']) != 1:
                        logger.debug(res)
                    res = res['rows'][0]['elements'][0]
                    df.loc[i, 'distance_l2_particulier_meters'] = res[
                        'distance']['value']
                    df.loc[i, 'duration_l2_particulier_seconds'] = res[
                        'duration']['value']
                except:
                    df.loc[i, 'distance_l2_particulier_meters'] = np.nan
                    df.loc[i, 'duration_l2_particulier_seconds'] = np.nan

            if i != 0 and i % 100 == 0:
                fcache_name = 'INCOMPLETE_{}set_distances_gmap_apis.csv'.format(
                    dataset)
                with Timer('-------- Caching in {}'.format(fcache_name)):
                    df[cols].to_csv(DATA_DIR / fcache_name)

    df[cols].to_csv(DATA_DIR / '{}_distances_gmap_apis.csv'.format(dataset))
