import logging
import os

import geopy
import googlemaps
import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy import GoogleV3, distance
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
                 debug=False,
                 source='openstreet',
                 token=os.getenv('GOOGLE_MAPS_TOKEN')):
        """
        Args:
            dataset (str): 'test' or 'train'
            source (str): where to get the datas, in ['gmaps', 'openstreet']
        """
        self.source = source
        self.dataset = dataset
        self.debug = debug

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

    def get_distances(self, force_download=False):
        if force_download:
            self._get_coordinates_clients()

        self._merge_coordinates()

    def _get_coordinates_clients(self):

        print('Reading dataset...')
        with Timer("Reading {} set".format(self.dataset)):
            df = pd.read_parquet(
                DATA_DIR / "{}.parquet.gzip".format(self.dataset))

            if self.debug:
                df = df.head(10)

        g = self.geolocator

        # Clients
        _df = df.drop_duplicates(subset=['VILLE', 'PAYS'])

        def _bar(row):
            return g.geocode({'city': row['VILLE'], 'country': row['PAYS']})

        print('Starting to load clients coordinates...')
        _df['LONG_LAT_CLIENT'] = _df.apply(_bar, axis=1)

        print('Saving clients coordinates...')
        fcache_name = '{}_{}set_geocodes_{}.pkl'.format(
            'VILLE_CLIENT', self.dataset, self.source)
        with Timer('Stored in {}'.format(fcache_name)):
            _df.to_pickle(DATA_DIR / fcache_name)

        # L2 organization
        __df = df.drop_duplicates(subset=['L2_ORGA_VILLE'])

        def _foo(row):
            # print({'city': row['L2_ORGA_VILLE'], 'country': 'FR'})
            return g.geocode({'city': row['L2_ORGA_VILLE'], 'country': 'FR'})
        print('Starting to load L2 coordinates...')
        __df['LONG_LAT_L2'] = _df.apply(_foo, axis=1)

        print('Saving L2 coordinates...')
        fcache_name = '{}_{}set_geocodes_{}.pkl'.format(
            'VILLE_L2', self.dataset, self.source)
        with Timer('Stored in {}'.format(fcache_name)):
            __df.to_pickle(DATA_DIR / fcache_name)

    def _merge_coordinates(self):

        with Timer("Reading {} set".format(self.dataset)):
            df = pd.read_parquet(
                DATA_DIR / "{}.parquet.gzip".format(self.dataset))

            if self.debug:
                df = df.head(10)

        # Clients
        fcache_name = '{}_{}set_geocodes_{}.pkl'.format(
            'VILLE_CLIENT', self.dataset, self.source)
        _df = pd.read_pickle(DATA_DIR / fcache_name)

        # L2 organization
        fcache_name = '{}_{}set_geocodes_{}.pkl'.format(
            'VILLE_L2', self.dataset, self.source)
        __df = pd.read_pickle(DATA_DIR / fcache_name)

        print('Merging data...')
        df = df.merge(_df[['LONG_LAT_CLIENT', 'VILLE', 'PAYS']], on=['VILLE', 'PAYS'], how='outer')
        df = df.merge(__df[['LONG_LAT_L2', 'L2_ORGA_VILLE']], on=['L2_ORGA_VILLE'], how='outer')

        def _foo(row):
            try:
                dist = int(distance.distance(row['LONG_LAT_CLIENT'].point,
                                             row['LONG_LAT_L2'].point).km)
            except:
                dist = np.nan

            return dist

        print('Computing distances...')
        df['DIST_CLIENT_L2'] = df.apply(_foo, axis=1)

        print('Saving distances...')
        fcache_name = '{}_{}set_geocodes_{}.pkl'.format(
            'DISTANCE', self.dataset, self.source)
        with Timer('Stored in {}'.format(fcache_name)):
            df.to_pickle(DATA_DIR / fcache_name)





# def generate_distance_matrix(dataset, token=os.getenv('GOOGLE_MAPS_TOKEN')):
#     if token is None:
#         raise Exception('I need a gmap token!')
#
#     gmaps = googlemaps.Client(key=token)
#
#     with Timer("Reading {} set".format(dataset)):
#         df = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))
#
#     df = GeocoderHomeserv.compute_address(df)
#
#     total_size = df.shape[0]
#     cols = [
#         'distance_l2_particulier_meters', 'duration_l2_particulier_seconds'
#     ]
#
#     with Timer('Requesting Google API', at_enter=True):
#         for i, row in df.iterrows():
#             addpart = row['address_particulier']
#             addl2 = row['address_l2_orga']
#             msg = '{}% --> from {} to {}'.format(i // total_size, addpart,
#                                                  addl2)
#             with Timer(msg):
#                 try:
#                     res = gmaps.distance_matrix(
#                         origins=addl2, destinations=addpart)
#                     if len(res['rows']) != 1 and len(
#                             res['rows'][0]['elements']) != 1:
#                         logger.debug(res)
#                     res = res['rows'][0]['elements'][0]
#                     df.loc[i, 'distance_l2_particulier_meters'] = res[
#                         'distance']['value']
#                     df.loc[i, 'duration_l2_particulier_seconds'] = res[
#                         'duration']['value']
#                 except:
#                     df.loc[i, 'distance_l2_particulier_meters'] = np.nan
#                     df.loc[i, 'duration_l2_particulier_seconds'] = np.nan
#
#             if i != 0 and i % 100 == 0:
#                 fcache_name = 'INCOMPLETE_{}set_distances_gmap_apis.csv'.format(
#                     dataset)
#                 with Timer('-------- Caching in {}'.format(fcache_name)):
#                     df[cols].to_csv(DATA_DIR / fcache_name)
#
#     df[cols].to_csv(DATA_DIR / '{}_distances_gmap_apis.csv'.format(dataset))
