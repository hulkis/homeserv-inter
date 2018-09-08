from wax_toolbox import Timer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEncoderByColMissVal(BaseEstimator, TransformerMixin):
    already_nan_cleaned = False

    def __init__(self, columns):
        # List of column names in the DataFrame that should be encoded
        self.columns = columns
        # Dictionary storing a LabelEncoder for each column
        self.label_encoders = {}
        for el in self.columns:
            self.label_encoders[el] = LabelEncoder()

    def handle_nans(self, x):
        if not self.already_nan_cleaned:
            with Timer("Cleaning NaNs for label encoding"):
                # Fill missing values with the string 'NaN'
                x[self.columns] = x[self.columns].fillna("NaN")

                # str to replace interpreted as NaN:
                lst_str_nans = ["nan", "", "-"]
                for s in lst_str_nans:
                    for col in self.columns:
                        x[col] = x[col].replace(s, "NaN")

        self.already_nan_cleaned = True
        return x

    def fit(self, x):
        x = self.handle_nans(x)

        for el in self.columns:
            # Only use the values that are not 'NaN' to fit the Encoder
            a = x[el][x[el] != "NaN"]
            self.label_encoders[el].fit(a)
        return self

    def transform(self, x):
        x = self.handle_nans(x)

        for el in self.columns:
            # Only use the values that are not 'NaN' to fit the Encoder
            a = x[el][x[el] != "NaN"]
            # Store an ndarray of the current column
            b = x[el].get_values()
            # Replace the elements in the ndarray that are not 'NaN'
            # using the transformer
            b[b != "NaN"] = self.label_encoders[el].transform(a)
            # Overwrite the column in the DataFrame
            x[el] = b

            # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#categorical_feature
            # Replace string "NaN" by -1 as lgb: all negative values will be treated as missing values
            # SHIT ! seg fault when -1 at init booster time, go for np.nan
            # conversion at the moment we read the parquet file...
            x[el] = x[el].replace("NaN", -1)

            x[el] = pd.to_numeric(x[el], downcast='signed')

        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
