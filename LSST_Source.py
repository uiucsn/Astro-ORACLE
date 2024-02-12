import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class LSST_Source:

    # List of time series features actually stored in the instance of the class.
    time_series_features = ['MJD', 'BAND', 'PHOTFLAG', 'FLUXCAL', 'FLUXCALERR']

    # List of other features actually stored in the instance of the class.
    other_features = ['RA', 'DEC', 'MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'VPEC', 'VPEC_ERR', 'HOSTGAL_FLAG', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_DDLR', 'HOSTGAL_CONFUSION', 'HOSTGAL_LOGMASS', 'HOSTGAL_LOGMASS_ERR', 'HOSTGAL_LOGSFR', 'HOSTGAL_LOGSFR_ERR', 'HOSTGAL_LOGsSFR', 'HOSTGAL_LOGsSFR_ERR', 'HOSTGAL_COLOR', 'HOSTGAL_COLOR_ERR', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y', 'HOSTGAL_MAGERR_u', 'HOSTGAL_MAGERR_g', 'HOSTGAL_MAGERR_r', 'HOSTGAL_MAGERR_i', 'HOSTGAL_MAGERR_z', 'HOSTGAL_MAGERR_Y']

    # Pass band to color dict
    colors = {
        'u': 'blue',
        'g': 'green',
        'r': 'red',
        'i': 'teal',
        'z': 'orange',
        'Y': 'purple',
    }

    def __init__(self, parquet_row, class_label) -> None:
        """Create an LSST_Source object to store both photometric and host galaxy data from the Elasticc simulations.

        Args:
            parquet_row (_type_): A row from the polars data frame that was generated from the Elasticc FITS files using fits_to_parquet.py
            class_label (str): The Elasticc class label for this LSST_Source object.
        """

        # Set all the class attributes
        setattr(self, 'ELASTICC_class', class_label)
        setattr(self, 'SNID', parquet_row['SNID'].to_numpy()[0])

        for key in parquet_row.columns:
            if key in self.other_features:
                setattr(self, key, parquet_row[key].to_numpy()[0])
            elif key in self.time_series_features:
                setattr(self, key, parquet_row[key].to_numpy()[0])

        # Run processing code on the light curves
        self.process_lightcurve()

    
    def process_lightcurve(self) -> None:
        """Process the flux information with phot flags. Processing is done using the following steps:
        1. Remove saturations.
        2. Keep the last non-detection before the trigger (if available).
        3. Keep non-detection after the trigger and before the last detection (if available). 
        Finally, all the time series data is modified to conform to the 4 steps mentioned above.
        """

        # Remove saturations from the light curves
        saturation_mask =  (self.PHOTFLAG & 1024) == 0 

        # Alter time series data to remove saturations
        for time_series_feature in self.time_series_features:
            setattr(self, time_series_feature, getattr(self, time_series_feature)[saturation_mask])

        # Find the first and last detections
        detection_mask = (self.PHOTFLAG & 4096) != 0
        first_detection_idx = np.where(detection_mask)[0][0]
        last_detection_idx = np.where(detection_mask)[0][-1]

        # Allow for one non detection before the trigger and add all non-detection after the trigger and before the last detection
        ts_start = max(first_detection_idx - 1, 0)
        ts_end = last_detection_idx
        idx = range(ts_start,ts_end)

        # Alter time series data to only preserve all data between trigger and last detections + 1 non detection before the trigger
        for time_series_feature in self.time_series_features:
            setattr(self, time_series_feature, getattr(self, time_series_feature)[idx])


    def plot_flux_curve(self) -> None:
        """Plot the SNANA calibrated flux vs time plot for all the data in the processed time series. All detections are marked with a star while non detections are marked with dots. Observations are color codded by their passband. This function is fundamentally a visualization tool and is not intended for making plots for papers.
        """

        # Colorize the data
        c = [self.colors[band] for band in self.BAND]
        patches = [mpatches.Patch(color=self.colors[band], label=band, linewidth=1) for band in self.colors]
        fmts = np.where((self.PHOTFLAG & 4096) != 0, '*', '.')

        # Plot flux time series
        for i in range(len(self.MJD)):
            plt.errorbar(x=self.MJD[i], y=self.FLUXCAL[i], yerr=self.FLUXCALERR[i], color=c[i], fmt=fmts[i], markersize = '10')

        # Labels
        plt.title(f"SNID: {self.SNID} | CLASS: {self.ELASTICC_class}")
        plt.xlabel('Time (MJD)')
        plt.ylabel('Calibrated Flux')
        plt.legend(handles=patches)

        plt.show()

    def get_classification_hierarchy(self):
        pass

    def get_ML_Tensor(self):

        # Dataframe for time series data.
        df_ts = pd.DataFrame()

        # Find time since last observation
        df_ts['time_since_first_obs'] = self.MJD - self.MJD[0]

        # 1 if it was a detection, zero otherwise.
        df_ts['detection_flag'] = np.where((self.PHOTFLAG & 4096) != 0, 1, 0)

        # Transform flux cal and flux cal err to more manageable values (more consistent order of magnitude)
        df_ts['FLUXCAL_asinh'] = np.arcsinh(self.FLUXCAL)
        df_ts['FLUXCALERR_asinh'] = np.arcsinh(self.FLUXCALERR)

        # One hot encoding for the pass band
        df_ts['band_label'] = self.BAND
        one_hot_encoding = pd.get_dummies(df_ts['band_label'], dtype=int)
        df_ts = df_ts.drop('band_label', axis=1)
        df_ts = df_ts.join(one_hot_encoding)
        np_ts = df_ts.to_numpy()

        # Consistency check
        assert np_ts.shape[0] == len(self.MJD), "Length of time series tensor does not match the number of mjd values."

        # Array for static features.
        np_static = []
        for other_feature in self.other_features:
            np_static.append(getattr(self, other_feature))
        np_static = np.array(np_static)

        pass

    def get_augmented_sources(self, min_length = 2) -> list:
        """Function to augment the length of time series data in an LSST_Source object and produce a list of several LSST_Source objects each with differing length. All objects start with the left most observation (earliest observation) with window length increasing by 1 every time. The minimum length of the time series can be adjusted using the min_length argument.

        Args:
            min_length (int, optional): Minimum length of time series data in the augmented sources. Defaults to 2. This values is chosen to ensure at least 1 detection.

        Returns:
            list: List of augmented LSST_Source objects. All of these objects share host properties with the parent object, however the time series lengths are different.
        """

        assert min_length >= 1, "Minimum length of the light curve should be >= 1."
        
        augmented_sources = []
        time_series_length = len(self.MJD)

        # Loop to apply windowing
        for data_length in range(time_series_length, min_length - 1, -1):

            # Create a copy of the object
            augmented_source = copy.deepcopy(self)
            
            # Trim the time series data 
            for time_series_feature in self.time_series_features:
                original_source_attribute = getattr(augmented_source, time_series_feature)
                setattr(augmented_source, time_series_feature, original_source_attribute[:data_length])

            # Append the data to the list if there is any time series data.
            if len(augmented_source.MJD >= 0):
                augmented_sources.append(augmented_source)

        return augmented_sources

    def __str__(self) -> str:

        to_return = str(vars(self))
        return to_return


if __name__=='__main__':


    pass

