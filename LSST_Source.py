import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class LSST_Source:

    # List of time series features actually stored in the instance of the class.
    time_series_features = ['MJD', 'BAND', 'PHOTFLAG', 'FLUXCAL', 'FLUXCALERR']

    # List of other features actually stored in the instance of the class.
    other_features = ['RA', 'DEC', 'MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'VPEC', 'VPEC_ERR', 'HOSTGAL_FLAG', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_DDLR', 'HOSTGAL_CONFUSION', 'HOSTGAL_LOGMASS', 'HOSTGAL_LOGMASS_ERR', 'HOSTGAL_LOGSFR', 'HOSTGAL_LOGSFR_ERR', 'HOSTGAL_LOGsSFR', 'HOSTGAL_LOGsSFR_ERR', 'HOSTGAL_COLOR', 'HOSTGAL_COLOR_ERR', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y', 'HOSTGAL_MAGERR_u', 'HOSTGAL_MAGERR_g', 'HOSTGAL_MAGERR_r', 'HOSTGAL_MAGERR_i', 'HOSTGAL_MAGERR_z', 'HOSTGAL_MAGERR_Y', 'SIM_EXPOSURE_u', 'SIM_EXPOSURE_g', 'SIM_EXPOSURE_r', 'SIM_EXPOSURE_i', 'SIM_EXPOSURE_z', 'SIM_EXPOSURE_Y']

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

        # Remove saturations from the light curves
        saturation_mask =  (self.PHOTFLAG & 1024) == 0

        detection_mask = (self.PHOTFLAG & 4096) != 0
        first_detection_idx = np.where(detection_mask)[0][0]
        last_detection_idx = np.where(detection_mask)[0][-1]

        # Allow for one non detection before the trigger and add all non-detection after the trigger and before the last detection
        ts_start = max(first_detection_idx - 1, 0)
        ts_end = last_detection_idx + 1
        idx = range(ts_start,ts_end)

        # Alter time series data to remove saturations and only preserve all data between trigger and last detections + 1 non detection before the trigger
        self.MJD = self.MJD[saturation_mask][idx]
        self.BAND = self.BAND[saturation_mask][idx]
        self.FLUXCAL = self.FLUXCAL[saturation_mask][idx]
        self.FLUXCALERR = self.FLUXCALERR[saturation_mask][idx]
        self.PHOTFLAG = self.PHOTFLAG[saturation_mask][idx]


    def plot_flux_curve(self) -> None:

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
        pass

    def get_augmented_sources(self, min_length = 2) -> list:

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

            # Append the data to the list
            augmented_sources.append(augmented_source)

        return augmented_sources

    def __str__(self) -> str:

        to_return = str(vars(self))
        return to_return


if __name__=='__main__':


    pass

