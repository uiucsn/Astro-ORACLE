# ELAsTiCC-Classification
Hierarchical classification of ELAsTiCC 2.


## STEP 1 - Convert the data to a more usable form:

The ELAsTiCC 2 training data set contains 32 different classes of astrophysical objects. Each object has 80 FITS files - 40 which contain the photometry and the remaining contain other information like the host galaxy properties. 

We care about having all the information (i.e. light curves + host galaxy) information for each object in a convenient format before we start any data augmentation. For this reason, we bind the HEAD and PHOT FITS files, extract the relevant information and store it as parquets. 

The code used for this conversion is in `fits_to_parquet.py`. This code is modified from an earlier version written by Kostya [here](https://github.com/hombit/yad).

### Convert Elasticc 2 train dataset

- Download the data with [this link](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/ELASTICC2_TRAIN_02.tar.bz2)
- Unpack it to `data/data/elasticc2_train/raw`
- `cd data/data/elasticc2_train`
- Convert all the data to parquet: `ls raw | sed 's/ELASTICC2_TRAIN_02_//' | xargs -IXXX -P32 python3 ../../../fits_to_parquet.py raw/ELASTICC2_TRAIN_02_XXX parquet/XXX.parquet`

## Step 2 - Data Augmentation

The parquet files contain all of the information from the fits files, organized by source. However, we need to drop unnecessary features and augment our data before we can start training the models. 

We decompose the parquet rows into instances of the `LSST_Source` class (see LSST_Source.py). The features from the rows that are stored in the class are mentioned in the `time_series_features` and `other_features` lists (again, see LSST_Source.py).

While ingesting the parquet row data, we also pre process the light curve. For the processing, we do four things:

1. Remove saturations.
2. Keep the last non-detection before the trigger (if available).
3. Keep non-detection after the trigger and before the last detection (if available). 

We can plot these flux curves using the `LSST_Source.plot_flux_curve()` function. 

At this point, we have the `LSST_Source` object with all the data we need. However, all the light curves are full length. This is not realistic since real alerts will get objects with partial light curves. For this reason, we apply windowing to all the time series data in the original light curve. Thus, we create (potentially) multiple instance of the `LSST_Source` object with differing light curve lengths. We do this using the `LSST_Source.get_augmented_sources()` function which returns a list of these augmented sources. It is important to note that all the other features (non time series) are shared between these augmented objects. 

Finally, once we have the augmented sources, we want to get the feature tensors for all these objects. 

## Step 3 - Building the Tensors

## Step 4 - Classification Hierarchy

## Step 5 - Machine learning models

TODO: build out object to tensor conversion. Build out pipelines to do this conversion for all the parquet data. Build out classification hierarchy.
