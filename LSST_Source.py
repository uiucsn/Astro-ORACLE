class LSST_Source:

    # List of features actually stored in the instance of the class.
    features = []

    def __init__(self, parquet_row, class_label) -> None:
        pass

    def plot_lightcurve(self) -> None:
        pass

    def get_light_curve(self) -> dict:
        pass

    def get_host_galaxy_properties(self) -> dict:
        pass

    def get_z(self) -> dict:
        pass

    def get_classification_hierarchy(self):
        pass


if __name__=='__main__':

    print('Hello world')

