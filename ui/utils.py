import wildboar.datasets

def getDatasetNames():
    """
    Get the names of the datasets available in wildboar.datasets compatible with GRSF.
    """
    return [dt for dt in wildboar.datasets.list_datasets() if dt not in ["OliveOil", 
                                                                        "Phoeme", 
                                                                        "PigAirwayPressure",
                                                                        "PigArtPressure",
                                                                        "PigCVP",
                                                                        "Fungi",
                                                                        "FiftyWords"]]
def getDataset(dataset_name:str):
    return wildboar.datasets.load_dataset(dataset_name)


def getDistanceMetrics():
    """
    Get the list of distance metrics available in wildboar.datasets.
    """
    return ["euclidean",
                    "normalized_euclidean",
                    "adtw",
                    "dtw",
                    "ddtw",
                    "wdtw",
                    "wddtw",
                    "lcss",
                    "wlcss",
                    "erp",
                    "edr",
                    "msm",
                    "twe",
                    "manhattan",
                    "minkowski",
                    "chebyshev",
                    "cosine",
                    "angular"]