import wildboar.datasets
from wildboar.datasets import clear_cache, refresh_repositories
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
    try:
        # Clear cache and refresh repositories
        clear_cache()
        refresh_repositories()
    except Exception as e:
        print(f"Error refreshing wildboar datasets: {e}")
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