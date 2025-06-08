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