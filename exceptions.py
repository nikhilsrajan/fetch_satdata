class FetchSatDataException(Exception):
    "Base class for all custom exceptions in this repository"


class MajorException(Exception):
    "Base class for all the exceptions that need to be urgent tended to."


class CatalogManagerException(FetchSatDataException):
    "For exceptions related to catalogmanager.py"


class DatacubeException(FetchSatDataException):
    "For exceptions related to datacube creation"


class MetadataException(FetchSatDataException):
    "For exceptions related to extract_metadata.py"