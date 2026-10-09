"""Supported structured feature transformers."""
from .Bag_Transformers import BasicBag
from .Image_Transformers import BasicImage
from .Series_Transformers import BasicSeries, BsplineSeries, FPCA, localRSeries, tsfreshSeries
from .Text_Transformers import BasicText

__all__ = [
    "BasicBag", "BasicImage", "BasicSeries", "BasicText",
    "BsplineSeries", "FPCA", "localRSeries", "tsfreshSeries", "activeTrans",
]


def __getattr__(name):
    # Avoid circular imports while StructureDataFrame imports SData.
    if name == "activeTrans":
        from .activeTrans import activeTrans
        return activeTrans
    raise AttributeError(name)
