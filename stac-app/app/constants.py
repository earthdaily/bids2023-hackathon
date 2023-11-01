from enum import Enum
from typing import List, Optional, Union


class EOProductLevel(str, Enum):
    L1C = "l1c"
    L2A = "l2a"
    L3A = "l3a"
    RTC = "rtc"


class STACBand(str, Enum):
    BLUE = "blue"
    GREEN = "green"
    RED = ("red",)
    REDEDGE1 = "rededge1"
    REDEDGE2 = "rededge2"
    REDEDGE3 = "rededge3"
    NIR = "nir"
    NIR08 = "nir08"
    NIR09 = "nir09"
    CIRRUS = "cirrus"
    SCL = "scl"
    SWIR16 = "swir16"
    SWIR22 = "swir22"
    VV = "vv"
    VH = "vh"

    def __str__(self):
        return self.value


class EOProductSource(str, Enum):
    SENTINEL2 = "sentinel-2"
    SENTINEL1 = "sentinel-1"
    VENUS = "venus"


class EOProductType(str, Enum):

    DTYPE: Optional[str]
    RESOLUTION: Optional[int]
    FILL_VALUE: Optional[Union[int, float]]
    DEFAULT_ASSETS: Optional[List[str]]
    RGB: Optional[List[str]]

    def __new__(
        cls,
        type: str,
        dtype: Optional[str] = None,
        resolution: Optional[int] = None,
        fill_value: Optional[Union[int, float]] = None,
        default_assets: Optional[List[str]] = [],
        rgb_names: Optional[List[str]] = [],
    ):
        obj = str.__new__(cls, type)  # type: ignore
        obj._value_ = type
        obj.DTYPE = dtype
        obj.RESOLUTION = resolution
        obj.FILL_VALUE = fill_value
        obj.DEFAULT_ASSETS = default_assets
        obj.RGB = rgb_names

        return obj

    SENTINEL2_L2A = (
        f"{EOProductSource.SENTINEL2}-{EOProductLevel.L2A}",
        "uint16",
        10,
        0,
        [
            STACBand.BLUE,
            STACBand.GREEN,
            STACBand.RED,
            STACBand.REDEDGE1,
            STACBand.REDEDGE2,
            STACBand.REDEDGE3,
            STACBand.NIR,
            STACBand.NIR08,
            STACBand.NIR09,
            STACBand.CIRRUS,
            STACBand.SCL,
            STACBand.SWIR16,
            STACBand.SWIR22,
        ],
        [
            STACBand.BLUE,
            STACBand.GREEN,
            STACBand.RED,
        ],
    )
    VENUS_L2A = (
        f"{EOProductSource.VENUS}-{EOProductLevel.L2A}",
        "float32",
        5,
        0,
        [
            "image_file_SRE_B3",
            "image_file_SRE_B4",
            "image_file_SRE_B7",
            "image_file_SRE_B8",
            "image_file_SRE_B9",
            "image_file_SRE_B10",
        ],
        [
            "image_file_SRE_B3",
            "image_file_SRE_B4",
            "image_file_SRE_B7",
        ],
    )

    def __str__(self):
        return self.value

    @property
    def source(self):
        return EOProductSource(str(self.value).rsplit("-", 1)[0])

    @property
    def level(self):
        return EOProductLevel(str(self.value).rsplit("-", 1)[1])


class SCLMaskLabel:
    # Note: This is for Sentinel2 L2A only
    NO_DATA = 0
    SATURED_OR_DEFECTED = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIIFED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11
