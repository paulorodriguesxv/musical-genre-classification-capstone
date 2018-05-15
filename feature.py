from enum import auto, unique, Enum

@unique
class SoundFeature(Enum):
    FREQUENCE = auto()
    ZCR = auto()
    RMS = auto()
    SC = auto()
    MFCC = auto()