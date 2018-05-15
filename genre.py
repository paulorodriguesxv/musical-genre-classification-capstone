from enum import auto, unique, Enum

@unique
class Genre(Enum):
    BLUES = auto()
    CLASSICAL = auto()
    JAZZ = auto()
    POP = auto()
    ROCK = auto()