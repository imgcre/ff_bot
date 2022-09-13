from enum import Enum, auto
import inspect
import re

class State(Enum):
    FroceUse = '使用'
    Unknown = auto()

print([v.value for v in State if type(v.value) is str])