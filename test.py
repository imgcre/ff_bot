from enum import Enum, auto
import inspect
import re

class State(Enum):
    FroceUse = '使用'
    Unknown = auto()

print(State('使用'))