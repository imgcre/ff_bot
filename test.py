from enum import Enum, auto


class Hi():
    def __str__(self) -> str:
        return 'hi'

print(' '.join([Hi(), Hi()]))