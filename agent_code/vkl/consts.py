BOMB = "BOMB"
UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
WAIT = "WAIT"

ACTIONS = INT2STR = [
    BOMB,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    WAIT,
]
STR2INT = {}
for i, v in enumerate(INT2STR):
    STR2INT[v] = i

N_CHANNELS = 8
FIELD_SIZE = 17
