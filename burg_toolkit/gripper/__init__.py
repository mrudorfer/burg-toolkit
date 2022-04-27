from .base import GripperBase, TwoFingerGripperVisualisation

# 2 fingers
from .wsg_50 import WSG50
from .wsg_32 import WSG32
from .robotiq_2f_85 import Robotiq2F85
from .robotiq_2f_140 import Robotiq2F140
from .ezgripper import EZGripper
from .sawyer import Sawyer
from .franka import Franka
from .rg2 import RG2
from .barrett_hand_2f import BarrettHand2F

# 3 fingers
from .robotiq_3f import Robotiq3F
from .barrett_hand import BarrettHand
from .kinova_3f import Kinova3F


two_finger_grippers = [
    Franka,
    Robotiq2F85,
    Robotiq2F140,
    WSG32,
    WSG50,
    EZGripper,
    Sawyer,
    # RG2,  # skip as it does not work properly
    # BarrettHand2F,  # skip for now as it jumps around
]

# these are not implemented yet
three_finger_grippers = [
    Kinova3F,
    Robotiq3F,
    # BarrettHand  # skip for now as it jumps around
]

all_grippers = [
    *two_finger_grippers,
    *three_finger_grippers
]
