from .base import GripperBase

# 2 fingers
from .wsg_50 import GripperWSG50
from .wsg_32 import GripperWSG32
from .robotiq_2f_85 import GripperRobotiq2F85
from .robotiq_2f_140 import GripperRobotiq2F140
from .ezgripper import GripperEZGripper
from .sawyer import GripperSawyer
from .franka import GripperFranka
from .rg2 import GripperRG2
from .barrett_hand_2f import GripperBarrettHand2F
# # 3 fingers
from .robotiq_3f import GripperRobotiq3F
from .barrett_hand import GripperBarrettHand
from .kinova_3f import GripperKinova3F


two_finger_grippers = {
    'franka': GripperFranka,
    'robotiq_2f_85': GripperRobotiq2F85,
    'robotiq_2f_140': GripperRobotiq2F140,
    'wsg_32': GripperWSG32,
    'wsg_50': GripperWSG50,
    'ezgripper': GripperEZGripper,
    'sawyer': GripperSawyer,
    'rg2': GripperRG2,
    'barrett_hand_2f': GripperBarrettHand2F,
}

three_finger_grippers = {
    'kinova_3f': GripperKinova3F,
    'robotiq_3f': GripperRobotiq3F,
    # 'barrett_hand':     GripperBarrettHand  # skip for now because it has different signature
}

all_grippers = {
    **two_finger_grippers,
    **three_finger_grippers
}
