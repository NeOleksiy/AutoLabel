from .face_keypoint import FaceKeypointTask
from .animal_pose import AnimalPoseKeypointTask
from .human_pose import HumanPoseKeypointTask
from .segmentation import SegmentationTask

__all__ = [
    "FaceKeypointTask",
    "AnimalPoseKeypointTask",
    "HumanPoseKeypointTask",
    "SegmentationTask",
]
