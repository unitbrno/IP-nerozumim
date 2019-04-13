"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import os
from typing import List, Dict


class PathProvider:
    def __init__(self):
        self.root = os.path.abspath('data/develop')

    def get_ground_truth_path(self, image_path: str) -> str:
        """
        For input image - return path to corresponding ground_truth image.
        :param image_path: path to *.tiff image
        """
        return image_path.replace(self.root, os.path.join(self.root, '_ground_truths'), 1).rstrip('.tiff') + '.png'

    def get_paths_dict(self) -> Dict[int, List[str]]:
        """
        :return: Dict with Category index as key and list of paths of category-images as value.
        """
        paths = {}
        i = 1
        for subdir in sorted(os.listdir(self.root)):
            if subdir.startswith('Category'):
                subdir_path = os.path.join(self.root, subdir)
                paths[i] = [os.path.join(subdir_path, img) for img in os.listdir(os.path.join(subdir_path))]
                i += 1

        return paths

    def get_paths_list(self) -> List[str]:
        """
        :return: list of paths to all input images in the dataset
        """
        paths = []
        for subdir in sorted(os.listdir(self.root)):
            if subdir.startswith('Category'):
                subdir_path = os.path.join(self.root, subdir)
                paths.extend(os.path.join(subdir_path, img) for img in os.listdir(os.path.join(subdir_path)))

        return paths
