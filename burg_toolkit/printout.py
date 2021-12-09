import os
import tempfile

import numpy as np
import cv2
import cv2.aruco
from fpdf import FPDF

from . import constants, mesh_processing, util


class MarkerInfo:
    """
    Class that holds info about aruco markers, allowing to make detections or recreate the marker setup.
    """
    def __init__(self, aruco_dict, marker_count_x, marker_count_y, marker_size_mm, marker_spacing_mm):
        if aruco_dict not in constants.ARUCO_DICT.keys():
            raise ValueError(f'{aruco_dict} is not an aruco dictionary. Choose from: {constants.ARUCO_DICT.keys()}')
        self.aruco_dict = aruco_dict
        self.marker_count_x = int(marker_count_x)
        self.marker_count_y = int(marker_count_y)
        self.marker_size_mm = int(marker_size_mm)
        self.marker_spacing_mm = int(marker_spacing_mm)

    @classmethod
    def from_area(cls, area, aruco_dict, marker_size_mm, marker_spacing_mm):
        # area: (x, y) in meters
        marker_count_x = area[0]*1000 // (marker_size_mm + marker_spacing_mm)
        marker_count_y = area[1]*1000 // (marker_size_mm + marker_spacing_mm)
        return cls(aruco_dict, marker_count_x, marker_count_y, marker_size_mm, marker_spacing_mm)

    def to_dict(self):
        return vars(self)

    @classmethod
    def from_dict(cls, dictionary):
        return MarkerInfo(**dictionary)

    def get_dictionary(self):
        return cv2.aruco.getPredefinedDictionary(constants.ARUCO_DICT[self.aruco_dict])

    def get_board(self):
        aruco_board = cv2.aruco.GridBoard_create(
            self.marker_count_x, self.marker_count_y, self.marker_size_mm, self.marker_spacing_mm,
            self.get_dictionary())
        return aruco_board

    def get_board_size_mm(self):
        size_x_mm = self.marker_count_x * self.marker_size_mm + (self.marker_count_x - 1) * self.marker_spacing_mm
        size_y_mm = self.marker_count_y * self.marker_size_mm + (self.marker_count_y - 1) * self.marker_spacing_mm
        return size_x_mm, size_y_mm

    def get_board_size(self):
        return tuple(size/1000 for size in self.get_board_size_mm())

    def get_board_image(self, px_per_mm):
        size_x_mm, size_y_mm = self.get_board_size_mm()
        size_x_px = px_per_mm * size_x_mm
        size_y_px = px_per_mm * size_y_mm

        aruco_board = self.get_board()
        image = aruco_board.draw((size_x_px, size_y_px))
        return image


class Printout:
    """
    A Printout is basically an aruco marker board to which we add scenes by projecting the object instances.
    Using marker detection on the (printed) printout, we can then infer the poses of the objects.
    You need to be careful to not overlay too many markers though - the more of them are visible the better the pose
    estimation will be.

    These Printouts are an extended and more flexible version of the GRASPA templates by  Bottarel et al.
    You can define arbitrary sizes, get png or export as pdf and split up to a more reasonable and printable page size.
    See https://github.com/robotology/GRASPA-benchmark

    :param size: tuple, size of the printout. Should be at least as big as the ground planes of the scenes it will
                 contain. Use sizes in burg.constants. For custom sizes, units are in meter and the bigger value should
                 come first.
    :param marker_info: optional MarkerInfo object, if None given will use following default marker specs:
                        aruco dict 'DICT_4X4_250', marker_size 57 mm, spacing 19 mm
    """
    def __init__(self, size=constants.SIZE_A2, marker_info=None):
        self._size = size
        if marker_info is None:
            marker_info = MarkerInfo.from_area(size, aruco_dict='DICT_4X4_250', marker_size_mm=57, marker_spacing_mm=19)
        self.marker_info = marker_info

        self.scenes = []

    def to_dict(self):
        """
        :return: a dictionary representation of this object (for saving/loading yaml files)
        """
        dictionary = {
            'size': list(self._size),
            'marker_info': self.marker_info.to_dict()
        }
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        """
        :param: a dictionary read from file (same format as self.to_dict required)

        :return: a Printout object corresponding to the dictionary
        """
        marker_info = MarkerInfo.from_dict(dictionary['marker_info'])
        size = tuple(dictionary['size'])
        return cls(size, marker_info)

    def __str__(self):
        return 'Printout:\n' + util.dict_to_str(self.to_dict())

    def get_marker_frame(self):
        # according to opencv docs https://docs.opencv.org/3.4/db/da9/tutorial_aruco_board_detection.html
        # the coordinate system of the aruco board is in the bottom left corner of the board, x going to the left,
        # y moving up and z pointing out of the board plane
        # the bottom left corner of the board is placed closest to our world frame, so the transform can be computed
        # by adding the offset/border which is used to paste the aruco image into the full image

        board_size = self.marker_info.get_board_size()
        offset = [(full - board) / 2 for full, board in zip(self._size, board_size)]
        print('offset', offset)
        marker_frame = np.eye(4)
        marker_frame[0:2, 3] += offset
        return marker_frame

    def _check_size(self, size):
        # make sure given size is smaller or equal to self._size
        if len(self._size) != len(size):
            raise ValueError('given size has different number of dimensions than own size')
        for i in range(len(size)):
            if self._size[i] < size[i]:
                raise ValueError('given size must not exceed the own size')

    def add_scene(self, scene):
        """
        Adds the scene to the printout by projecting all objects onto the canvas.
        Note that this is a shallow function which only stores the scene reference. The actual projecting will happen
        once you generate an image or pdf of this Printout. Note that every change you make to the scene after adding
        it will therefore affect the generated image/pdf.

        :param scene: burg.core.Scene to be projected onto the printout. Must be of same (or smaller) size.
        """
        self._check_size(scene.ground_area)
        self.scenes.append(scene)

    @staticmethod
    def _add_scene_projection(image, scene, px_per_mm):
        """
        Adds a projection image of the current scene on top of the given image.
        The objects will be projected onto the xy plane, where the average of the triangle's z-value is used to
        determine its color. The closer to the ground, the darker the color gets.

        :param image: numpy image to draw on
        :param px_per_mm: resolution in pixels per mm

        :return: numpy image with 1 channel, uint8, (will also be altered in-place)
        """
        if scene.out_of_bounds_instances():
            raise ValueError('some instances are out of bounds, cannot create a projection on bounded canvas.')

        # create projection for each mesh
        meshes = scene.get_mesh_list(with_bg_objects=False, with_plane=False)
        for mesh in meshes:
            mesh = mesh_processing.as_trimesh(mesh)

            # compute average z-value for triangles based on vertices, for coloring and order of drawing
            z_values = np.average(mesh.triangles[:, :, 2], axis=-1)
            order = np.argsort(-z_values)

            # get pixel coordinates from world coordinates
            triangle_points = np.rint(mesh.triangles[:, :, :2] * px_per_mm * 1000).astype(np.int32)  # round to int
            # flip y axis, because image coordinate system starts at the top left
            triangle_points[:, :, 1] = image.shape[0] - triangle_points[:, :, 1]

            for triangle, z_val in zip(triangle_points[order], z_values[order]):
                c = min(int(800 * z_val ** (1 / 2)), 200)  # fancy look-up table for color, clip at 200 intensity
                cv2.fillConvexPoly(image, triangle, c)

        return image

    def _generate_marker_image(self, px_per_mm):
        """
        Puts a base marker image in the center of an image the size of the template.

        :param px_per_mm: int, determines resolution of image

        :return: numpy uint8 image with aruco markers placed in the centre
        """
        # create canvas for full image (height = y, width = x)
        img_size = (int(px_per_mm * 1000 * self._size[1]), int(px_per_mm * 1000 * self._size[0]))  # rows first
        image = np.full(img_size, fill_value=255, dtype=np.uint8)

        # create marker image, size is smaller than the full image
        marker_img = self.marker_info.get_board_image(px_per_mm)
        print('marker img shape:', marker_img.shape)

        # paste marker image in center of full image
        border_y = (img_size[0] - marker_img.shape[0]) // 2
        border_x = (img_size[1] - marker_img.shape[1]) // 2
        image[border_y:border_y + marker_img.shape[0], border_x:border_x + marker_img.shape[1]] = marker_img

        return image

    def get_image(self, px_per_mm=5):
        """
        Generates an image of the printout.

        :param px_per_mm: int, desired resolution

        :return: ndarray with grayscale image of the Printout.
        """
        image = self._generate_marker_image(px_per_mm)

        for scene in self.scenes:
            image = self._add_scene_projection(image, scene, px_per_mm=px_per_mm)

        return image

    def save_image(self, filename, px_per_mm=5):
        """
        Saves an image of the printout to given filename. The file type defines which mode is used. We recommend
        to save as '.png' file.

        :param filename: str, path to file.
        :param px_per_mm: int, desired resolution
        """
        image = self.get_image(px_per_mm)
        cv2.imwrite(filename, image)

    def save_pdf(self, filename, page_size=None, margin_mm=6.35, px_per_mm=5):
        """
        Generates a pdf file and saves it to `filename`.

        Make sure you can print the pdf to scale, otherwise the marker transforms will not be correct and the object
        silhouettes will not fit. You can adjust the margin to avoid problems with the printer.
        Margin of 6.35mm should generally be fine. You can try 0 and see if it works for you - this way most of the
        printout will be visible.
        https://stackoverflow.com/questions/3503615/what-are-the-minimum-margins-most-printers-can-handle/19581039

        :param filename: str, where to save the pdf file.
        :param page_size: tuple, page size to use for the pdf file. The printout will be split up if necessary. Use
                          sizes in burg.constants. You can also use custom tuples, then units should be in meter and
                          the bigger value should come first. If None, will use the size of the Printout as page size.
        :param margin_mm: float, will keep a margin to all sides without changing the scale/position of the printout.
                          The parts of the printout within the margin will not be visible.
        :param px_per_mm: int, desired resolution of image
        """
        width_mm, height_mm = (self._size[0] * 1000, self._size[1] * 1000)
        if page_size is None:
            target_width_mm, target_height_mm = width_mm, height_mm
        else:
            target_width_mm, target_height_mm = page_size[0] * 1000, page_size[1] * 1000

        # determine orientation for splitting pages, choose the one with least number of pages
        n_pages_landscape = np.ceil(width_mm / target_width_mm) * np.ceil(height_mm / target_height_mm)
        n_pages_portrait = np.ceil(width_mm / target_height_mm) * np.ceil(height_mm / target_width_mm)

        if n_pages_portrait < n_pages_landscape:
            pdf = FPDF(orientation='P', unit='mm', format=(target_height_mm, target_width_mm))
            target_width_mm, target_height_mm = target_height_mm, target_width_mm
        else:
            pdf = FPDF(orientation='L', unit='mm', format=(target_height_mm, target_width_mm))
        pdf.set_title('BURG Printout')

        image = self.get_image(px_per_mm)

        for page_x in range(int(np.ceil(width_mm / target_width_mm))):
            for page_y in range(int(np.ceil(height_mm / target_height_mm))):
                pdf.add_page()
                # crop img according to page size and margins
                left = int((page_x * target_width_mm + margin_mm) * px_per_mm)
                upper = int((page_y * target_height_mm + margin_mm) * px_per_mm)
                right = int(min((page_x + 1) * target_width_mm - margin_mm, width_mm) * px_per_mm)
                lower = int(min((page_y + 1) * target_height_mm - margin_mm, height_mm) * px_per_mm)
                crop = image[upper:lower+1, left:right+1]

                # we need a temporary file, so we can put the cropped image into the pdf
                img_file_handle, img_file = tempfile.mkstemp(suffix='.png')
                cv2.imwrite(img_file, crop)
                actual_width_mm = (right - left) / px_per_mm
                pdf.image(img_file, x=margin_mm, y=margin_mm, w=actual_width_mm, type='PNG')
                os.close(img_file_handle), os.remove(img_file)  # clear temporary

        pdf.output(filename, 'F')
