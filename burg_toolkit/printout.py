import os
import tempfile

import numpy as np
import cv2
import cv2.aruco
from fpdf import FPDF

from . import constants, mesh_processing, util


class MarkerInfo:
    """
    Class that holds info about aruco markers and a specific grid board pattern.

    :param aruco_dict: string, specifies an aruco dictionary. See burg.constants.ARUCO_DICT for keys.
    :param marker_count_x: int, number of markers in x direction
    :param marker_count_y: int, number of markers in y direction
    :param marker_size_mm: int, side length of a marker in mm
    :param marker_spacing_mm: int, width of the gap between markers in mm
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
        """
        Creates a MarkerInfo object by providing an area (in meter) instead of marker counts.
        Will try to fit as many markers into the area as possible.
        Note that the resulting size of the aruco board can be smaller than the given area.

        :param area: tuple, (x, y) in meters
        :param aruco_dict: string, specifies an aruco dictionary. See burg.constants.ARUCO_DICT for keys.
        :param marker_size_mm: int, side length of a marker in mm
        :param marker_spacing_mm: int, width of the gap between markers in mm

        :return: MarkerInfo object
        """
        # area: (x, y) in meters
        marker_count_x = area[0]*1000 // (marker_size_mm + marker_spacing_mm)
        marker_count_y = area[1]*1000 // (marker_size_mm + marker_spacing_mm)
        return cls(aruco_dict, marker_count_x, marker_count_y, marker_size_mm, marker_spacing_mm)

    def to_dict(self):
        """
        Creates a dictionary representation of this object which holds all relevant information.

        :return: dict
        """
        return vars(self)

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a MarkerInfo object based on a dictionary.

        :param dictionary: dict, needs to provide all required fields for the default constructor. Must not have
                           additional fields.

        :return: MarkerInfo object
        """
        return MarkerInfo(**dictionary)

    def get_dictionary(self):
        """
        Provides the cv2 aruco dictionary represented by the self.aruco_dict key

        :return: cv2.aruco dictionary
        """
        return cv2.aruco.getPredefinedDictionary(constants.ARUCO_DICT[self.aruco_dict])

    def get_board(self):
        """
        Provides the cv2 aruco board.

        :return: cv2.aruco.GridBoard
        """
        aruco_board = cv2.aruco.GridBoard_create(
            self.marker_count_x, self.marker_count_y, self.marker_size_mm, self.marker_spacing_mm,
            self.get_dictionary())
        return aruco_board

    def get_board_size_mm(self):
        """
        Computes the board size in millimeter.

        :return: tuple, (size_x_mm, size_y_mm)
        """
        size_x_mm = self.marker_count_x * self.marker_size_mm + (self.marker_count_x - 1) * self.marker_spacing_mm
        size_y_mm = self.marker_count_y * self.marker_size_mm + (self.marker_count_y - 1) * self.marker_spacing_mm
        return size_x_mm, size_y_mm

    def get_board_size(self):
        """
        Computes the board size in meter.

        :return: tuple, (size_x_m, size_y_m)
        """
        return tuple(size/1000 for size in self.get_board_size_mm())

    def get_board_image(self, px_per_mm):
        """
        Produces an image of the aruco grid board.

        :param px_per_mm: int, resolution

        :return: image as ndarray (note that rows=y and cols=x)
        """
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
        """
        Gives the pose of the marker board with respect to the scene's reference frame.

        :return: (4, 4) ndarray with marker board pose in scene reference frame.
        """
        # according to opencv docs https://docs.opencv.org/3.4/db/da9/tutorial_aruco_board_detection.html
        # the coordinate system of the aruco board is in the bottom left corner of the board, x pointing to the right,
        # y up, and z out of the board plane
        # the bottom left corner of the board is the one closest to our world frame, the orientation is the same
        # transform is hence just adding the border used to paste the aruco image into the full image
        board_size = self.marker_info.get_board_size()
        offset = [(full - board) / 2 for full, board in zip(self._size, board_size)]
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
        Adds a projection image of the given scene on top of the given image.
        The objects will be projected onto the xy plane, where the average of the triangle's z-value is used to
        determine its color. The closer to the ground, the darker the color gets.

        :param image: numpy image to draw on
        :param px_per_mm: resolution in pixels per mm

        :return: numpy image with 1 channel, uint8, (will also be altered in-place)
        """
        if scene.out_of_bounds_instances():
            raise ValueError('some instances are out of bounds, cannot create a projection on bounded canvas.')

        assert image.shape[0] >= scene.ground_area[1] * px_per_mm * 1000, f'i: {image.shape}, s: {scene.ground_area}'
        assert image.shape[1] >= scene.ground_area[0] * px_per_mm * 1000, f'i: {image.shape}, s: {scene.ground_area}'

        # naive approach would be to create a projection for each mesh
        # however, output then depends on the order of objects, as some objects that are drawn later could hide
        # objects that have been drawn earlier but are actually below the later objects
        # therefore we first gather all the triangles of all meshes, sort them by z, and draw them in that order
        meshes = scene.get_mesh_list(with_bg_objects=False, with_plane=False)
        if not meshes:
            return image

        triangles = []
        for mesh in meshes:
            mesh = mesh_processing.as_trimesh(mesh)
            triangles.append(mesh.triangles)

        triangles = np.concatenate(triangles)
        # compute average z-value for triangles based on vertices, for coloring and order of drawing
        z_values = np.average(triangles[:, :, 2], axis=-1)
        order = np.argsort(-z_values)

        # get pixel coordinates from world coordinates
        triangle_points = np.rint(triangles[:, :, :2] * px_per_mm * 1000).astype(np.int32)  # round to int
        # flip y axis, because image coordinate system starts at the top left
        triangle_points[:, :, 1] = image.shape[0] - triangle_points[:, :, 1]

        for triangle, z_val in zip(triangle_points[order], z_values[order]):
            c = min(int(800 * max(z_val, 0) ** (1 / 2)), 200)  # fancy look-up table for color, clip at 200 intensity
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
        saving as '.png' file.

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


class PrintoutDetector:
    def __init__(self, printout):
        # get aruco info
        self.aruco_dict = printout.marker_info.get_dictionary()
        self.aruco_board = printout.marker_info.get_board()
        self.marker_frame = printout.get_marker_frame()
        self.parameters = cv2.aruco.DetectorParameters_create()  # default params

        # remember predictions from the latest frame
        self.tvec = None
        self.rvec = None

    def detect(self, image, camera_matrix, distortion_coefficients):
        """
        attempts to detect markers in the given image.

        :param image: ndarray as opencv image, either 2-dim (gray image) or 3-dim (bgr image).
        :param camera_matrix: ndarray (3, 3), the camera matrix K
        :param distortion_coefficients: list or ndarray (5), the distortion coefficients D

        :return: ndarray as opencv image or None, ndarray with drawn in markers, None if no markers detected
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            raise ValueError(f'image has unexpected shape: {image.shape} (expected 2 or 3 dims)')

        corners, ids, rejected_image_points = \
            cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters, cameraMatrix=camera_matrix,
                                    distCoeff=distortion_coefficients)

        # only if any markers have been found
        if len(corners) > 0:
            # try to refine the detection - since we have grid board, we know where to expect markers
            corners, ids, rejected_image_points, recovered_indices = \
                cv2.aruco.refineDetectedMarkers(gray, self.aruco_board, corners, ids, rejected_image_points,
                                                camera_matrix, distortion_coefficients)

            # estimate pose of the board using all detected markers
            num_markers, self.rvec, self.tvec = \
                cv2.aruco.estimatePoseBoard(corners, ids, self.aruco_board, camera_matrix,
                                            distortion_coefficients, rvec=self.rvec, tvec=self.tvec,
                                            useExtrinsicGuess=self.rvec is not None)

            # drawing markers
            frame = cv2.aruco.drawDetectedMarkers(image, corners, ids)
            frame = cv2.aruco.drawAxis(frame, camera_matrix, distortion_coefficients, self.rvec, self.tvec, 20)
            return frame

        # if no markers detected
        self.rvec = None
        self.tvec = None
        return None

    def get_camera_pose_cv(self):
        """
        z-axis points towards the scene, y-axis points down.
        """
        if self.rvec is None or self.tvec is None:
            raise ValueError('call to detect must be successful in order to get camera pose. current pose unknown.')

        # rvec tvec are pose of markers wrt camera
        pose = np.eye(4)
        pose[0:3, 0:3] = cv2.Rodrigues(self.rvec)[0]
        pose[0:3, 3] = self.tvec.flatten() / 1000  # marker size was given in mm

        # compute camera wrt to markers and account for offset of markers in scene
        # camera_frame = self.marker_frame @ np.linalg.inv(pose)
        camera_frame = self.marker_frame @ np.linalg.inv(pose)
        return camera_frame

    def get_camera_pose_opengl(self):
        """
        z-axis pointing away from the scene, y-axis points up.
        """
        camera_frame = self.get_camera_pose_cv()

        # flip axes to have OpenGL camera
        camera_frame[0:3, 2] = -camera_frame[0:3, 2]  # z away from scene
        camera_frame[0:3, 1] = -camera_frame[0:3, 1]  # y up

        return camera_frame
