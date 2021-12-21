"""
===================
Scene Visualizer
===================

This script helps to set up scenes by detecting BURG printouts and overlaying objects from a given scene file.

Before the overlay can be used, you will have to calibrate your camera to determine intrinsic parameters and
distortion coefficients by using:
> python scene_visualizer.py --calibrate --calibration_dir /foo/calib_data/

After that, you can use the scene visualizer like this:
> python scene_visualizer.py --calibration_dir /foo/calib_data/ --scene /bar/scene_file.yaml

By default, video device 0 will be used.
"""

import argparse
import os

import burg_toolkit as burg
import cv2
import cv2.aruco
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list_ports', action='store_true', default=False,
                        help='checks all ports if camera devices are available')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='option to activate calibration routine')
    parser.add_argument('--calibration_dir', type=str, default='tmp/calib/',
                        help='path to calibration data folder (defaults to tmp/calib/)')
    parser.add_argument('--port', type=int, default=0, help='port of camera device (default is 0)')
    parser.add_argument('--scene', type=str, help='path to scene file that shall be visualised')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='for additional output')
    return parser.parse_args()


def detect_scene(port, calib_path, scene_filename, verbose):
    # marker recognition based on this tutorial:
    # https://docs.opencv.org/3.4/db/da9/tutorial_aruco_board_detection.html

    scene, object_library, printout = burg.Scene.from_yaml(scene_filename)
    if verbose:
        print(object_library)
        print(scene)
        print(printout)
        marker_frame = burg.visualization.create_frame(pose=printout.get_marker_frame())
        burg.visualization.show_geometries([scene, marker_frame])

    if printout is None:
        raise ValueError('Could not load info on printout from scene file.')

    # get aruco info
    aruco_dict = printout.marker_info.get_dictionary()
    aruco_board = printout.marker_info.get_board()
    parameters = cv2.aruco.DetectorParameters_create()  # default params

    # get camera info
    camera_matrix = np.load(os.path.join(calib_path, 'camera_matrix.npy'))
    distortion_coeffs = np.load(os.path.join(calib_path, 'distortion_coefficients.npy'))
    render_engine = None

    # enter video loop
    cap = cv2.VideoCapture(port)
    if not cap.isOpened():
        print(f'cannot open camera at port {port}')
        return

    print('video stream opened')
    print('press q to quit')
    while True:
        # capture a frame
        ok, frame = cap.read()
        if not ok:
            print('something wrong with the stream... sorry mate')
            break

        # setup render engine if not already done
        if render_engine is None:
            w, h = frame.shape[1], frame.shape[0]
            cam = burg.render.Camera.from_camera_matrix(w, h, camera_matrix)

            render_engine = burg.render.PyRenderEngine()
            render_engine.setup_scene(scene, cam, ambient_light=[0.3, 0.3, 0.3], with_plane=False)

        # detect markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_image_points = \
            cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=camera_matrix,
                                    distCoeff=distortion_coeffs)

        # only if any markers have been found
        if len(corners) > 0:
            # try to refine the detection - since we have grid board, we know where to expect markers
            corners, ids, rejected_image_points, recovered_indices = \
                cv2.aruco.refineDetectedMarkers(gray, aruco_board, corners, ids, rejected_image_points,
                                                camera_matrix, distortion_coeffs)

            # estimate pose of the board using all detected markers
            num_markers, rvec, tvec = \
                cv2.aruco.estimatePoseBoard(corners, ids, aruco_board, camera_matrix, distortion_coeffs,
                                            rvec=None, tvec=None)

            # drawing markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            frame = cv2.aruco.drawAxis(frame, camera_matrix, distortion_coeffs, rvec, tvec, 20)

            # render image of the scene, crop and fill in
            color, depth = render_engine.render(camera_pose=compute_camera_pose(rvec, tvec, printout))
            mask = depth != 0
            frame[mask] = color[mask]

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


def compute_camera_pose(rvec, tvec, printout):
    # rvec tvec are pose of markers wrt camera
    pose = np.eye(4)
    pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    pose[0:3, 3] = tvec.flatten() / 1000  # marker size was given in mm

    # compute camera wrt to markers and account for offset of markers in scene
    camera_frame = printout.get_marker_frame() @ np.linalg.inv(pose)

    # flip axes to have OpenGL camera
    camera_frame[0:3, 2] = -camera_frame[0:3, 2]  # z away from scene
    camera_frame[0:3, 1] = -camera_frame[0:3, 1]  # y up

    return camera_frame


def list_ports(total=5):
    """
    This method goes through ports 0...4 and checks which ones are working.
    Results will be printed on stdout.
    Method based on: https://stackoverflow.com/a/62639343/1264582
    """
    print(f'checking ports 0 to {total}...')
    summary = ['\nsummary:']
    for dev_port in range(total):
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            summary.append(f'port {dev_port} is not working.')
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                summary.append(f'port {dev_port} is working and reads images ({h}, {w})')
            else:
                summary.append(f'port {dev_port} is present but not available to read images ({h}, {w})')
            camera.release()

    print('\n'.join(summary))


def video_capture(port, save_path):
    """
    capture and display video of camera at given port, allow saving frames to save_path
    """
    cap = cv2.VideoCapture(port)
    if not cap.isOpened():
        print(f'cannot open camera at port {port}')
        return

    print('video stream opened')
    print('press q to quit')
    print('press s to save current image to folder:', save_path)
    saved_img_count = 0
    while True:
        # capture a frame
        ok, frame = cap.read()
        if not ok:
            print('something wrong with the stream... sorry mate')
            break

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # save image
            fn = os.path.join(save_path, f'image{saved_img_count:03d}.png')
            cv2.imwrite(fn, frame)
            print(f'saved image to {fn}')
            saved_img_count += 1

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


def camera_calibration(images_path, calib_path):
    # use this pattern: https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
    # has 9 rows and 6 columns
    # using code from opencv tutorial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    chessboard_size = (9, 6)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    image_files = os.listdir(images_path)
    print(f'found {len(image_files)} images in folder, start processing...')
    image_files = [os.path.join(images_path, image_fn) for image_fn in image_files]
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ok, corners = cv2.findChessboardCorners(gray, chessboard_size)
        # if we found sth, add object points, image points (after refining them)
        if ok:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ok)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f'did not find anything in {fname}')
    cv2.destroyAllWindows()

    ret_val, camera_matrix, distortion_coefficients, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f'computed calibration data from {len(objpoints)} images.')

    print('camera matrix:\n', camera_matrix)
    print('distortion coefficients (k1, k2, p1, p2, k3):\n', distortion_coefficients)

    burg.io.make_sure_directory_exists(calib_path)
    cm_fn = os.path.join(calib_path, 'camera_matrix.npy')
    dc_fn = os.path.join(calib_path, 'distortion_coefficients.npy')

    np.save(cm_fn, camera_matrix)
    np.save(dc_fn, distortion_coefficients)

    print(f'camera matrix saved to {cm_fn}')
    print(f'distortion coefficients saved to {dc_fn}')

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, distortion_coefficients)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print(f'total reprojection error is {mean_error}')


if __name__ == "__main__":
    args = parse_args()
    if args.list_ports:
        list_ports()
        exit(0)

    if args.calibrate:
        calibration_images_path = os.path.join(args.calibration_dir, 'images')
        burg.io.make_sure_directory_exists(calibration_images_path)
        print('please take 10+ pictures of the 9x6 chessboard pattern from various angles:')
        video_capture(args.port, calibration_images_path)
        print('thanks, performing calibration now')
        camera_calibration(calibration_images_path, args.calibration_dir)
        exit(0)

    if args.scene is None:
        print('No scene file provided. Please use --help to learn about usage.')
        exit(0)
    detect_scene(args.port, args.calibration_dir, args.scene, args.verbose)
