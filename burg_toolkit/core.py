import logging
import os
import tempfile
import copy
from collections import UserDict

import numpy as np
import yaml
import pybullet as p
from PIL import Image, ImageDraw, ImageOps
import cv2
from fpdf import FPDF

from . import io, visualization
from . import mesh_processing
from . import render
from . import constants


class StablePoses:
    """
    Contains the stable poses of an ObjectType, i.e. poses and estimated probabilities of these poses.
    StablePoses object can be indexed and iterated. It holds the poses ordered from highest to lowest probability.

    :param probabilities: (n,) ndarray or list with probabilities as float values
    :param poses: (n, 4, 4) ndarray or nested list with corresponding poses
    """
    def __init__(self, probabilities, poses):
        self.probabilities = np.array(probabilities)
        self._p_norm = self.probabilities / self.probabilities.sum()
        self.poses = np.array(poses).reshape((-1, 4, 4))
        if len(self.probabilities) != len(self.poses):
            raise ValueError(f'probabilities and poses need to be same length. got {len(self.probabilities)} ' +
                             f'probabilities and {len(self.poses)} poses.')

        # sort in descending order, so that highest probability is first element
        sorted_indices = np.argsort(-self.probabilities)
        self.probabilities = self.probabilities[sorted_indices]
        self.poses = self.poses[sorted_indices]

    def sample_pose(self, uniformly=False):
        """
        Sample a pose from the set of poses, according to the probability of each individual pose.

        :param uniformly: If set to True, the pose probabilities will be ignored and we sample uniformly instead.

        :return: (4, 4) ndarray with one pose
        """
        rng = np.random.default_rng()
        if uniformly:
            index = rng.choice(len(self))
        else:
            index = rng.choice(len(self), p=self._p_norm)
        return self.poses[index]

    def __len__(self):
        assert len(self.probabilities) == len(self.poses), "probs and poses need to be same length"
        return len(self.probabilities)

    def __getitem__(self, item):
        if type(item) == int:
            return self.probabilities[item], self.poses[item]
        elif (type(item) == slice) or (type(item) == list) or (type(item) == np.ndarray):
            return StablePoses(self.probabilities[item], self.poses[item])
        else:
            raise TypeError('unknown index type calling StablePoses.__getitem__')

    def __iter__(self):
        assert len(self.probabilities) == len(self.poses), "probs and poses need to be same length"
        for i in range(len(self.probabilities)):
            yield self.probabilities[i], self.poses[i]

    def __str__(self):
        elems = [f'{len(self)} stable poses:']
        for prob, pose in self:
            elems.append(f'probability: {prob}, pose:\n{pose}')
        return '\n'.join(elems)


class ObjectType:
    """
    Describes an Object Type.
    Needs an identifier and a mesh, the latter can be provided either directly or as filename.
    Thumbnail, VHACD and URDF can be created from that.

    :param identifier: object identifier as string
    :param name: name of the object type as string
    :param mesh: open3d.geometry.TriangleMesh associated with the object (leave blank if filename provided)
    :param mesh_fn: filename where to find the mesh
    :param thumbnail_fn: filename of an image of the object
    :param vhacd_fn: filename of the vhacd mesh of the object
    :param urdf_fn: filename of the urdf file of the object
    :param mass: mass of object in kg (defaults to 0, which means fixed in space in simulations)
    :param friction_coeff: friction coefficient, defaults to 0.24
    :param stable_poses: either dataclass StablePoses or dict with probabilities and poses (or None)
    """
    def __init__(self, identifier, name=None, mesh=None, mesh_fn=None, thumbnail_fn=None, vhacd_fn=None, urdf_fn=None,
                 mass=None, friction_coeff=None, stable_poses=None):
        self.identifier = identifier
        self.name = name
        if mesh is not None and mesh_fn is not None:
            raise ValueError('Cannot create ObjectType if both mesh and mesh_fn are given. Choose one.')
        if mesh is None and mesh_fn is None:
            raise ValueError('Cannot create ObjectType with no mesh - must provide either mesh or mesh_fn.')
        self._mesh = mesh
        self.mesh_fn = mesh_fn
        self.thumbnail_fn = thumbnail_fn
        self.vhacd_fn = vhacd_fn
        self.urdf_fn = urdf_fn
        self.mass = mass or 0
        self.friction_coeff = friction_coeff or 0.24
        if isinstance(stable_poses, StablePoses):
            self.stable_poses = stable_poses
        elif isinstance(stable_poses, dict):
            self.stable_poses = StablePoses(probabilities=stable_poses['probabilities'], poses=stable_poses['poses'])
        elif stable_poses is None:
            self.stable_poses = None
        else:
            raise ValueError(f'unrecognised type of stable_poses: {type(stable_poses)}')

    @property
    def mesh(self):
        """Loads the mesh from file the first time it is used."""
        if self._mesh is None:
            self._mesh = io.load_mesh(mesh_fn=self.mesh_fn)
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

    # todo: method to save the current mesh to a new, given mesh_fn

    def generate_vhacd(self, vhacd_fn):
        """
        Generates an approximate convex decomposition of the object's mesh and stores it in given filename.

        :param vhacd_fn: Where to store the vhacd mesh. Will be set as property in this object type.
        """
        mesh_file = self.mesh_fn
        # mesh file could be None if object type has been given a mesh directly
        # will store in a temporary file then
        tmp_mesh_file, tmp_mesh_file_handle = None, None
        if mesh_file is None:
            tmp_mesh_file_handle, tmp_mesh_file = tempfile.mkstemp(suffix='.obj', text=True)
            io.save_mesh(tmp_mesh_file, self._mesh)
            mesh_file = tmp_mesh_file

        log_file_handle, log_file = tempfile.mkstemp(suffix='.log', text=True)

        # more documentation on this here: https://github.com/kmammou/v-hacd
        p.connect(p.DIRECT)
        p.vhacd(mesh_file, vhacd_fn, log_file)
        p.disconnect()

        self.vhacd_fn = vhacd_fn

        # dump the temporary files
        os.close(log_file_handle)
        os.remove(log_file)

        if tmp_mesh_file_handle is not None:
            os.close(tmp_mesh_file_handle)
        if tmp_mesh_file is not None:
            os.remove(tmp_mesh_file)

    def generate_urdf(self, urdf_fn, use_vhacd=True):
        """
        Method generates the urdf file for this object type, to be used in simulation.
        It needs to link to a mesh file. This can be either the original mesh or the VHACD. Usually we want this
        to be the VHACD. If it does not exist yet, it will be created in the same dir as the `urdf_fn`.

        :param urdf_fn: Path for the urdf to be generated (directory must exist).
        :param use_vhacd: Whether to use the vhacd (True, default) or the actual mesh (False).
        """
        logging.debug(f'generating urdf for {self.identifier}')
        name = self.identifier
        mass = self.mass
        origin = [0, 0, 0]  # mesh origin will be placed at origin when loading urdf in simulation
        inertia, com = mesh_processing.compute_mesh_inertia(self.mesh, mass)
        logging.debug(f'inertia: {inertia}')
        logging.debug(f'com: {com}')
        logging.debug(f'center: {self.mesh.get_center()}')

        if use_vhacd:
            if self.vhacd_fn is None:
                # we can just generate the vhacd here, assume same path as urdf
                vhacd_fn = os.path.join(os.path.dirname(urdf_fn), f'{name}_vhacd.obj')
                logging.debug(f'creating VHACD at {vhacd_fn}')
                self.generate_vhacd(vhacd_fn)
            rel_mesh_fn = os.path.relpath(self.vhacd_fn, os.path.dirname(urdf_fn))
        else:
            if self.mesh_fn is None:
                raise ValueError('ObjectType has no mesh_fn, but need mesh_fn linked in urdf (or use_vhacd)')
            rel_mesh_fn = os.path.relpath(self.mesh_fn, os.path.dirname(urdf_fn))

        io.save_urdf(urdf_fn, rel_mesh_fn, name, origin, inertia, com, mass)
        self.urdf_fn = urdf_fn

    def generate_thumbnail(self, thumbnail_fn):
        """
        Method generates a thumbnail picture in the specific file.

        :param thumbnail_fn: Path for the thumbnail to be generated
        """
        logging.debug(f'generating thumbnail for {self.identifier}')
        render.MeshRenderer().render_thumbnail(self.mesh, thumbnail_fn=thumbnail_fn)
        self.thumbnail_fn = thumbnail_fn

    def __str__(self):
        elems = [
            f'ObjectType: {self.identifier} ({self.name})',
            f'\tmass:\t\t{self.mass} kg',
            f'\tfriction:\t{self.friction_coeff}',
            f'\tmesh_fn:\t{self.mesh_fn}',
            f'\tvhacd_fn:\t{self.vhacd_fn}',
            f'\turdf_fn:\t{self.urdf_fn}',
            f'\tthumbnail_fn:\t{self.thumbnail_fn}',
            f'\tstable poses:\t{"none" if self.stable_poses is None else len(self.stable_poses)}'
        ]
        return '\n'.join(elems)


class ObjectInstance:
    """
    Describes an instance of an object type in the object library and a pose.

    :param object_type: an ObjectType referring to the type of this object instance
    :param pose: (4, 4) np array - homogenous transformation matrix
    """

    def __init__(self, object_type, pose=None):
        self.object_type = object_type
        if pose is None:
            self.pose = np.eye(4)
        else:
            self.pose = pose

    def __str__(self):
        return f'instance of {self.object_type.identifier} object type. pose:\n{self.pose}'

    def get_mesh(self):
        """
        Returns a copy of the mesh of the object type in the pose of the instance.

        :return: open3d.geometry.TriangleMesh
        """
        if self.object_type.mesh is None:
            raise ValueError('no mesh associated with this object type')
        mesh = copy.deepcopy(self.object_type.mesh)
        mesh.transform(self.pose)
        return mesh


class ObjectLibrary(UserDict):
    """
    Contains a library of ObjectType objects and adds some convenience methods to it.
    Acts like a regular python dict.

    :param name: string, name of the ObjectLibrary.
    :param description: string, description of the ObjectLibrary.
    """
    def __init__(self, name=None, description=None):
        super().__init__()
        self.name = name or 'default library'
        self.description = description or 'no description available'

    @classmethod
    def from_yaml(cls, yaml_fn):
        """
        Loads an ObjectLibrary described in the specified yaml file.

        :param yaml_fn: Filename of the YAML file.

        :return: ObjectLibrary containing all the objects.
        """
        with open(yaml_fn, 'r') as stream:
            data = yaml.safe_load(stream)

        logging.debug(f'reading object library from {yaml_fn}')
        logging.debug(f'keys: {[key for key in data.keys()]}')

        library = cls(data['name'], data['description'])

        # need to prepend the base directory of library to all paths, since yaml stores relative paths
        lib_dir = os.path.dirname(yaml_fn)
        for item in data['objects']:
            # item is a dictionary, we can pass it to the ObjectType constructor by unpacking it
            obj = ObjectType(**item)

            # complete the paths of all files
            obj.mesh_fn = cls._get_abs_path(obj.mesh_fn, lib_dir)
            obj.vhacd_fn = cls._get_abs_path(obj.vhacd_fn, lib_dir)
            obj.urdf_fn = cls._get_abs_path(obj.urdf_fn, lib_dir)
            obj.thumbnail_fn = cls._get_abs_path(obj.thumbnail_fn, lib_dir)

            library[obj.identifier] = obj

        return library

    def to_yaml(self, yaml_fn):
        """
        Saves an ObjectLibrary to the specified yaml file.
        All object properties will be saved as well (although no direct changes to the mesh are saved).
        Paths will be made relative to the base directory of the yaml file.

        :param yaml_fn: Filename where to store the object library.
        """
        # create the dictionary structure
        lib_dict = {
            'name': self.name,
            'description': self.description,
            'objects': []
             }

        lib_dir = os.path.dirname(yaml_fn)
        for _, item in self.data.items():
            stable_poses = None
            if item.stable_poses is not None:
                stable_poses = {
                    'probabilities': item.stable_poses.probabilities.tolist(),
                    'poses': item.stable_poses.poses.tolist()
                }
            obj_dict = {
                'identifier': item.identifier,
                'name': item.name,
                'thumbnail_fn': self._get_rel_path(item.thumbnail_fn, lib_dir),
                'mesh_fn': self._get_rel_path(item.mesh_fn, lib_dir),
                'vhacd_fn': self._get_rel_path(item.vhacd_fn, lib_dir),
                'urdf_fn': self._get_rel_path(item.urdf_fn, lib_dir),
                'mass': item.mass,
                'friction_coeff': item.friction_coeff,
                'stable_poses': stable_poses
            }
            lib_dict['objects'].append(obj_dict)

        with open(yaml_fn, 'w') as lib_file:
            yaml.dump(lib_dict, lib_file)

    @staticmethod
    def _get_abs_path(fn, base_dir):
        if fn is None:
            return None
        return os.path.join(base_dir, fn)

    @staticmethod
    def _get_rel_path(fn, base_dir):
        if fn is None:
            return None
        return os.path.relpath(fn, base_dir)

    def generate_vhacd_files(self, directory, override=False):
        """
        Calls the ObjectType's method to generate approximate convex decompositions for the object types in this lib.

        :param directory: where to put the vhacd files.
        :param override: If set to true, will create new vhacd files for all object types. If false, will create only
                         for those whose vhacd files are missing.
        """
        io.make_sure_directory_exists(directory)
        for name, obj in self.data.items():
            if override or obj.vhacd_fn is None:
                vhacd_fn = os.path.join(directory, f'{obj.identifier}_vhacd.obj')
                obj.generate_vhacd(vhacd_fn=vhacd_fn)

    def generate_urdf_files(self, directory, use_vhacd=True, override=False):
        """
        Calls the ObjectType's method to generate a urdf file for all object types in this library.
        If VHACD is used, but no VHACD available, VHACD will be created and stored in the same directory.
        Override parameter does not propagate through to VHACD creation - if VHACD exists it will be used.
        If you want to override VHACD files, generate them directly.

        :param directory: where to put the urdf files.
        :param use_vhacd: whether to link to vhacd meshes (True, default) or original meshes (False).
        :param override: If set to true, will create new urdf files for all object types. If false, will create only
                         for those whose urdf files are missing.
        """
        io.make_sure_directory_exists(directory)
        for name, obj in self.data.items():
            if override or obj.urdf_fn is None:
                urdf_fn = os.path.join(directory, f'{obj.identifier}.urdf')
                obj.generate_urdf(urdf_fn=urdf_fn, use_vhacd=use_vhacd)

    def generate_thumbnails(self, directory, override=False):
        """
        Calls the ObjectType's method to generate thumbnail for the object types in this library.

        :param directory: where to put the thumbnails.
        :param override: If set to true, will create new vhacd files for all object types. If false, will create only
                         for those whose vhacd files are missing.
        """
        io.make_sure_directory_exists(directory)
        for name, obj in self.data.items():
            if override or obj.thumbnail_fn is None:
                thumbnail_fn = os.path.join(directory, f'{obj.identifier}.png')
                obj.generate_thumbnail(thumbnail_fn)

    def compute_stable_poses(self, verify_in_sim=True, override=False):
        """
        Computes stable poses for all contained ObjectTypes.
        Requires the object's `mesh` (or `mesh_fn`). If verifying in simulation, requires the `urdf_fn` as well.

        :param verify_in_sim: Whether or not to verify the computed stable poses in simulation.
        :param override: If set to true, will override existing stable poses. If false, will keep stable poses for
                         object types that have some.
        """
        for name, obj in self.data.items():
            if override or obj.stable_poses is None:
                mesh_processing.compute_stable_poses(obj, verify_in_sim=verify_in_sim)

    def __len__(self):
        return len(self.data.keys())

    def __str__(self):
        return f'ObjectLibrary: {self.name}, {self.description}\nObjects:\n{[key for key in self.data.keys()]}'

    def print_details(self):
        print(f'ObjectLibrary:\n\t{self.name}\n\t{self.description}\nObjects:')
        for idx, (identifier, object_type) in enumerate(self.data.items()):
            print(f'{idx}: {object_type}')


class Scene:
    """
    contains all information about a scene
    """
    def __init__(self, ground_area=constants.SIZE_A3, objects=None, bg_objects=None):
        self.ground_area = ground_area
        self.objects = objects or []
        self.bg_objects = bg_objects or []

    def get_mesh_list(self, with_bg_objects=True, with_plane=True):
        """
        provides the scene objects as meshes

        :param with_bg_objects: Whether or not to include the background objects.
        :param with_plane: If True, will also create a mesh to visualise the ground area.

        :return: list of o3d.geometry.TriangleMesh of the object instances in this scene
        """
        instances = self.objects
        if with_bg_objects:
            instances.extend(self.bg_objects)

        meshes = []
        for instance in instances:
            meshes.append(instance.get_mesh())

        if with_plane:
            meshes.append(visualization.create_plane(size=self.ground_area, centered=False))

        return meshes

    def out_of_bounds_instances(self, margin=None):
        """
        Gives a list of object instance indices in this scene that exceed the bounds of ground_area-margin.

        :param margin: If provided, will subtract this margin from ground_area.

        :return: List of indices of the object instances that are out of bounds. If none are, return an empty list.
        """
        meshes = self.get_mesh_list(with_bg_objects=False, with_plane=False)
        x_min, y_min = 0, 0
        x_max, y_max = self.ground_area
        if margin is not None:
            x_min += margin
            y_min += margin
            x_max -= margin
            y_max -= margin

        out_of_bounds = []
        for i, mesh in enumerate(meshes):
            x1, y1, _ = mesh.get_min_bound()
            x2, y2, _ = mesh.get_max_bound()
            if x1 <= x_min or y1 <= y_min or x2 >= x_max or y2 >= y_max:
                out_of_bounds.append(i)

        return out_of_bounds

    def colliding_instances(self, with_bg_objects=True):
        """
        Gives a list of object instance indices in this scene that are in collision.
        Note that collisions between bg_objects are not detected, but option `with_bg_objects` can be set to
        detect collisions WITH the bg_objects.

        :param with_bg_objects: If True, will also check for collisions with background objects.

        :return: List of indices of the object instances that are in collision. If none are, return an empty list.
        """
        meshes = self.get_mesh_list(with_bg_objects=with_bg_objects, with_plane=False)
        collisions = mesh_processing.collisions(meshes)  # pairs of indices (potentially including bg_objects)

        max_idx = len(self.objects) - 1  # to exclude bg_object indices
        colliding_object_indices = []
        for i1, i2 in collisions:
            if i1 <= max_idx and i1 not in colliding_object_indices:
                colliding_object_indices.append(i1)
            if i2 <= max_idx and i2 not in colliding_object_indices:
                colliding_object_indices.append(i2)

        return colliding_object_indices

    def create_projection_image(self, px_per_mm=2, transparent=True, max_z=0.01, color_upper=(100, 100, 100),
                                color_lower=(0, 0, 0)):
        """
        Creates a projection image of the current scene.
        The objects will be projected onto the xy plane, whereas different colors are used for the triangles that are
        fully below `max_z` (`color_lower`), and all others (`color_upper`).
        First the upper triangles are drawn, then the lower ones.

        :param px_per_mm: resolution in pixels per mm
        :param transparent: If True, returned image will have 4 channels (rgba), else 3 (rgb)
        :param max_z: float, The boundary deciding whether triangles are in "upper" or "lower"
        :param color_upper: 3-tuple of ints in [0, 255]
        :param color_lower: 3-tuple of ints in [0, 255]

        :return: ndarray of shape (w, h, c), where c is 3 or 4 depending on `transparent`, and (w, h) are determined
                 based on the `ground_area` of the scene and the `px_per_mm` value.
        """
        if self.out_of_bounds_instances():
            raise ValueError('some instances are out of bounds, cannot create a projection on bounded canvas.')

        # parse colors and adjust depending on transparency
        bg_color = (255, 255, 255)
        img_mode = 'RGB'
        if transparent:
            color_upper = (*color_upper, 255)
            color_lower = (*color_lower, 255)
            bg_color = (*bg_color, 0)
            img_mode = 'RGBA'

        # create empty canvas
        px_per_m = px_per_mm * 1000.0
        img_size = [int(px_per_m * dim) for dim in self.ground_area]
        im = Image.new(img_mode, img_size, color=bg_color)

        # create projection for each mesh
        draw = ImageDraw.Draw(im)
        meshes = self.get_mesh_list(with_bg_objects=False, with_plane=False)
        for mesh in meshes:
            mesh = mesh_processing.as_trimesh(mesh)

            # find which triangles are close to ground (fully below max_z)
            low_mask = (mesh.triangles[:, :, 2] < max_z).all(axis=1)
            up_mask = (1 - low_mask).astype(bool)

            # first draw all the upper triangles
            for t in mesh.triangles[up_mask][:, :, :2]:  # (n, 3, 3), we only want projection, i.e. (n, 3, 2)
                img_points = np.rint(t * px_per_m)  # round to int
                # we actually need to convert it to list, otherwise pillow cannot draw it (awful)
                draw.polygon(img_points.flatten().tolist(), fill=color_upper, outline=color_upper)

            # now draw the lower triangles
            for t in mesh.triangles[low_mask][:, :, :2]:  # (n, 3, 3), we only want projection, i.e. (n, 3, 2)
                img_points = np.rint(t * px_per_m)  # round to int
                draw.polygon(img_points.flatten().tolist(), fill=color_lower, outline=color_lower)

        # flip the image, as y-axis is pointing into other direction
        im = ImageOps.flip(im)
        return np.array(im)

    def create_projection_heatmap(self, px_per_mm=2, transparent=True):
        """
        Creates a projection image of the current scene.
        The objects will be projected onto the xy plane, whereas the average of the triangles z-value is used to
        determine its color. The closer to the ground, the darker the color gets.

        :param px_per_mm: resolution in pixels per mm
        :param transparent: If True, returned image will have 4 channels (rgba), else 3 (rgb)

        :return: ndarray of shape (w, h, c), where c is 3 or 4 depending on `transparent`, and (w, h) are determined
                 based on the `ground_area` of the scene and the `px_per_mm` value.
        """
        if self.out_of_bounds_instances():
            raise ValueError('some instances are out of bounds, cannot create a projection on bounded canvas.')

        # parse colors and adjust depending on transparency
        bg_color = (255, 255, 255)
        img_mode = 'RGB'
        if transparent:
            bg_color = (*bg_color, 0)
            img_mode = 'RGBA'

        # create empty canvas
        px_per_m = px_per_mm * 1000.0
        img_size = [int(px_per_m * dim) for dim in self.ground_area]
        im = Image.new(img_mode, img_size, color=bg_color)

        z_clip = 0.10  # z values are capped here, to make sure we stay within 255

        # create projection for each mesh
        draw = ImageDraw.Draw(im)
        meshes = self.get_mesh_list(with_bg_objects=False, with_plane=False)
        for mesh in meshes:
            mesh = mesh_processing.as_trimesh(mesh)

            # compute average z-value for triangles based on vertices
            # sort triangles by z-value, draw from top to bottom
            z_values = np.average(mesh.triangles[:, :, 2], axis=-1)
            z_values = np.minimum(z_values, z_clip)
            order = np.argsort(-z_values)

            # first draw all the upper triangles
            for triangle, z_val in zip(mesh.triangles[order][:, :, :2], z_values[order]):
                img_points = np.rint(triangle * px_per_m)  # round to int
                c = int(800 * z_val**(1/2))
                if transparent:
                    color = (c, c, c, 255)
                else:
                    color = (c, c, c)

                draw.polygon(img_points.flatten().tolist(), fill=color, outline=color)

        # flip the image, as y-axis is pointing into other direction
        im = ImageOps.flip(im)
        return np.array(im)


class Printout:
    """
    Creates Printouts similar to GRASPA:
    # see https://github.com/robotology/GRASPA-benchmark/blob/master/src/layout-printer/layout_printer.py
    """
    def __init__(self, size=constants.SIZE_A2, px_per_mm=5, aruco_dict='DICT_4X4_250', marker_size_mm=57,
                 marker_spacing_mm=19):
        self._size = size
        self._px_per_mm = px_per_mm
        self._img_size = (int(px_per_mm * 1000 * size[1]), int(px_per_mm * 1000 * size[0]))  # rows first

        # generate PIL image with RGBA ready to overlay scenes, and required marker info
        self._pil_image, self.marker_info = self._generate_marker_image(aruco_dict, marker_size_mm, marker_spacing_mm)

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

        :param scene: burg.core.Scene to be projected onto the printout. Must be of same (or smaller) size.
        """
        self._check_size(scene.ground_area)

        scene_img = scene.create_projection_heatmap(px_per_mm=self._px_per_mm, transparent=True)
        assert self._img_size[0] >= scene_img.shape[0] and self._img_size[1] >= scene_img.shape[1], \
            f'scene image size too large. printout: {self._img_size}, scene: {scene_img.shape}'

        # alpha channel is basically a mask we use to not draw over the markers unless there is an object
        scene_img = Image.fromarray(scene_img, mode='RGBA')
        rgba_image = self._pil_image.convert(mode='RGBA')
        self._pil_image = Image.alpha_composite(rgba_image, scene_img).convert(mode='P')

    def _generate_marker_image(self, aruco_dict, marker_size_mm, marker_spacing_mm):
        """
        Uses cv2.aruco markers from `aruco_dict` with the given sizes and puts them on a canvas.
        Will put as many markers in the center of the canvas as possible regarding the dimensions.
        For being able to perform marker detection, the crucial marker information will be returned alongside the
        image.

        Note that this function is based on the implementation of GRASPA templates by Fabrizio Bottarel et al.
        see https://github.com/robotology/GRASPA-benchmark and specifically
        https://github.com/robotology/GRASPA-benchmark/blob/master/src/layout-printer/layout_printer.py

        :param aruco_dict: string, name of cv2.aruco dict
        :param marker_size_mm: marker size in mm
        :param marker_spacing_mm: marker spacing in mm

        :return: (img, marker_info) - PIL image with aruco markers placed in the centre, dictionary with required info
                 for pose estimation of the created aruco board
        """
        if aruco_dict not in constants.ARUCO_DICT.keys():
            raise ValueError(f'{aruco_dict} is not an aruco dictionary. Choose from: {constants.ARUCO_DICT.keys()}')

        marker_size = marker_size_mm * self._px_per_mm
        marker_spacing = marker_spacing_mm * self._px_per_mm

        marker_count_x = self._img_size[1] // (marker_size + marker_spacing)
        marker_count_y = self._img_size[0] // (marker_size + marker_spacing)

        # get the aruco board definition
        aruco_board = cv2.aruco.GridBoard_create(
            marker_count_x, marker_count_y, marker_size, marker_spacing,
            cv2.aruco.getPredefinedDictionary(constants.ARUCO_DICT[aruco_dict]))

        # draw the aruco board to specific size
        size_aruco_x = marker_count_x * marker_size + (marker_count_x - 1) * marker_spacing
        size_aruco_y = marker_count_y * marker_size + (marker_count_y - 1) * marker_spacing
        aruco_img = aruco_board.draw((size_aruco_x, size_aruco_y), 0)

        # create full image according to printout size and paste aruco image in center
        border_x = np.int32((self._img_size[1] - size_aruco_x) / 2)
        border_y = np.int32((self._img_size[0] - size_aruco_y) / 2)

        image = np.full(self._img_size, fill_value=255, dtype=np.uint8)
        image[border_y:border_y + size_aruco_y, border_x:border_x + size_aruco_x] = aruco_img
        pil_image = Image.fromarray(image, mode='P')  # P: 8bit grayscale

        # determine the marker origin with respect to world frame (of the scene), convert to [m]
        origin_x = (border_x + size_aruco_x) / self._px_per_mm / 1000
        origin_y = (border_y + size_aruco_y) / self._px_per_mm / 1000
        marker_frame = np.eye(4)
        marker_frame[0, 3] = origin_x
        marker_frame[1, 3] = origin_y

        # save all infos required to recreate marker board for detection
        aruco_info = {
            'dictionary': aruco_dict,
            'marker_count_x': marker_count_x,
            'marker_count_y': marker_count_y,
            'marker_size_mm': marker_size_mm,
            '   spacing_mm': marker_spacing_mm,
            'marker_frame': marker_frame
        }

        return pil_image, aruco_info

    def generate(self):
        return np.array(self._pil_image)

    def save_image(self, filename):
        self._pil_image.save(filename)  # mode is inferred from filename

    def save_pdf(self, filename, split_to_size=None):
        width_mm, height_mm = (self._size[0]*1000, self._size[1]*1000)
        if split_to_size is None:
            target_width_mm, target_height_mm = width_mm, height_mm
        else:
            target_width_mm, target_height_mm = split_to_size[0]*1000, split_to_size[1]*1000

        # determine orientation for splitting pages, choose the one with least number of pages
        n_pages_landscape = np.ceil(width_mm / target_width_mm) * np.ceil(height_mm / target_height_mm)
        n_pages_portrait = np.ceil(width_mm / target_height_mm) * np.ceil(height_mm / target_width_mm)

        if n_pages_portrait < n_pages_landscape:
            pdf = FPDF(orientation='P', unit='mm', format=(target_height_mm, target_width_mm))
            target_width_mm, target_height_mm = target_height_mm, target_width_mm
        else:
            pdf = FPDF(orientation='L', unit='mm', format=(target_height_mm, target_width_mm))
        pdf.set_title('BURG Printout')

        for page_x in range(int(np.ceil(width_mm / target_width_mm))):
            for page_y in range(int(np.ceil(height_mm / target_height_mm))):
                pdf.add_page()
                print(f'current pos in page: {pdf.get_x(), pdf.get_y()}')
                # crop img according to page
                left = page_x * target_width_mm * self._px_per_mm
                upper = page_y * target_height_mm * self._px_per_mm
                right = min((page_x+1) * target_width_mm, width_mm) * self._px_per_mm
                lower = min((page_y+1) * target_height_mm, height_mm) * self._px_per_mm
                img = self._pil_image.crop(box=(left, upper, right, lower))

                # save the image in temporary file, so we can put it into the pdf
                img_file_handle, img_file = tempfile.mkstemp(suffix='.png')
                img.save(img_file, format='PNG')
                actual_width_mm = (right - left) / self._px_per_mm  # will usually be target_width_mm, except for last
                pdf.image(img_file, x=0, y=0, w=actual_width_mm, type='PNG')

                # clear the temporary files
                os.close(img_file_handle), os.remove(img_file)

        pdf.output(filename, 'F')
