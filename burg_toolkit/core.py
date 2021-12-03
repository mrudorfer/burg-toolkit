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
        self.name = name or identifier  # just duplicate identifier if no name given
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

    def has_all_attributes(self):
        """
        Check whether this object has all attributes, or something is missing.
        Checks only attributes that can be created automatically, i.e. vhacd, urdf, thumbnail, stable poses.

        :return: bool
        """
        sth_missing = \
            self.thumbnail_fn is None or \
            self.vhacd_fn is None or \
            self.urdf_fn is None or \
            self.stable_poses is None
        return not sth_missing

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
        self.filename = None

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
        library.filename = yaml_fn

        # need to prepend the base directory of library to all paths, since yaml stores relative paths
        lib_dir = os.path.dirname(yaml_fn)
        for item in data['objects']:
            # item is a dictionary, we can pass it to the ObjectType constructor by unpacking it
            obj = ObjectType(**item)

            # complete the paths of all files
            obj.mesh_fn = io.get_abs_path(obj.mesh_fn, lib_dir)
            obj.vhacd_fn = io.get_abs_path(obj.vhacd_fn, lib_dir)
            obj.urdf_fn = io.get_abs_path(obj.urdf_fn, lib_dir)
            obj.thumbnail_fn = io.get_abs_path(obj.thumbnail_fn, lib_dir)

            library[obj.identifier] = obj

        return library

    def to_yaml(self, yaml_fn=None):
        """
        Saves an ObjectLibrary to the specified yaml file.
        All object properties will be saved as well (although no direct changes to the mesh are saved).
        Paths will be made relative to the base directory of the yaml file.

        :param yaml_fn: Filename where to store the object library. This will override the object library's filename
                        property. If yaml_fn is None, will use the filename property of ObjectLibrary.
        """
        if yaml_fn is not None:
            self.filename = yaml_fn
        if self.filename is None:
            raise ValueError('No filename given. Cannot store ObjectLibrary to yaml.')

        # create the dictionary structure
        lib_dict = {
            'name': self.name,
            'description': self.description,
            'objects': []
        }

        lib_dir = os.path.dirname(self.filename)
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
                'thumbnail_fn': io.get_rel_path(item.thumbnail_fn, lib_dir),
                'mesh_fn': io.get_rel_path(item.mesh_fn, lib_dir),
                'vhacd_fn': io.get_rel_path(item.vhacd_fn, lib_dir),
                'urdf_fn': io.get_rel_path(item.urdf_fn, lib_dir),
                'mass': item.mass,
                'friction_coeff': item.friction_coeff,
                'stable_poses': stable_poses
            }
            lib_dict['objects'].append(obj_dict)

        with open(self.filename, 'w') as lib_file:
            yaml.dump(lib_dict, lib_file)

    def _prepare_directory(self, directory, default):
        """
        Will choose either `directory`, if given, or otherwise construct a directory based on where the library file
        is located and the given `default` directory. Makes sure the directory exists, if it is a new one.
        """
        if directory is None:
            if self.filename is None:
                raise ValueError('no directory specified, also library has no filename from which it can be inferred')
            directory = os.path.join(os.path.dirname(self.filename), default)
        io.make_sure_directory_exists(directory)
        return directory

    def generate_vhacd_files(self, directory=None, override=False):
        """
        Calls the ObjectType's method to generate approximate convex decompositions for the object types in this lib.

        :param directory: where to put the vhacd files. If None, will put in library_dir/vhacd
        :param override: If set to true, will create new vhacd files for all object types. If false, will create only
                         for those whose vhacd files are missing.
        """
        directory = self._prepare_directory(directory, default='vhacd')
        for name, obj in self.data.items():
            if override or obj.vhacd_fn is None:
                vhacd_fn = os.path.join(directory, f'{obj.identifier}_vhacd.obj')
                obj.generate_vhacd(vhacd_fn=vhacd_fn)

    def generate_urdf_files(self, directory=None, use_vhacd=True, override=False):
        """
        Calls the ObjectType's method to generate a urdf file for all object types in this library.
        If VHACD is used, but no VHACD available, VHACD will be created and stored in the same directory.
        Override parameter does not propagate through to VHACD creation - if VHACD exists it will be used.
        If you want to override VHACD files, generate them directly.

        :param directory: where to put the urdf files. If None, will put in library_dir/urdf
        :param use_vhacd: whether to link to vhacd meshes (True, default) or original meshes (False).
        :param override: If set to true, will create new urdf files for all object types. If false, will create only
                         for those whose urdf files are missing.
        """
        directory = self._prepare_directory(directory, default='urdf')
        for name, obj in self.data.items():
            if override or obj.urdf_fn is None:
                urdf_fn = os.path.join(directory, f'{obj.identifier}.urdf')
                obj.generate_urdf(urdf_fn=urdf_fn, use_vhacd=use_vhacd)

    def generate_thumbnails(self, directory=None, override=False):
        """
        Calls the ObjectType's method to generate thumbnail for the object types in this library.

        :param directory: where to put the thumbnails. If None, will put in library_dir/thumbnails
        :param override: If set to true, will create new vhacd files for all object types. If false, will create only
                         for those whose vhacd files are missing.
        """
        directory = self._prepare_directory(directory, default='thumbnails')
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

    def objects_have_all_attributes(self):
        """
        :return: bool, True if all contained ObjectTypes have all attributes.
        """
        has_all_attributes = True
        for name, obj in self.data.items():
            has_all_attributes &= obj.has_all_attributes()
        return has_all_attributes

    def compute_all_attributes(self, override=False):
        """
        Computes all missing attributes of the contained object types, such as vhacd, urdf, thumbnails and stable poses.
        Note: this may take some time.

        :param override: Even if attributes are present, will override those, i.e. computes everything anew.
        """
        self.generate_vhacd_files(override=override)
        self.generate_urdf_files(override=override)
        self.compute_stable_poses(override=override)
        self.generate_thumbnails(override=override)


class Scene:
    """
    A class to hold information about a scene, specified by some ground area, a list of object instances and a list
    of background object instances (which are usually considered fixed in space / immovable, i.e. obstacles or
    support surfaces like table, shelf, etc.).

    :param ground_area: tuple (x, y), dimension of the scene in [m], you can use predefined sizes in burg.constants.
                        Scene origin is in (0, 0) and ground area extends to (x, y).
    :param objects: list of ObjectInstances
    :param bg_objects: list of ObjectInstances
    """

    def __init__(self, ground_area=constants.SIZE_A3, objects=None, bg_objects=None):
        self.ground_area = ground_area
        self.objects = objects or []
        self.bg_objects = bg_objects or []

    def __str__(self):
        return f'Scene:\n\tground area: {self.ground_area}' \
               f'\n\t{len(self.objects)} objects: {[instance.object_type.identifier for instance in self.objects]}' \
               f'\n\t{len(self.bg_objects)} bg objects: {[bg.object_type.identifier for bg in self.bg_objects]}'

    def to_yaml(self, yaml_fn, object_library=None, printout=None):
        """
        Saves this Scene to the specified yaml file. Basically saves the object identifiers and the pose. For laoding,
        an ObjectLibrary will be required. Providing an ObjectLibrary to this function will store the path and allow
        to load ObjectLibrary with this scene file.
        Printout info can be stored alongside the scene, which is relevant for marker detection and scene visualisation.
        Paths will be made relative to the base directory of the yaml file.

        :param yaml_fn: Filename where to store the scene.
        :param object_library: If provided, will store the path to the library in the scene file.
        :param printout: If provided, will store printout/marker info in the scene file.
        """
        yaml_dir = os.path.dirname(yaml_fn)
        lib_fn = None if object_library is None else io.get_rel_path(object_library.filename, yaml_dir)

        # create the dictionary structure
        scene_dict = {
            'object_library_fn': lib_fn,
            'ground_area_x': self.ground_area[0],
            'ground_area_y': self.ground_area[1],
            'objects': [],
            'bg_objects': [],
        }

        # add object and bg_object instances
        for instance_list, name in zip([self.objects, self.bg_objects], ['objects', 'bg_objects']):
            for instance in instance_list:
                if object_library is not None and instance.object_type.identifier not in object_library.keys():
                    logging.warning(f'Object type {instance.object_type.identifier} not found in ObjectLibrary. ' +
                                    f'May not be able to restore from saved scene file.')
                instance_dict = {
                    'object_type': instance.object_type.identifier,
                    'pose': instance.pose.tolist()
                }
                scene_dict[name].append(instance_dict)

        if printout is None:
            printout_dict = None
        else:
            printout_dict = {
                'pdf_fn': io.get_rel_path(printout.pdf_fn, yaml_dir),
                'image_fn': io.get_rel_path(printout.image_fn, yaml_dir),
                'marker_info': printout.marker_info
            }
            # convert from np array to list
            printout_dict['marker_info']['marker_frame'] = printout_dict['marker_info']['marker_frame'].tolist()
        scene_dict['printout'] = printout_dict

        with open(yaml_fn, 'w') as scene_file:
            yaml.dump(scene_dict, scene_file)

    @classmethod
    def from_yaml(cls, yaml_fn, object_library=None):
        """
        Loads a Scene described in the specified yaml file.
        If you have the ObjectLibrary loaded already, please provide it. Otherwise it will be loaded with the scene.

        :param yaml_fn: Filename of the YAML file.
        :param object_library: An ObjectLibrary to use. If None, will try to load the ObjectLibrary from the scene file.

        :return: tuple with (Scene, ObjectLibrary, printout_info dict): Scene will be the loaded scene; ObjectLibrary
                 is either the one provided, or if None provided then the one read from the scene file; printout_info
                 is a dictionary with the filenames of printouts and marker_info, or None if not available in the
                 scene file.
        """
        with open(yaml_fn, 'r') as stream:
            data = yaml.safe_load(stream)
        scene_dir = os.path.dirname(yaml_fn)

        logging.debug(f'reading scene from {yaml_fn}')
        logging.debug(f'keys: {[key for key in data.keys()]}')

        ground_area = (data['ground_area_x'], data['ground_area_y'])

        # make sure we have an object library to draw object types from
        if object_library is None:
            lib_fn = data['object_library_fn']
            if lib_fn is None:
                raise ValueError('Scene file does not refer to an ObjectLibrary, please provide an ObjectLibrary.')
            object_library = ObjectLibrary.from_yaml(io.get_abs_path(lib_fn, scene_dir))

        # gather instances and bg_instances
        objects = []
        bg_objects = []
        for source_list, dest_list in zip([data['objects'], data['bg_objects']], [objects, bg_objects]):
            for item in source_list:
                identifier = item['object_type']
                if identifier not in object_library.keys():
                    raise ValueError(f'ObjectType {identifier} not found in given ObjectLibrary. Unable to load scene.')
                object_type = object_library[identifier]
                pose = np.array(item['pose'])
                dest_list.append(ObjectInstance(object_type, pose))

        scene = cls(ground_area, objects, bg_objects)

        # finally add the printout info if available
        printout_info = data['printout']
        if printout_info is not None:
            printout_info['image_fn'] = io.get_abs_path(printout_info['image_fn'], scene_dir)
            printout_info['pdf_fn'] = io.get_abs_path(printout_info['pdf_fn'], scene_dir)
            printout_info['marker_info']['marker_frame'] = np.array(printout_info['marker_info']['marker_frame'])

        return scene, object_library, printout_info

    def get_mesh_list(self, with_bg_objects=True, with_plane=True):
        """
        provides the scene objects as meshes (i.e. transformed according to the pose in the scene)

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

    def create_projection_heatmap(self, px_per_mm=2, transparent=True):
        """
        Creates a projection image of the current scene.
        The objects will be projected onto the xy plane, where the average of the triangle's z-value is used to
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

        # create projection for each mesh
        draw = ImageDraw.Draw(im)
        meshes = self.get_mesh_list(with_bg_objects=False, with_plane=False)
        for mesh in meshes:
            mesh = mesh_processing.as_trimesh(mesh)

            # compute average z-value for triangles based on vertices
            # sort triangles by z-value, draw from top to bottom
            z_values = np.average(mesh.triangles[:, :, 2], axis=-1)
            order = np.argsort(-z_values)

            for triangle, z_val in zip(mesh.triangles[order][:, :, :2], z_values[order]):
                img_points = np.rint(triangle * px_per_m)  # round to int
                c = min(int(800 * z_val ** (1 / 2)), 200)  # fancy look-up table, clip at 200 intensity
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
    A Printout is basically an aruco marker board. By adding scenes the projections of the object instances will be
    visible on the printout, overlaid over the markers. Using the marker detection, you can infer the poses of the
    objects. All required info are stored in `marker_info`. You need to be careful to not overlay too many markers
    though - the more of them are visible the better the pose estimation will be.

    These Printouts are an extended and more flexible version of the GRASPA templates by  Bottarel et al.
    You can define arbitrary sizes, get png or export as pdf and split up to a more reasonable and printable page size.

    :param size: tuple, size of the printout. Should be at least as big as the ground planes of the scenes it will
                 contain. Use sizes in burg.constants. For custom sizes, units are in meter and the bigger value should
                 come first.
    :param px_per_mm: int, defines the resolution of the underlying image and affects visual quality of the output.
    :param aruco_dict: string, determines which aruco dictionary to use. See burg.constants.ARUCO_DICT for options.
                       Number of contained markers must be sufficient to fill the whole template size.
    :param marker_size_mm: int, desired size of markers in mm
    :param marker_spacing_mm: int, desired gap between markers in mm
    """

    def __init__(self, size=constants.SIZE_A2, px_per_mm=5, aruco_dict='DICT_4X4_250', marker_size_mm=57,
                 marker_spacing_mm=19):
        self._size = size
        self._px_per_mm = px_per_mm
        self._img_size = (int(px_per_mm * 1000 * size[1]), int(px_per_mm * 1000 * size[0]))  # rows first

        # generate PIL image with RGBA ready to overlay scenes, and required marker info
        self._pil_image, self.marker_info = self._generate_marker_image(aruco_dict, marker_size_mm, marker_spacing_mm)

        # memorize which files we saved to
        self.image_fn = None
        self.pdf_fn = None

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
        self._pil_image = Image.alpha_composite(rgba_image, scene_img)

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
            'spacing_mm': marker_spacing_mm,
            'marker_frame': marker_frame
        }

        return pil_image, aruco_info

    def get_image(self):
        """
        :return: ndarray with grayscale image of the Printout.
        """
        return np.array(self._pil_image.convert('P'))

    def save_image(self, filename):
        """
        Saves an image of the printout to given filename. The file type defines which mode is used. We recommend
        to save as '.png' file.

        :param filename: str, path to file.
        """
        self._pil_image.save(filename)  # mode is inferred from filename
        self.image_fn = filename

    def save_pdf(self, filename, page_size=None, margin_mm=6.35):
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

        for page_x in range(int(np.ceil(width_mm / target_width_mm))):
            for page_y in range(int(np.ceil(height_mm / target_height_mm))):
                pdf.add_page()
                # crop img according to page size and margins
                left = (page_x * target_width_mm + margin_mm) * self._px_per_mm
                upper = (page_y * target_height_mm + margin_mm) * self._px_per_mm
                right = min((page_x + 1) * target_width_mm - margin_mm, width_mm) * self._px_per_mm
                lower = min((page_y + 1) * target_height_mm - margin_mm, height_mm) * self._px_per_mm
                img = self._pil_image.crop(box=(left, upper, right, lower))

                # save the image in temporary file, so we can put it into the pdf
                img_file_handle, img_file = tempfile.mkstemp(suffix='.png')
                img.save(img_file, format='PNG')
                actual_width_mm = (right - left) / self._px_per_mm
                pdf.image(img_file, x=margin_mm, y=margin_mm, w=actual_width_mm, type='PNG')

                # clear the temporary files
                os.close(img_file_handle), os.remove(img_file)

        pdf.output(filename, 'F')
        self.pdf_fn = filename
