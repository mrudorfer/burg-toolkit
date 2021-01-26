import pymeshlab as ml
import numpy as np


def convert_mesh_to_point_cloud(mesh_paths, with_normals=False, percentage=1.0, scale_factor=1.0):
    """
    for each mesh file in mesh_paths a point cloud will be sampled using poisson disk sampling
    :param mesh_paths: list of paths to mesh files
    :param with_normals: if true, normals are included, i.e. returns Nx6 instead of Nx3
    :param percentage: specifies radius for sampling (the larger the fewer points we will get), defaults to 1.0
    :param scale_factor: each vertex is multiplied with this scale factor, defaults to 1.0
    :return: list of point clouds, length is len(mesh_paths) and each PC is either Nx3 or Nx6
    """

    single = False
    if not type(mesh_paths) is list:
        mesh_paths = [mesh_paths]
        single = True

    results = []

    for mesh_path in mesh_paths:
        ms = ml.MeshSet()

        ms.load_new_mesh(mesh_path)
        ms.apply_filter('matrix_set_from_translation_rotation_scale',
                        scalex=scale_factor, scaley=scale_factor, scalez=scale_factor)
        ms.apply_filter('poisson_disk_sampling', radius=ml.Percentage(percentage))
        mesh = ms.current_mesh()
        points = mesh.vertex_matrix()

        if with_normals:
            normals = mesh.vertex_normal_matrix()
            points_and_normals = np.concatenate((points, normals), axis=1)
            results.append(points_and_normals)
        else:
            results.append(points)

    if single:
        return results[0]
    else:
        return results
