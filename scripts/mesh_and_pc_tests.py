import open3d as o3d

import burg_toolkit as burg

mesh_fn = '../data/samples/flathead-screwdriver/flatheadScrewdriverMediumResolution.ply'
# texture_fn = '../data/samples/flathead-screwdriver/flatheadScrewdriver.png'

mesh = burg.io.load_mesh(mesh_fn)
burg.mesh_processing.check_properties(mesh)

pc = burg.mesh_processing.poisson_disk_sampling(mesh)
o3d.visualization.draw([mesh, pc])
