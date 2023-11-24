"""
Tutorial to introduce grasps and demonstrate the usage of grasp sampler.
"""
import numpy as np
import open3d as o3d
import burg_toolkit as burg


def show_grasp_pose_definition():
    print('show_grasp_pose_definition().')
    print('A grasp is defined as a 6d pose (position and orientation). There are different conventions, but the one'
          ' we are using in this toolkit is as follows:\n'
          '\tThe position of the grasp is in the center between the finger tips of the gripper.\n'
          '\tx-axis (red): points towards either finger tip; it is the axis along which the fingers close.\n'
          '\tz-axis (blue): points towards the gripper base; it is the axis along which the gripper approaches.\n'
          '\ty-axis (green): follows from the previous two.\n'
          'Note that the gripper is symmetric, i.e., the x-axis could point in either direction and the grasp would'
          ' be the same.')
    print('Close the open3d window to proceed.')
    gs = burg.grasp.GraspSet.from_translations(np.asarray([0, 0, 0]).reshape(-1, 3))
    gripper = burg.gripper.TwoFingerGripperVisualisation()
    burg.visualization.show_grasp_set([burg.visualization.create_frame(size=0.02)],
                                      gs, gripper=gripper)


def make_sphere(pos=None, radius=0.01):
    """ utility function to make an o3d trianglemesh sphere """
    if pos is None:
        pos = [0, 0, 0]
    sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_vis.translate(pos)
    sphere_vis.compute_vertex_normals()
    return sphere_vis


def make_arrow(pos=None, direction=None, point_to_pos=False):
    """ utility function to make an o3d trianglemesh arrow. """
    cylinder_height = 5.0 / 100
    cone_height = 4.0 / 100
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=1. / 100,
        cone_radius=1.5 / 100,
        cylinder_height=cylinder_height,
        cone_height=cone_height)
    arrow.compute_vertex_normals()

    if point_to_pos:
        translation = [0, 0, -(cylinder_height+cone_height)]
        arrow.translate(translation)

    arrow.rotate(burg.util.rotation_to_align_vectors([0, 0, 1], direction), center=[0, 0, 0])
    arrow.translate(pos)
    return arrow


def create_connecting_line(p1, p2, length_margin=0.2):
    """ utility function to draw line connecting between two points. """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction) + length_margin
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=length)
    cylinder.rotate(burg.util.rotation_to_align_vectors([0, 0, 1], direction), center=[0, 0, 0])
    cylinder.translate((p1+p2)/2)
    cylinder.compute_vertex_normals()
    return cylinder


def explain_antipodal_grasp():
    print('explain_antipodal_grasp().')
    print('A grasp with a two-finger gripper is basically described by the contact points with the object and the '
          'surface normals at those points. The surface normals are perpendicular to the surface. ')
    print('Ideally, we would like the finger tips to exert force in the direction of those surface normals, as this'
          ' will hold even if the surface is relatively slippery (i.e. has a small friction coefficient). '
          'The more friction the surface provides, the more we can allow to deviate from the surface normal.')
    print('Consider the following two contact points on a box-shaped object.')
    print('(Close the visualizer window to continue.)')

    # create a box as test object
    box = o3d.geometry.TriangleMesh.create_box(0.2, 0.3, 0.1)
    box.compute_triangle_normals()

    # first contact point
    c1 = np.array([0, 0.15, 0.05])
    n1 = np.array([-1, 0, 0])

    # second contact point
    c2 = np.array([0.2, 0.18, 0.03])
    n2 = np.array([1, 0, 0])

    c1_vis = make_sphere(c1)
    c2_vis = make_sphere(c2)

    burg.visualization.show_geometries([
        box, c1_vis, c2_vis
    ])

    print('Now we display the normal forces at the contact points (as arrows) as well as the connecting line between '
          'the contact points (which would be aligned with the x-axis of our grasp). We can see that there is an '
          'angle between them, i.e., the grasp is not optimal and relies on some friction to achieve force closure. '
          'The friction coefficient determines how much of an angle there can be without the grasp slipping.')
    print('(Exit visualizer to continue.)')

    # visualise normal forces
    n1_vis = make_arrow(c1, -n1, point_to_pos=True)
    n2_vis = make_arrow(c2, -n2, point_to_pos=True)

    # visualise connecting vector
    line = create_connecting_line(c1, c2)
    burg.visualization.show_geometries([
        c1_vis, c2_vis, n1_vis, n2_vis, line
    ])


def sample_grasps(scene, target_obj):
    print('sample_grasps().')
    print('We will now use the antipodal grasp sampler from the toolkit to find grasps that satisfy the antipodal '
          'constraint. This means that the angles of the normal forces are within the friction cones of each '
          'contact point. The grasp sampler randomly selects a first contact point on the object surface and shoots '
          'rays within the friction cone towards the inside of the object. Wherever a ray exits the object, we '
          'have a potential second contact point. If the contact points satisfy the antipodal constraint, the grasp '
          'should be stable. We now sample a number of different approach angles to determine the gripper\'s '
          'orientation.')
    print('In the following we see an example contact point pair with a number of approach directions. Note that '
          'some of those or even all of those might be actually in collision. We will deal with that in the next '
          'step.')
    opening_width = 0.08
    gripper = burg.gripper.TwoFingerGripperVisualisation(opening_width=opening_width)

    # here we actually sample grasps!
    ags = burg.sampling.AntipodalGraspSampler(n_orientations=7)
    graspset, contacts = ags.sample(target_obj, n=7, max_gripper_width=opening_width)
    print('(Exit visualizer to continue.)')
    burg.visualization.show_grasp_set([scene], graspset, gripper=gripper)

    print('In order to find out which grasps are colliding, we perform a simple collision check between the '
          'simplified gripper model and the objects in the scene. The result is stored in the scores of the graspset. '
          'We can visualize the graspset and provide a function that maps the score to a colour - in our case we '
          'signify a colliding grasp with red and a non-colliding grasp with green.')
    print('(Exit visualizer to continue.)')
    graspset.scores = ags.check_collisions(graspset, scene, gripper_mesh=gripper.mesh)  # need python-fcl
    burg.visualization.show_grasp_set([scene], graspset, gripper=gripper,
                                      score_color_func=lambda s: [s, 1-s, 0])

    graspset = graspset[graspset.scores == 0]
    print(f'In this little example, we found {len(graspset)} grasps that were successful. Although not always perfect, '
          f'the antipodal grasp sampler is (currently) in fact one of the best sampling schemes to quickly find a '
          f'large number of high-quality grasps.')
    print('Let us try and sample a larger number of grasps, perform the collision checks, and see what we get!')
    print('Note that this may take a moment.')
    print('(Close the visualizer to continue).')
    graspset, contacts = ags.sample(target_obj, n=200, max_gripper_width=opening_width)
    graspset.scores = ags.check_collisions(graspset, scene, gripper_mesh=gripper.mesh)  # need python-fcl
    burg.visualization.show_grasp_set([scene], graspset, gripper=gripper,
                                      score_color_func=lambda s: [s, 1-s, 0])

    print('We can now filter the grasps, i.e. remove the ones that are colliding, and we are left with a set of '
          'grasps that are highly likely to work in the simulation.')
    graspset = graspset[graspset.scores == 0]
    print(f'Number of grasps: {len(graspset)}')

    burg.visualization.show_grasp_set([scene], graspset, gripper=gripper,
                                      score_color_func=lambda s: [s, 1 - s, 0])

    return graspset


def simulate_grasps(scene, graspset, target_obj):
    print('simulate_grasps().')
    print('Now we can test the grasp in simulation. Here we apply an actual gripper model the refine the collision '
          'detection, and we execute the grasp (if not in collision). Note that we only simulate the gripper and '
          'not the whole robot to simplify things.')
    print('Note: This uses pyBullet. Instead of closing the pyBullet window, please just stay on the console and follow'
          'the instructions here. When you close the window, the program will just terminate.')
    print('Press Enter to check out the simulation.')
    input()

    sim = burg.sim.GraspSimulator(scene, verbose=True)
    gripper_type = burg.gripper.Franka
    idx = np.random.choice(len(graspset))
    score = sim.execute_grasp(gripper_type, graspset[idx], target_obj)
    sim.dismiss()
    print('As you could see, the result of the grasp was the following:')
    print(burg.sim.GraspScores.score2description(score))
    print('For simulating a larger number of grasps, we use pyBullet\'s direct mode. It will not visualize each run, '
          'but just do the computations. Let us simulate all the previously found grasps and see what scores we get.')
    print('Enter to continue.')
    input()

    scores = burg.sim.GraspSimulator.simulate_graspset(graspset, scene, target_obj, gripper_type)
    print('We have now simulated all the grasps. Hit enter to show the results.')
    input()
    burg.sim.GraspScores.print_summary(scores)
    print('With these results, we could now further reduce our graspset to those that worked in simulation. These are'
          ' the grasps which are most likely to work in the real world too!')
    print('That is the end of this tutorial, thanks and good bye.')


if __name__ == '__main__':
    show_grasp_pose_definition()
    explain_antipodal_grasp()

    scene_fn = '../examples/scenes/scene01.yaml'
    target_obj_idx = 1
    print('For the next steps, we need to load a scenario from file.')
    print('Attempting to load scene from: ', scene_fn)
    scene, lib, _ = burg.Scene.from_yaml(scene_fn)
    print('loaded.')
    target_object = scene.objects[target_obj_idx]
    print('Targeting object', target_object.object_type.identifier)

    gs = sample_grasps(scene, target_object)
    simulate_grasps(scene, gs, target_object)
