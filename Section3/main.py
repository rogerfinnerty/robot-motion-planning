"""
ME570 homework 3
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import geometry as geometry
import potential as pot
import robot as robot


def sphere_test_collision(save_plot=False):
    """
    Generates one figure with a sphere (with arbitrary parameters) and
    nb_points=100 random points that are colored according to the sign of
    their distance from the sphere (red for negative, green for positive).
    Generates a second figure in the same way (and the same set of points)
    but flipping the sign of the radius  r of the sphere. For each sampled
    point, plot also the result of the output  pointsSphere.
    """
    center = np.array([[0],[0]])
    radius = -2
    d_influence = 1
    sphere = geometry.Sphere(center, radius, d_influence)
    sphere.plot('k')
    n_test_pts = 100
    # Generate 100 random tests points in range [(-2,-2), (2,2)]
    test_points = 4 * np.random.rand(2, n_test_pts) + np.array([[-2],[-2]])
    # Compute distances between test points and the sphere
    dist = sphere.distance(test_points)
    for idx in range(n_test_pts):
        if dist[0,idx] <= 0:
            plt.scatter(float(test_points[0,idx]), float(test_points[1,idx]), color = 'r')
        else:
            plt.scatter(float(test_points[0,idx]), float(test_points[1,idx]), color = 'g')
    
    if save_plot:
        plt.savefig('assets/sphere_test_collision.png')

def planner_run_plot_test(rep_wgt=1, shape='quadratic', lr=1e-2, steps=1000, save_plot=False):
    """
    Show the results of Planner.run_plot for each goal location in
    world.xGoal, and for different interesting combinations of
    potential['repulsive_weight'],  potential['shape'],  epsilon, and
    nb_steps. In each case, for the object of class  Planner should have the
    attribute  function set to  Total.eval, and the attribute  control set
    to the negative of  Total.grad.

    :param rep_wgt: repulsive weight for the potential function
    :param shape: shape of the potential function
    :param lr: learning rate (epsilon) for the planner
    :param steps: number of steps for the planner
    :param save_plot: if True, saves the plots to files and an animation of the planner paths
    """
    world = pot.SphereWorld() # load problem data
    nb_goal = world.x_goal.shape[1]
    nb_start = world.x_start.shape[1]

    def negative_grad(x):
        return -total_u.grad(x)

    for goal_idx in range(nb_goal):
        _, (ax1, ax2) = plt.subplots(ncols=2)
        world.plot(ax1)
        goal_paths = []
        for start_idx in range(nb_start):
            potential = {
                'x_goal': world.x_goal[:,goal_idx],
                'repulsive_weight': rep_wgt,
                'shape': shape
            }
            total_u = pot.Total(world, potential)
            planner = pot.Planner(
                function = total_u.eval,
                control = negative_grad,
                epsilon = lr,
                nb_steps = steps
                )
            x_path, u_path = planner.run(world.x_start[:,start_idx], ax1)
            ax1.plot(x_path[0,:], x_path[1,:])
            ax1.axis('equal')
            ax1.set_title('Potential-based planning paths')
            domain = np.linspace(1,100, steps)
            ax2.plot(domain, u_path.reshape((steps,)))
            ax2.set_title('Potential function')
            ax2.axis('equal')
            goal_paths.append(x_path)

        if save_plot:
            os.makedirs('assets', exist_ok=True)
            plt.savefig(f'assets/planner_run_plot_goal_{goal_idx}.png')

            path_frames = []
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
            for path_idx, path in enumerate(goal_paths):
                valid_points = []
                for idx in range(path.shape[1]):
                    if not np.isnan(path[0, idx]):
                        valid_points.append(path[:, idx])
                if len(valid_points) > 1:
                    path_frames.append((np.column_stack(valid_points), colors[path_idx % len(colors)]))

            if len(path_frames) > 0:
                animation_fig, animation_ax = plt.subplots(figsize=(6, 6))
                world.plot(animation_ax)
                animation_ax.set_title(f'Planner path animation for goal {goal_idx}')
                animation_ax.set_aspect('equal')
                animation_ax.set_xlim(-11, 11)
                animation_ax.set_ylim(-11, 11)

                max_frames = max(path.shape[1] for path, _ in path_frames)

                def update(frame):
                    animation_ax.clear()
                    world.plot(animation_ax)
                    animation_ax.set_title(f'Planner path animation for goal {goal_idx}')
                    animation_ax.set_aspect('equal', adjustable='box')
                    animation_ax.set_xlim(-11, 11)
                    animation_ax.set_ylim(-11, 11)
                    for path_points, color in path_frames:
                        frame_idx = min(frame, path_points.shape[1] - 1)
                        displayed_points = path_points[:, :frame_idx + 1]
                        if displayed_points.size > 0:
                            animation_ax.plot(displayed_points[0, :], displayed_points[1, :], color=color, lw=2)
                            animation_ax.plot(displayed_points[0, -1], displayed_points[1, -1], 'o', color=color, markersize=6)
                    return []

                frame_step = 40
                frame_indices = range(0, max_frames, frame_step)

                animation = FuncAnimation(
                    animation_fig, 
                    update, 
                    frames=frame_indices, 
                    interval=10, 
                    blit=False
                )
                animation.save(
                    f'assets/planner_path_animation_goal_{goal_idx}.gif', 
                    writer='pillow',
                    fps=10
                )
                plt.close(animation_fig)
                print(f"Saved animation for goal {goal_idx} as 'assets/planner_path_animation_goal_{goal_idx}.gif'")

def clfcbf_control_test_singlesphere(save_img=False):
    """
    Use the provided function Grid.plot_threshold ( ) to visualize 
    the CLF-CBF control field for a single filled-in sphere
    """
    # A single sphere whose edge intersects the origin
    world = pot.SphereWorld()
    world.world = [
        geometry.Sphere(center=np.array([[0], [-2]]),
                              radius=2,
                              distance_influence=1)
    ]
    world.x_goal = np.array([[0], [-6]])
    pars = {
        'repulsive_weight': 2,
        'x_goal': np.array([[0], [-6]]),
        'shape': 'conic'
    }

    xx_ticks = np.linspace(-10, 10, 23)
    grid = geometry.Grid(xx_ticks, xx_ticks)

    clfcbf = pot.Clfcbf_Control(world, pars)
    clfcbf.control(np.zeros((2,1)))
    plt.figure()
    world.plot()
    grid.plot_threshold(clfcbf.control, 1)
    if save_img:
        plt.savefig('assets/clfcbf_control_singlesphere.png')

def clfcbf_run_plot_test(rep_wgt=1, shape='conic', lr=1e-3, steps=20):
    """
    Use the function Planner.run_plot to run the planner based on the
    CLF-CBF framework, and show the results for one combination of
    repulsive_weight and  epsilon that makes the planner work reliably.
    """
    world = pot.SphereWorld() # load problem data
    nb_goal = world.x_goal.shape[1]

    for idx in range(nb_goal):
        # Single goal location
        goal = world.x_goal[:, idx]
        # Create parameters dictionary
        potential_dict = {
            'x_goal': goal,
            'repulsive_weight': rep_wgt,
            'shape': shape
        }
        # Create clf/cbf object
        control = pot.Clfcbf_Control(world, potential_dict)
        # Create path planner obejct
        planner = pot.Planner(
            function = control.function,
            control = control.control,
            epsilon = lr,
            nb_steps = steps
        )
        planner.run_plot()


# Report questions
def run_sphere_collision_test(save_plot=False):
    """
    Sphere collision test
    """
    plt.figure()
    sphere_test_collision(save_plot=save_plot)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.axis('equal')
    plt.show()

def plot_sphere_potential_fields(save_plot=False):
    """
    Report question 2.1
    """
    sphere_world = pot.SphereWorld()
    spheres = sphere_world.world[0:2] # extract first two spheres
    xx_ticks = np.linspace(-11,11,51)

    for idx, sphere in enumerate(spheres):
        test_grid = geometry.Grid(xx_ticks, xx_ticks)
        plt.figure()
        sphere.plot('k')
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.axis('equal')
        rep_sphere = pot.RepulsiveSphere(sphere)
        test_grid.plot_threshold(rep_sphere.grad,10, save_plot=save_plot, outfile=f'assets/sphere_{idx}_repulsive.png')
        plt.show()

    for idx, sphere in enumerate(spheres):
        test_grid = geometry.Grid(xx_ticks, xx_ticks)
        plt.figure()
        sphere.plot('k')
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.axis('equal')
        rep_sphere = pot.RepulsiveSphere(sphere)
        test_grid.plot_threshold(rep_sphere.grad,10, save_plot=save_plot, outfile=f'assets/sphere_{idx}_repulsive.png')
        plt.show()

def run_planner_test(save_plot=False):
    """
    Report question 2.3
    """
    rep_wgt = 1
    shape = 'conic'
    lr = 1e-2
    steps = 15000
    planner_run_plot_test(rep_wgt, shape, lr, steps, save_plot)

def plot_3d_potential(save_plot=False):
    """
    Report question 2.4
    """
    plt.figure()
    world = pot.SphereWorld()
    world.plot()

    potential = {
            'x_goal': world.x_goal[:,0],
            'repulsive_weight': 0.05,
            'shape': 'conic'
            }

    total_u = pot.Total(world, potential)
    def negative_grad(x):
        return -total_u.grad(x)
    planner = pot.Planner(
        function = total_u.eval,
        control = negative_grad,
        epsilon = 1e-2,
        nb_steps = 2500
        )

    xx_ticks = np.linspace(-11,11,50)
    grid = geometry.Grid(xx_ticks, xx_ticks)
    grid.plot_threshold(planner.function, save_plot=save_plot, outfile='assets/potential_3d.png')
    plt.show()

def plot_clfcbf_control_singlesphere(save_img=False):
    """
    Report question 3.3
    """
    clfcbf_control_test_singlesphere(save_img=save_img)
    plt.show()

def plot_clfcbf_control_3_5():
    """
    Report question 3.5
    """
    clfcbf_run_plot_test(rep_wgt=2, shape='conic', lr=0.1, steps=1500)
    plt.show()

def plot_sphere_world_control_field(save_plot=False):
    """
    Report question 3.6
    """
    plt.figure()
    world = pot.SphereWorld()
    world.plot()
    xx_ticks = np.linspace(-8,8,10)
    test_grid = geometry.Grid(xx_ticks, xx_ticks)
    potential = {
        'x_goal': world.x_goal[:,0],
        'repulsive_weight': 0.1,
        'shape': 'quadratic'
    }
    clf_cbf = pot.Clfcbf_Control(world, potential)
    test_grid.plot_threshold(clf_cbf.control)

    if save_plot:
        plt.savefig('assets/clf_cbf_control_sphere_world.png')
    plt.show()

def two_link_sphere_world(save_gif=False):
    """
    Report question 4.2
    """
    world = pot.SphereWorld()
    potential = {
        'x_goal': world.x_goal[:,0], # first goal location
        'repulsive_weight': 1, 
        'shape': 'quadratic'
    }
    two_link = robot.TwoLinkPotential(world, potential)
    epsilon = 1e-3
    nb_steps = 1500
    two_link.run_plot(epsilon, nb_steps, save_gif)

if __name__ == '__main__':
    # Sphere collision test (1)
    # run_sphere_collision_test(save_plot=True)
    
    # Plot potential vector (2.1)
    # plot_sphere_potential_fields(save_plot=True)
        
    # Potential-based planning 
    # run_planner_test(save_plot=True)

    # 3D potential plot
    # plot_3d_potential(save_plot=True)

    # CLF-CBF control field for a single sphere
    # plot_clfcbf_control_singlesphere(save_img=True)
    
    # plot_clfcbf_control_3_5()
    # plot_sphere_world_control_field(save_plot=True)

    two_link_sphere_world(save_gif=True)
    # plt.show()

    pass
