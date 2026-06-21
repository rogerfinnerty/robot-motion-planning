"""
ME570 homework 3
"""
import numpy as np
from matplotlib import pyplot as plt
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
        plt.savefig('images/sphere_test_collision.png')

def planner_run_plot_test(rep_wgt=1, shape='quadratic', lr=1e-2, steps=1000, save_plot=False):
    """
    Show the results of Planner.run_plot for each goal location in
    world.xGoal, and for different interesting combinations of
    potential['repulsive_weight'],  potential['shape'],  epsilon, and
    nb_steps. In each case, for the object of class  Planner should have the
    attribute  function set to  Total.eval, and the attribute  control set
    to the negative of  Total.grad.
    """
    world = pot.SphereWorld() # load problem data
    nb_goal = world.x_goal.shape[1]
    nb_start = world.x_start.shape[1]

    def negative_grad(x):
        return -total_u.grad(x)

    for goal_idx in range(nb_goal):
        _, (ax1, ax2) = plt.subplots(ncols=2)
        world.plot(ax1)
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
    
        if save_plot:
            plt.savefig(f'images/planner_run_plot_goal_{goal_idx}.png')

def clfcbf_control_test_singlesphere(save_img=False):
    """
    Use the provided function Grid.plot_threshold ( ) to visualize 
    the CLF-CBF control field for a single filled-in sphere
    """
    # A single sphere whose edge intersects the origin
    world = potential.SphereWorld()
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

    clfcbf = potential.Clfcbf_Control(world, pars)
    clfcbf.control(np.zeros((2,1)))
    plt.figure()
    world.plot()
    grid.plot_threshold(clfcbf.control, 1)
    if save_img:
        plt.savefig('images/clfcbf_control_singlesphere.png')

def clfcbf_run_plot_test(rep_wgt=1, shape='conic', lr=1e-3, steps=20):
    """
    Use the function Planner.run_plot to run the planner based on the
    CLF-CBF framework, and show the results for one combination of
    repulsive_weight and  epsilon that makes the planner work reliably.
    """
    world = potential.SphereWorld() # load problem data
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
        control = potential.Clfcbf_Control(world, potential_dict)
        # Create path planner obejct
        planner = potential.Planner(
            function = control.function,
            control = control.control,
            epsilon = lr,
            nb_steps = steps
        )
        planner.run_plot()


# Report questions
def question_1(save_plot=False):
    """
    Sphere collision test
    """
    plt.figure()
    sphere_test_collision(save_plot=save_plot)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.axis('equal')
    plt.show()

def question_2_1(save_plot=False):
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
        test_grid.plot_threshold(rep_sphere.grad,10, save_plot=save_plot, outfile=f'images/sphere_{idx}_repulsive.png')
        plt.show()

    for sphere in spheres:
        test_grid = geometry.Grid(xx_ticks, xx_ticks)
        plt.figure()
        sphere.plot('k')
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.axis('equal')
        rep_sphere = pot.RepulsiveSphere(sphere)
        test_grid.plot_threshold(rep_sphere.grad,10, save_plot=save_plot, outfile=f'images/sphere_{idx}_repulsive.png')
        plt.show()

def question_2_3(save_plot=False):
    """
    Report question 2.3
    """
    rep_wgt = 1
    shape = 'conic'
    lr = 1e-2
    steps = 15000
    planner_run_plot_test(rep_wgt, shape, lr, steps, save_plot)
    plt.show()

def question_2_4(save_plot=False):
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
    grid.plot_threshold(planner.function, save_plot=save_plot, outfile='images/potential_3d.png')
    plt.show()

def question_3_3(save_img=False):
    """
    Report question 3.3
    """
    clfcbf_control_test_singlesphere(save_img=save_img)
    plt.show()

def question_3_5():
    """
    Report question 3.5
    """
    clfcbf_run_plot_test(rep_wgt=2, shape='conic', lr=0.1, steps=1500)
    plt.show()

def question_3_6():
    """
    Report question 3.6
    """
    plt.figure()
    world = potential.SphereWorld()
    world.plot()
    xx_ticks = np.linspace(-8,8,10)
    test_grid = geometry.Grid(xx_ticks, xx_ticks)
    potential = {
        'x_goal': world.x_goal[:,0],
        'repulsive_weight': 0.1,
        'shape': 'quadratic'
    }
    clf_cbf = potential.Clfcbf_Control(world, potential)
    test_grid.plot_threshold(clf_cbf.control)
    plt.show()

def question_4_2(save_gif=False):
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
    two_link.run_plot(epsilon, nb_steps)
    plt.show()

if __name__ == '__main__':
    # Sphere collision test
    # question_1(save_plot=True)

    # Plot potential vector 
    # question_2_1(save_plot=True)

    # Potential-based planning
    # question_2_3(save_plot=True)

    # 3D potential plot
    # question_2_4(save_plot=True)

    # CLF-CBF control field for a single sphere
    # question_3_3(save_img=True)
    # question_3_5()
    question_3_6()


    # question_4_2()
    # plt.show()
