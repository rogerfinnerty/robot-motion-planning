"""
Main file for ME570 HW2
"""

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import io as scio

import geometry as geometry
import robot as robot


def twolink_plot_collision_test(nb_configurations, save_plot):
    """
    This function generates 30 random configurations, loads the  points variable from the file
    twolink_testData.mat (provided with the homework), and then display the results
    using  twolink_plotCollision to plot the manipulator in red if it is in collision, and green
    otherwise.
    """
    two_link = robot.TwoLink()
    theta_random = 2 * math.pi * np.random.rand(2, nb_configurations)
    test_data = scio.loadmat('twolink_testData.mat')
    obstacle_points = test_data['obstaclePoints']
    plt.plot(obstacle_points[0, :], obstacle_points[1, :], 'r*')
    for i_theta in range(0, nb_configurations):
        theta = theta_random[:, i_theta:i_theta + 1]
        two_link.plot_collision(theta, obstacle_points)
    if save_plot:
        plt.savefig('images/twolink_collision_test.png')
    


def grid_eval_example():
    """ Example of the use of Grid.mesh and Grid.eval functions"""
    def fun(x_vec):
        return math.sin(x_vec[0])

    example_grid = geometry.Grid(np.linspace(-3, 3), np.linspace(-3, 3))
    fun_eval = example_grid.eval(fun)
    [xx_grid, yy_grid] = example_grid.mesh()
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.plot_surface(xx_grid, yy_grid, fun_eval)
    plt.show()


def torus_twolink_plot_jacobian():
    """
    For each one of the curves used in Question~ q:torusDrawChartsCurves, do the following:
 - Use Line.linspace to compute the array  thetaPoints for the curve;
 - For each one of the configurations given by the columns of  thetaPoints:
 - Use Twolink.plot to plot the two-link manipulator.
 - Use Twolink.jacobian to compute the velocity of the end effector, and then use quiver to draw
that velocity as an arrow starting from the end effector's position.   The function should produce a
total of four windows (or, alternatively, a single window with four subplots), each window (or
subplot) showing all the configurations of the manipulator superimposed on each other. You can use
matplotlib.pyplot.ion and insert a time.sleep command in the loop for drawing the manipulator, in
order to obtain a ``movie-like'' presentation of the motion.
    """
    a_lines = [
        np.array([[3 / 4 * math.pi], [0]]),
        np.array([[3 / 4 * math.pi], [3 / 4 * math.pi]]),
        np.array([[-3 / 4 * math.pi], [3 / 4 * math.pi]]),
        np.array([[0], [-3 / 4 * math.pi]])
    ]
    b_line = np.array([[-1], [-1]])

    nb_points=7
    two_link = robot.TwoLink()

    for a_line in a_lines:
        plt.figure()
        theta_points = geometry.line_linspace(a_line, b_line, t_min=0, t_max=1, nb_points=nb_points)
        theta_dot=a_line
        for idx in range(nb_points):
            single_theta = theta_points[:,idx].reshape(2,1)
            end_eff_pos, _, _ = two_link.kinematic_map(single_theta)
            two_link.plot(single_theta,'k')
            vertex_eff_dot = two_link.jacobian(single_theta, theta_dot)
            vertex_eff_dot /= np.linalg.norm(vertex_eff_dot)
            eff_x = end_eff_pos[0,0]
            eff_y = end_eff_pos[1,0]
            d_x = vertex_eff_dot[0,0]
            d_y = vertex_eff_dot[1,0]
            plt.arrow(eff_x, eff_y, d_x, d_y, width = 0.05, color='b')
            plt.axis('equal')
        plt.show()

# Question 1
def rotation_3d(theta, save_plot=False):
    """
    For a given angle theta, compute the 3D rotation matrix 
    that corresponds to a rotation of angle theta in the plane
    """
    e_x = np.array([[1],[0],[0]])
    e_y = np.array([[0],[1],[0]])
    e_z = np.array([[0],[0],[1]])
    rot_2d = geometry.rot2d(theta)
    rot_3d = np.array([
        [-rot_2d[0,0], -rot_2d[0,1], 0],
        [-rot_2d[1,0], -rot_2d[1,1], 0],
        [0,0,1]
    ])
    result = np.matmul(rot_3d, e_x)
    print(result)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0,e_x[0,0]], [0,e_x[1,0]], [0,e_x[2,0]], label="ex")
    ax.plot([0,result[0,0]], [0,result[1,0]], [0,result[2,0]], label="rotated ex")
    plt.show()
    if save_plot:
        fig.savefig('images/rotation_3d.png')
    # ax.axis('equal')

# Question 3
def question_3(save_plot=False):
    """
    Plot a taurus
    """
    test_torus = geometry.Torus()
    test_torus.plot(alpha=0.5)
    test_torus.plot_curves(save_plot)
    plt.show()

# Question 4
def question_4(theta, save_plot=False):
    two_link_test = robot.TwoLink()
    two_link_test.plot(theta, 'k')
    plt.axis('equal')
    twolink_plot_collision_test(nb_configurations = 1, save_plot=save_plot)
    plt.show()

# Question 5
def question_5_3():
    """Report 5.3"""
    two_link = robot.TwoLink()
    theta_vals = [
        np.array([[0],[0]]),
        np.array([[0],[math.pi/2]]),
        np.array([[math.pi],[math.pi/2]]),
        np.array([[math.pi],[math.pi]])
        ]
    a_lines =[
        np.array([[1],[0]]),
        np.array([[0],[1]])
    ]
    for theta in theta_vals:
        for a_line in a_lines:
            plt.figure()
            theta_dot = a_line
            two_link.plot(theta,'k')
            # Transform end effector position
            vertex_eff_transf, _, _ = two_link.kinematic_map(theta)
            eff_x = vertex_eff_transf[0,0]    # x-coord
            eff_y = vertex_eff_transf[1,0]    # y-coord
            vertex_eff_dot = two_link.jacobian(theta, theta_dot)
            #vertex_eff_dot /= np.linalg.norm(vertex_eff_dot)
            d_x = vertex_eff_dot[0,0]
            d_y = vertex_eff_dot[1,0]
            plt.scatter(eff_x, eff_y)
            plt.arrow(eff_x, eff_y, d_x, d_y, width=0.05, color='b')
            title = str
            plt.title(f'Theta=[{theta[0,0]:.2f}, {theta[1,0]:.2f}]\
                      a_line = [{a_line[0,0]}, {a_line[1,0]}]')
            plt.axis('equal')
            plt.show()

if __name__ == "__main__":
    # Question 1
    theta = math.pi/4
    # rotation_3d(theta, save_plot=True)

    # question_3(save_plot=True)

    theta = np.array([[math.pi/4],[math.pi/4]])
    question_4(theta, save_plot=True)
    # question_5_3()
    # torus_twolink_plot_jacobian()