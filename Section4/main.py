"""
ME570 Homework 4, Fall 2023
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scio

import graph as graph
import robot as robot
import imageio
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def graph_search_test(save_img=False, outname='assets/graph_search_test.gif'):
    """
    Call graph_search to find a path between the bottom left node and the
    top right node of the  graphVectorMedium graph from the
    graph_test_data_load function (see Question~ q:graph test data). Then
    use Graph.plot() to visualize the result.
    """
    name = 'graphVectorMedium'
    test_graph = graph.Graph(graph.graph_test_data_load(name))
    nb_nodes = len(test_graph.graph_vector)
    idx_start = 0
    idx_goal = nb_nodes-1
    path = test_graph.search(idx_start, idx_goal)
    test_graph.plot(flag_labels=True, idx_goal=idx_goal, flag_backpointers=False)

    # Animate path by drawing it incrementally and save frames to create a GIF
    path_x = path[0, :]
    path_y = path[1, :]
    line, = plt.plot([], [], 'r-', linewidth=6)
    point, = plt.plot([], [], 'ro', markersize=8)

    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)

    fig = plt.gcf()
    # Use an Agg canvas to reliably render each frame independent of the interactive backend
    canvas = FigureCanvas(fig)
    images = []
    for i in range(path.shape[1]):
        line.set_data(path_x[:i+1], path_y[:i+1])
        point.set_data([path_x[i]], [path_y[i]])

        # Draw to the Agg canvas and capture the RGB buffer for this frame
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = canvas.tostring_argb()
        arr = np.frombuffer(buf, dtype='uint8').reshape((h, w, 4))
        img = arr[:, :, 1:4].copy()
        images.append(img)

        plt.pause(0.1)

    # Ensure final path is plotted
    plt.plot(path_x, path_y, 'r')

    # Save GIF if requested
    if save_img and images:
        gif_path = os.path.join('assets', 'graph_search_test.gif')
        imageio.mimsave(gif_path, images, duration=0.1, loop=0)
        print(f"Saved GIF to {gif_path}")

    if save_img:
        plt.savefig('assets/graph_search_test.png')
        print("Graph search test image saved to assets/graph_search_test.png")

def two_link_plot(theta_path, outname):
    '''
    Plot a two-link path both on the graph and in the workspace
    '''
    twolink_graph = robot.TwoLinkGraph()

    # Configuration space plot
    fig_config, ax_config = plt.subplots(figsize=(6, 6))
    twolink_graph.plot()
    ax_config.plot(theta_path[0, :], theta_path[1, :], 'r', linewidth=2)
    ax_config.set_xlabel(r'$\theta_1$ (rad)')
    ax_config.set_ylabel(r'$\theta_2$ (rad)')
    ax_config.set_title('Two-link configuration space')
    ax_config.set_aspect('equal', adjustable='box')

    # Workspace plot and animation
    twolink = robot.TwoLink()
    obstacle_points = scio.loadmat('data/twolink_testData.mat')['obstaclePoints']
    fig, ax = plt.subplots()
    ax.scatter(obstacle_points[0, :], obstacle_points[1, :], marker='*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Two-link manipulator workspace')
    twolink.animate(theta_path, outname, axes=ax)

    os.makedirs('assets', exist_ok=True)
    config_path = os.path.join('assets', 'twolink_configuration_space_test_plot.png')
    fig_config.savefig(config_path, dpi=200, bbox_inches='tight')
    print(f"Saved configuration-space figure to {config_path}")

    plt.show()

def twolink_test_path(theta_m, outname='assets/twolink_test_path.gif'):
    '''
    Visualize, both in the graph, and in the workspace, a sample path where the second link 
    rotates and then the first link rotates (both with constant speeds).
    '''
    theta_path = np.vstack((np.zeros((1, 75)), np.linspace(0, theta_m, 75)))
    theta_path = np.hstack(
        (theta_path,
         np.vstack((np.linspace(0, theta_m, 75), theta_m * np.ones((1, 75))))))
    two_link_plot(theta_path, outname)

def two_link_search(theta_start: np.ndarray, theta_goal: np.ndarray, outname='assets/twolink_solved.gif'):
    """
    Test the two-link path planning functionality.
    """
    test_data = scio.loadmat('data/twolink_testData.mat')
    obstacle_points = test_data['obstaclePoints']

    two_link = robot.TwoLink()  # for plot animation
    two_link_graph = robot.TwoLinkGraph()  # for path planning
    theta_path = two_link_graph.search_start_goal(theta_start, theta_goal)  # run A*

    # Plot the robot workspace with obstacles and animation
    fig, ax = plt.subplots()
    ax.scatter(obstacle_points[0, :], obstacle_points[1, :], marker='*', color='r')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Two-link manipulator workspace')
    two_link.animate(theta_path, outname, axes=ax)

    # Plot the configuration space with the path in joint-angle coordinates
    fig_config, ax_config = plt.subplots(figsize=(6, 6))
    plt.sca(ax_config)
    two_link_graph.plot()
    ax_config.plot(theta_path[0, :], theta_path[1, :], 'b-', linewidth=2)
    ax_config.scatter(theta_start[0, 0], theta_start[1, 0], color='g', s=80, marker='o', label='start')
    ax_config.scatter(theta_goal[0, 0], theta_goal[1, 0], color='m', s=80, marker='*', label='goal')
    ax_config.set_xlabel(r'$\theta_1$ (rad)')
    ax_config.set_ylabel(r'$\theta_2$ (rad)')
    ax_config.set_title('Two-link configuration space')
    ax_config.legend()
    ax_config.set_aspect('equal', adjustable='box')

    os.makedirs('assets', exist_ok=True)
    config_path = os.path.join('assets', 'twolink_configuration_space.png')
    fig_config.savefig(config_path, dpi=200, bbox_inches='tight')
    print(f"Saved configuration-space figure to {config_path}")

    # plt.figure()
    # two_link_graph.plot()
    # plt.plot(theta_path[0,:], theta_path[1,:])

if __name__=='__main__':
    ### DEMO GRAPHS ###

    # # Plot the small and medium graphs
    # graph.graph_test_data_plot(save_img=True)
    # plt.show()

    # Solve medium graph and plot the solution path
    # graph_search_test(save_img=True)
    # plt.show()

    ### SPHERE WORLD GRAPH TESTS ###

    # # Plot solved sphereworld graph 
    # nb_cells = 20
    # graph.test_sphereworld_graph_plot_solved(nb_cells, save_img=True)
    # plt.show()

    ### TWO-LINK MANIPULATOR TESTS ###

    # # Plot a two-link path both on the graph and in the workspace
    # theta_path = np.array([[0.76, 2.72], [0.12, 5.45]])
    # two_link_plot(theta_path)
    # plt.show()

    # # Animate a sample two-link path where the second link rotates and then the first link rotates (both with constant speeds)
    theta_m = 3 / 4 * np.pi
    twolink_test_path(theta_m, outname='assets/twolink_test_path.gif')
    plt.show()

    # Two-link path planning
    # theta_start = np.array([[0.76], [0.12]])
    # theta_goal = np.array([[2.72], [5.45]])
    # two_link_search(theta_start, theta_goal, outname='assets/twolink_solved.gif')
