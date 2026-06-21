"""
ME570 Homework 4, Fall 2023
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scio

import me570_graph
import me570_robot

def graph_search_test(save_img=False):
    """
    Call graph_search to find a path between the bottom left node and the
    top right node of the  graphVectorMedium graph from the
    graph_test_data_load function (see Question~ q:graph test data). Then
    use Graph.plot() to visualize the result.
    """
    name = 'graphVectorMedium'
    test_graph = me570_graph.Graph(me570_graph.graph_test_data_load(name))
    nb_nodes = len(test_graph.graph_vector)
    idx_start = 0
    idx_goal = nb_nodes-1
    path = test_graph.search(idx_start, idx_goal)
    test_graph.plot(flag_labels=True, idx_goal=idx_goal, flag_backpointers=False)
    # Plot path
    plt.plot(path[0,:], path[1,:], 'r')
    plt.show()
    if save_img:
        plt.savefig('images/graph_search_test.png')

def twolink_search_plot_solution(theta_path):
    '''
    Plot a two-link path both on the graph and in the workspace
    '''
    twolink_graph = me570_robot.TwoLinkGraph()
    plt.figure(1)
    twolink_graph.plot()
    plt.plot(theta_path[0, :], theta_path[1, :], 'r')

    twolink = me570_robot.TwoLink()
    obstacle_points = scio.loadmat('twolink_testData.mat')['obstaclePoints']
    plt.figure(2)
    plt.scatter(obstacle_points[0, :], obstacle_points[1, :], marker='*')
    twolink.animate(theta_path)

    plt.show()

def twolink_test_path(theta_m):
    '''
    Visualize, both in the graph, and in the workspace, a sample path where the second link 
    rotates and then the first link rotates (both with constant speeds).
    '''
    theta_path = np.vstack((np.zeros((1, 75)), np.linspace(0, theta_m, 75)))
    theta_path = np.hstack(
        (theta_path,
         np.vstack((np.linspace(0, theta_m, 75), theta_m * np.ones((1, 75))))))
    twolink_search_plot_solution(theta_path)

def question3_3():
    test_data = scio.loadmat('twolink_testData.mat')
    obstacle_points = test_data['obstaclePoints']
    plt.plot(obstacle_points[0, :], obstacle_points[1, :], 'r*')

    # Start, goal configurations
    theta_start = np.array([[0.76], [0.12]])
    theta_goal = np.array([[2.72], [5.45]])

    two_link = me570_robot.TwoLink()    # for plot animation
    two_link_graph = me570_robot.TwoLinkGraph()     # for path planning
    theta_path = two_link_graph.search_start_goal(theta_start, theta_goal)  # run A*

    two_link.animate(theta_path)
    plt.figure()

    two_link_graph.plot()
    plt.plot(theta_path[0,:], theta_path[1,:])
    plt.show()

if __name__=='__main__':
    # me570_graph.graph_test_data_plot(save_img=True)
    # plt.show()
    # question3_3()
    # graph_search_test(save_img=True)
    theta_m = 3 / 4 * np.pi
    twolink_test_path(theta_m)
    # twolink_search_plot_solution(np.array([[0.76, 2.72], [0.12, 5.45]]))
    
