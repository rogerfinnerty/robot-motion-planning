#!/usr/bin/env python3
"""
Test functions for HW1
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, pi, sin

import geometry as geometry
import pqueue as queue
import robot as robot

def edge_is_collision_test(save_fig=False):
    """
    The function creates an edge from  [0;0] to
    [1,1] and a second random edge with endpoints
    contained in the square [0,1] [0,1], and plots them in green if they do
    not overlap, and in red otherwise.
    """
    vertices = [np.array([[0, 1], [0, 1]]), np.random.rand(2, 2)]
    edges = [geometry.Edge(x) for x in vertices]
    flag_collision = edges[0].is_collision(edges[1])
    if flag_collision:  # collision
        style = 'r'
    else:   # no collision
        style = 'g'
    style = style + '-o'
    plt.figure()
    for edge in edges:
        edge.plot(style)

    if save_fig:
        if flag_collision:
            plt.savefig("assets/edge_collision.png", dpi=300)
        else:
            plt.savefig("assets/edge_no_collision.png", dpi=300)

    plt.show()

def polygon_is_self_occluded_test(nb_points=61, save_fig=False):
    """
    Visually test the function polygon_isSelfOccluded by picking random
    arrangements for  vertexPrev and  vertexNext, and systematically picking
    the position of point. The meaning of the green and red lines are
    similar to those shown in  fig:self-occlusion.

    :param nb_points: number of points to test for self-occlusion
    """
    vertex = np.zeros((2, 1))
    angles_test = np.random.rand(2) * 2 * pi

    vertex_prev = np.array([[cos(angles_test[0])], [sin(angles_test[0])]])
    vertex_next = np.array([[cos(angles_test[1])], [sin(angles_test[1])]])

    angle_point = np.linspace(0, 2 * pi, nb_points)
    point = np.vstack([cos(angle_point), sin(angle_point)])

    polygon = geometry.Polygon(np.hstack([vertex_prev, vertex, vertex_next]))
    polygon.plot(color='k')

    for i_point in range(nb_points):
        if polygon.is_filled():
            title = "Solid polygon"
        else: 
            title = "Hollow polygon"
        # check collision against the second vertex
        # (index 1, corresponding to vertex)
        flag_occluded = polygon.is_self_occluded(1, point[:, [i_point]])
        if flag_occluded:
            style = 'r'
        else:
            style = 'g'
        plt.plot([0, point[0, i_point]], [0, point[1, i_point]], style)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().axis('equal')
    plt.text(vertex_prev[0], vertex_prev[1], 'vertex_prev')
    plt.text(vertex_next[0], vertex_next[1], 'vertex_next')
    plt.text(vertex[0], vertex[1], 'vertex')
    plt.title(title)

    if save_fig:
        if polygon.is_filled():
            plt.savefig("assets/polygon_self_occlusion_test_solid.png", dpi=300)
        else:
            plt.savefig("assets/polygon_self_occlusion_test_hollow.png", dpi=300)

    plt.show()

def polygon_is_visible_test(save_fig=False):
    """
    This function should perform the following operations:
     - Create an array  test_points with dimensions [2 x 5] containing
    points generated uniformly at random using np.random.rand and scaled to
    approximately occupy the rectangle [0,5] [-2,2] (i.e., the x coordinates
    of the points should fall between 0 and 5, while the y coordinates
    between -2 and 2).
     - Obtain the polygons  polygon1 and  polygon2 from TwoLink.Polygons.
     - item:test-polygon For each polygon  polygon1,  polygon2, display a
    separate figure using the following:
     - Create the array  test_points_with_polygon by concatenating
    test_points with the coordinates of the polygon (i.e., the coordinates
    of the polygon become also test points).
     - Plot the polygon (use Polygon.plot).
     - item:test-visibility For each vertex v in the polygon:
     - Compute the visibility of each point in  test_points_with_polygon
    with respect to that polygon (using Polygon.is_visible).
     - Plot lines from the vertex v to each point in
    test_points_with_polygon in green if the corresponding point is visible,
    and in red otherwise.
     - Reverse the order of the vertices in the two polygons using
    Polygon.flip.
     - Repeat item item:test-polygon above with the reversed polygons.
    """

    # Create array
    test_points = np.random.rand(2,1)
    # Scale x coordinates to range [0,5]
    test_points[0,:] *= 5
    # Scale y coordinates to range [-2,2]
    test_points[1,:] = test_points[1,:] * 4 - 2

    # Obtain polygons
    [polygon1, polygon2] = robot.polygons

    for idx, polygon in enumerate([polygon1, polygon1, polygon2, polygon2]):
        plt.figure()
        test_points_with_polygon = np.hstack( (test_points, polygon.vertices) )
        polygon.plot(color = 'k')

        if polygon.is_filled():
            title = "Solid polygon"
        else:
            title = "Hollow polygon"

        vertices = polygon.vertices
        n_vertices = vertices.shape[1]
        n_test_pts = test_points_with_polygon.shape[1]

        for vertex_idx in range(n_vertices):
            # Extract the vertex
            vertex = polygon.vertices[:,vertex_idx].reshape(2,1)
            # Determine visibility of each test point from vertex v
            test_pts_visibility = polygon.is_visible(vertex_idx, test_points_with_polygon)
            for test_pt_idx in range(n_test_pts):
                # Extract a single test point
                test_pt = test_points_with_polygon[:,test_pt_idx].reshape(2,1)
                # Plot test point
                plt.scatter(test_pt[0], test_pt[1], color = 'k')
                # Plot line from vertex to test point based on visibility
                if test_pts_visibility[test_pt_idx]:
                    style = 'g'
                else:
                    style = 'r'
                plt.plot([vertex[0], test_pt[0]], [vertex[1], test_pt[1]], style)
        plt.title(title)

        if save_fig:
            if idx == 0 or idx == 1:
                id = '1'
            else:
                id = '2'
            if polygon.is_filled():
                plt.savefig(f"assets/polygon_visibility_test_solid_{id}.png", dpi=300)
            else:
                plt.savefig(f"assets/polygon_visibility_test_hollow_{id}.png", dpi=300)

        plt.show()

        polygon.flip()

def polygon_is_collision_test_plot(polygon, test_points, idx, save_fig=False):
    """
    Helper function for polygon_is_collision_test to run tests and plot the
    results for a single polygon
    """
    test_points_with_polygon = np.hstack((polygon.vertices, test_points))
    plt.figure()
    polygon.plot(color='k')
    if polygon.is_filled():
        title = "Solid polygon"
    else:
        title = "Hollow polygon"

    green_x = []
    green_y = []
    red_x = []
    red_y = []

    flag_points = polygon.is_collision(test_points_with_polygon)
    for i, point in enumerate(test_points_with_polygon.T):
        x_point = point[0]
        y_point = point[1]
        if flag_points[i] is True:
            red_x.append(x_point)
            red_y.append(y_point)
        else:
            green_x.append(x_point)
            green_y.append(y_point)

    plt.scatter(green_x, green_y, color='g')
    plt.scatter(red_x, red_y, color='r')
    plt.title(title)

    if save_fig:
        if polygon.is_filled():
            plt.savefig(f"assets/polygon_collision_test_solid_{idx+1}.png", dpi=300)
        else:
            plt.savefig(f"assets/polygon_collision_test_hollow_{idx+1}.png", dpi=300)

def polygon_is_collision_test(save_fig=False):
    """
    This function is the same as polygon_is_visible_test, but instead of
    step  item:test-visibility, use the following:
     - Compute whether each point in  test_points_with_polygon is in
    collision with the polygon or not using Polygon.is_collision.
     - Plot each point in  test_points_with_polygon in green if it is not in
    collision, and red otherwise.  Moreover, increase the number of test
    points from 5 to 100 (i.e.,  testPoints should have dimension [2 x
    100]).
    """
    test_points = np.random.rand(2, 100)

    # Scale x coordinates to uniformly cover [0, 5)
    test_points[0, :] *= 5

    # Scale y coordinates to uniformly cover [-2, 2)
    #   formula used: low + ((high - low) * random_value)
    test_points[1, :] *= 4  # high - low
    test_points[1, :] -= 2  # low

    # Loop over polygon1 and polygon2
    for idx, polygon in enumerate(robot.polygons):
        polygon_is_collision_test_plot(polygon, test_points, idx, save_fig)
        polygon.flip()
        polygon_is_collision_test_plot(polygon, test_points, idx, save_fig)

def priority_test():
    """
    The function should perform the following steps:  enumerate
     - Initialize an empty queue as the object  p_queue.
     - Add three elements (as shown in Table~tab:priority-test-inputs and in
    that order) to that queue.
     - Extract a minimum element.
     - Add another element (as shown in Table~tab:priority-test-inputs).
     - Check if an element (which is in the queue) is present.
     - Check if an element (which is  not in the queue) is present.
     - Remove all elements by repeated extractions.  enumerate After each
    step, display the content of  p_queue.
    """
    p_queue = queue.PriorityQueue()
    p_queue.print_queue()
    print('\n')

    p_queue.insert('Oranges', 4.5)
    p_queue.insert('Apples', 1)
    p_queue.insert('Bananas', 2.7)
    p_queue.print_queue()
    print('\n')

    min_key, min_cost = p_queue.min_extract()
    print(f"Minimum key: {min_key}, cost($): {min_cost}")

    p_queue.insert('Cantaloupe', 3)
    p_queue.print_queue()
    print('\n')

    test_keys = ['Apples', 'Bananas', '(1,5)']
    for key in test_keys: 
        if p_queue.is_member(key):
            print(f"{key} is present in the queue.")
        else:
            print(f"{key} is not present in the queue.")
    print('\n')
    
    # Remove elements of the queue starting from the last element
    for i in range(len(p_queue.queue_list)):
        extracted_item = p_queue.queue_list.pop()
        print(f"Removed queue item with key: {extracted_item[0]}, value: {extracted_item[1]}") 
        p_queue.print_queue()
    print('\n')

def descending_display_test():
    """
    Function for testing functionality of method in PriorityQueue to display elements of the 
    queue in descending order 
    """
    p_queue = queue.PriorityQueue()
    p_queue.insert('(x0, y0)', 4.5)
    p_queue.insert('(x1, y1)', 1)
    p_queue.insert('(x2, y2)', 2.7)
    p_queue.insert('(x3, y3)', 3)
    print(f"Original priority queue:")
    p_queue.print_queue()
    print("\n")

    print(f"Sorted priority queue:")
    p_queue.display_descending()

def flip_test():
    """
    Function to test for flipping vertices, detection of hollow/solid polygons
    """
    vert1 = np.array([[1,2,3], [1, 2, 1]])
    poly1 = geometry.Polygon(vert1) # clockwise (hollow)
    poly2 = geometry.Polygon(vert1) # counter-clockwise (solid)
    poly2.flip()

    print(poly1.vertices)
    print(poly2.vertices)

    flag1 = poly1.is_filled()
    flag2 = poly2.is_filled()
    if flag1 :
        print("polygon 1 is solid")
    else:
        print("polygon 1 is hollow")

    if flag2 :
        print("polygon 2 is solid")
    else:
        print("polygon 2 is hollow")

def point_self_occluded_test():
    """
    Function to test for self-occlusion
    """
    poly1, poly2 = robot.polygons

    # Create array
    test_points = np.random.rand(2,1)
    # Scale x coordinates to range [0,5]
    test_points[0,:] *= 5
    # Scale y coordinates to range [-2,2]
    test_points[1,:] = test_points[1,:] * 4 - 2

    n_test_pts = test_points.shape[1]

    if poly1.is_filled():
        title = "Solid polygon"
    else:
        title = "Hollow polygon"

    plt.figure()
    poly1.plot(color='k')
    plt.title(title)

    plt.show()

if __name__ == "__main__":
    # edge_is_collision_test()
    # polygon_is_self_occluded_test()
    # polygon_is_visible_test()
    # polygon_is_collision_test()
    point_self_occluded_test()
