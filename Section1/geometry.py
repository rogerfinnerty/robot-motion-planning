#!/usr/bin/env python3
"""
Classes and functions for Polygons and Edges
"""

import math

import numpy as np
from matplotlib import pyplot as plt


class Polygon:
    """
    Class for plotting, drawing, checking visibility and collision with
    polygons.
    """
    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    def flip(self):
        """
        Reverse the order of the vertices (i.e., transform the polygon from
        filled in to hollow and vice versa).
        """
        self.vertices = np.flip(self.vertices,1)

    def plot(self, **kwargs):
        """
        Plot the polygon using Matplotlib.
        """
        vertices = self.vertices
        #plt.xlim(np.min(vertices[0,:]) * 2, np.max(vertices[0,:]) * 2)
        #plt.ylim(np.min(vertices[1,:]) * 2, np.max(vertices[1,:]) * 2)
        n_vert = vertices.shape[1]
        # Plot last vector
        x = vertices[0,n_vert-1]
        y = vertices[1,n_vert-1]
        u = vertices[0,0]
        v = vertices[1,0]
        plt.arrow(x,y,(u-x),(v-y), width=0.02, head_width = 0.02, **kwargs)
        for i in range (n_vert - 1):
            x = vertices[0,i]
            y = vertices[1,i]
            u = vertices[0,i+1]
            v = vertices[1,i+1]
            plt.arrow(x,y,(u-x),(v-y), width=0.02, head_width = 0.06, **kwargs)

    def is_filled(self):
        """
        Checks the ordering of the vertices, and returns whether the polygon is
        filled in or not. A counter-clockwise ordering indicate that the polygon is filled in
        """
        # Determine order of points by calculating the cumulative sum of cross products
        # of adjacent lines
        cross_product_sum = 0
        n_vertices = self.vertices.shape[1]
        for idx in range(n_vertices):
            x1= self.vertices[:, idx][0]
            y1= self.vertices[:, idx][1]

            x2 = self.vertices[:, (idx+1) % n_vertices][0]
            y2 = self.vertices[:, (idx+1) % n_vertices][1]

            x3 = self.vertices[:, (idx+2) % n_vertices][0]
            y3 = self.vertices[:, (idx+2) % n_vertices][1]

            cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)

            cross_product_sum += cross_product

        # Counterclockwise if sum is positive
        return cross_product_sum > 0

    def is_self_occluded(self, idx_vertex, point):
        """
        Given the corner of a polygon, checks whether a given point is
        self-occluded or not by that polygon (i.e., if it is ``inside'' the
        corner's cone or not). Points on boundary (i.e., on one of the sides of
        the corner) are not considered self-occluded. Note that to check
        self-occlusion, we just need a vertex index  idx_vertex. From this, one
        can obtain the corresponding  vertex, and the  vertex_prev and
        vertex_next that precede and follow that vertex in the polygon. This
        information is sufficient to determine self-occlusion. To convince
        yourself, try to complete the corners shown in Figure~
        fig:self-occlusion with clockwise and counterclockwise polygons, and
        you will see that, for each example, only one of these cases can be
        consistent with the arrow directions.
        """
        n_vertices = self.vertices.shape[1]
        vertex_point = self.vertices[:,idx_vertex].reshape(2,1)
        vertex_next = self.vertices[:, (idx_vertex+1) % n_vertices].reshape(2,1)

        # Determine vertex_prev, accounting for edge cases
        if idx_vertex == 0:        # First vertex
            vertex_prev = self.vertices[:, n_vertices-1].reshape(2,1)
        else:                       # Middle vertex
            vertex_prev = self.vertices[:,idx_vertex-1].reshape(2,1)

        # Compute the edge-edge angle and the point-vertex-vertex_next angle
        edge_angle = angle(vertex_point, vertex_prev, vertex_next)
        point_angle = angle(vertex_point, point, vertex_next)

        if point_angle <= edge_angle:
            flag_point = False
        else:
            flag_point = True

        return flag_point

    def is_visible(self, idx_vertex, test_points):
        """
        Checks whether a point p is visible from a vertex v of a polygon. In
        order to be visible, two conditions need to be satisfied:
         - The point p should not be self-occluded with respect to the vertex
        v\\ (see Polygon.is_self_occluded).
         - The segment p--v should not collide with  any of the edges of the
        polygon (see Edge.is_collision).
        """
        n_points = test_points.shape[1] # Number of test points
        n_vertices = self.vertices.shape[1] # Number of vertices
        flag_points = [] # Initialize return array of Booleans

        vertex = self.vertices[:,idx_vertex].reshape(2,1)

        # Iterate over test points
        for point_idx in range(n_points):
            point_p = test_points[:, point_idx].reshape(2,1)
            line_p_v = Edge(np.hstack((vertex, point_p)))

            # Check condition 1 (not self-occluded)
            flag1 = self.is_self_occluded(idx_vertex, point_p)

            # Check condition 2 (p--v does not collide w any edges of the polygon)
            flag2 = False
            for vertex_idx in range(n_vertices):
                vert1 = self.vertices[:, vertex_idx].reshape(2,1)
                vert2 = self.vertices[:, ((vertex_idx+1) % n_vertices)].reshape(2,1)
                test_edge = Edge(np.hstack((vert1, vert2)))
                if test_edge.is_collision(line_p_v):
                    flag2 = True
            if not (flag1 or flag2):
                flag_points.append(True)
            else:
                flag_points.append(False)
        return flag_points

    def is_collision(self, test_points):
        """
        Checks whether the a point is in collision with a polygon (that is,
        inside for a filled in polygon, and outside for a hollow polygon). In
        the context of this homework, this function is best implemented using
        Polygon.is_visible.
        """
        n_test_pts = test_points.shape[1]
        n_vertices = self.vertices.shape[1]
        flag_points = []

        # The test point needs to only be visible from one vertex
        for test_pt_index in range(n_test_pts):
            flag = True
            point_p = test_points[:, test_pt_index].reshape(2,1)
            for vertex_idx in range(n_vertices):
                is_visible = self.is_visible(vertex_idx, point_p)
                if is_visible[0] is True:
                    flag = False
            flag_points.append(flag)

        return flag_points

class Edge:
    """
    Class for storing edges and checking collisions among them.
    """
    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    def is_collision(self, edge):
        """
         Returns  True if the two edges intersect.  Note: if the two edges
        overlap but are colinear, or they overlap only at a single endpoint,
        they are not considered as intersecting (i.e., in these cases the
        function returns  False). If one of the two edges has zero length, the
        function should always return the result that edges are
        non-intersecting.
        """
        # First edge (self) endpoints
        x11, x12 = self.vertices[0,:]
        y11,  y12 = self.vertices[1,:]
        # Second edge (edge) endpoints
        x21, x22 = edge.vertices[0,:]
        y21, y22 = edge.vertices[1,:]

        # Parameterize lines
        mat_a = np.array([[ (x12-x11), - (x22 - x21) ],[ (y12-y11), - (y22-y21) ]])
        mat_b = np.array([[(x21-x11)],[y21-y11]])
        # Compute parameter that generates intersection point
        # if lines are parallel (solver would return an error), they do not intersect
        try:
            t_self, t_edge = np.linalg.solve(mat_a, mat_b)
        except np.linalg.LinAlgError:
            return False
        # Check if both parameter vales are in range (0,1), indicating an intersection
        # if not, they don't
        flag1 = 0 < t_self < 1
        flag2 = 0 < t_edge < 1
        return (flag1 and flag2)

    def plot(self, *args, **kwargs):
        """ Plot the edge """
        plt.plot(self.vertices[0, :], self.vertices[1, :], *args, **kwargs)

def angle(vertex0, vertex1, vertex2, angle_type='unsigned'):
    """
    Compute the angle between two edges  vertex0-- vertex1 and  vertex0--
    vertex2 having an endpoint in common. The angle is computed by starting
    from the edge  vertex0-- vertex1, and then ``walking'' in a
    counterclockwise manner until the edge  vertex0-- vertex2 is found.
    """
    # tolerance to check for coincident points
    tol = 2.22e-16

    # compute vectors corresponding to the two edges, and normalize
    vec1 = vertex1 - vertex0
    vec2 = vertex2 - vertex0
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < tol or norm_vec2 < tol:
        # vertex1 or vertex2 coincides with vertex0, abort
        edge_angle = math.nan
        return edge_angle

    vec1 = vec1 / norm_vec1
    vec2 = vec2 / norm_vec2

    # Transform vec1 and vec2 into flat 3-D vectors,
    # so that they can be used with np.inner and np.cross
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()

    c_angle = np.inner(vec1flat, vec2flat)
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

    edge_angle = math.atan2(s_angle, c_angle)

    angle_type = angle_type.lower()
    if angle_type == 'signed':
        # nothing to do
        pass
    elif angle_type == 'unsigned':
        edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
    else:
        raise ValueError('Invalid argument angle_type')

    return edge_angle

def polygon_plot_test():
    """
    Test for instantiating Polygon class, plotting, and flipping methods
    """
    vert1 = np.array([[1,2,3,5,4],[1,3,9,2,0]])
    poly1 = Polygon(vert1) # Clockwise ordered
    poly1.plot(color='b')

    poly2 = Polygon(vert1) # Counter-clockwise ordered
    poly2.flip()
    poly2.plot(color='b')
