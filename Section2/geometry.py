"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""

import math
import numbers

import numpy as np
# import scipy
from matplotlib import pyplot as plt


def gca_3d():
    """
    Get current Matplotlib axes, and if they do not support 3-D plotting,
    add new axes that support it
    """
    fig = plt.gcf()
    if len(fig.axes) == 0 or not hasattr(plt.gca(), 'plot3D'):
        axis = fig.add_subplot(111, projection='3d')
    else:
        axis = plt.gca()
    return axis


def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental numerical
    types
    """
    if isinstance(var, numbers.Number):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size


def rot2d(theta):
    """
    Create a 2-D rotation matrix from the angle  theta according to (1).
    """
    rot_theta = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
    return rot_theta


def line_linspace(a_line, b_line, t_min, t_max, nb_points):
    """
    Generates a discrete number of  nb_points points along the curve
    (t)=( a(1)t + b(1), a(2)t + b(2))  R^2 for t ranging from  tMin to  tMax.
    """
    t_sequence = np.linspace(t_min, t_max, nb_points)
    theta_points = a_line * t_sequence + b_line
    return theta_points


class Grid:
    """
    A function to store the coordinates of points on a 2-D grid and evaluate arbitrary
    functions on those points.
    """
    def __init__(self, xx_grid, yy_grid):
        """
        Stores the input arguments in attributes.
        """
        self.xx_grid = xx_grid
        self.yy_grid = yy_grid

    def eval(self, fun):
        """
        This function evaluates the function fun (which should be a function)
        on each point defined by the grid.
        """

        dim_domain = [numel(self.xx_grid), numel(self.yy_grid)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_grid[idx_x]],
                                   [self.yy_grid[idx_y]]])
                fun_eval[idx_x, idx_y, :] = np.reshape(fun(x_eval),
                                                       [1, 1, dim_range[0]])

        # If the last dimension is a singleton, remove it
        if dim_range == [1]:
            fun_eval = np.reshape(fun_eval, dim_domain)

        return fun_eval

    def mesh(self):
        """
        Shorhand for calling meshgrid on the points of the grid
        """
        return np.meshgrid(self.xx_grid, self.yy_grid)

class Torus:
    """
    A class that holds functions to compute the embedding and display a torus and curves on it.
    """
    def phi(self, theta, r_val=3):
        """
        Compute the 3D embedding of a torus from the 2D parameterization theta. The torus is defined by the
        embedding function phi_torus(theta) = (x,y,z) R^3, where
        x = (r + cos(theta_2)) * cos(theta_1)
        y = (r + cos(theta_2)) * sin(theta_1)
        z = sin(theta_2).

        :param theta: 2xN array of points in the parameterization of the torus
        :param r_val: radius of the torus
        :return: 3xN array of points in R^3 corresponding to the embedding
        """
        nb_points = theta.shape[1]
        x_torus = np.zeros((3,nb_points))

        # Iterate over theta values
        for idx in range(nb_points):
            # Extract theta values
            theta_1 = theta[0,idx] # angle in the XY plane
            theta_2 = theta[1,idx] # angle in the XZ plane

            # Create 2D rotation matrix for theta 1
            rot_2d = rot2d(theta_1)

            # Circle map - point on unit circle in the XY plane
            phi_circle = np.matmul(rot_2d, np.array([[1],[0]]))

            # Create 2D rotation matrix for theta 2
            rot_3d_2d = rot2d(theta_2)

            # Create 3D rotation matrix, R3(theta2)
            rot_3d = np.array([
                [rot_3d_2d[0,0], rot_3d_2d[0,1], 0],
                [rot_3d_2d[1,0], rot_3d_2d[1,1],0],
                [0,0,1]
                ])

            # Projection matrix setting circle onto x-z plane
            projection_matrix = np.array([[1,0], [0,0], [0,1]])

            # Translation by 'r_value' in the x-direction
            translation_vector = np.array([[r_val],[0],[0]])

            # Project the circle onto the x-z plane and translate by r_val in the x-direction
            second_term = np.matmul(projection_matrix, phi_circle) + translation_vector
            # Apply 3D rotation matrix to the translated circle to get the final point on the torus
            x_torus[:,idx] = np.matmul(rot_3d, second_term).reshape(3)

        return x_torus

    def plot(self, alpha):
        """
        For each one of the chart domains U_i from the previous question:
        - Fill a  grid structure with fields  xx_grid and  yy_grid that define a grid of regular
          point in U_i. Use nb_grid=33.
        - Call the function Grid.eval with argument Torus.phi.
        - Plots the surface described by the previous step using the the Matplotlib function
        ax.plot_surface (where  ax represents the axes of the current figure) in a separate figure.
        Plot a final additional figure showing all the charts at the same time.   To better show
        the overlap between the charts, you can use different colors each one of them,
        and making them slightly transparent.
        """
        # Define number of points
        nb_grid_x = 33
        nb_grid_y = 33

        # Create grid
        x_vals = np.linspace(0, 2*math.pi, nb_grid_x)
        y_vals = np.linspace(0, 2*math.pi, nb_grid_y)
        grid = Grid(x_vals, y_vals)

        phi_eval = grid.eval(self.phi)
        #[xx_grid, yy_grid] = grid.mesh()
        phi_eval_1 = phi_eval[:,:,0].reshape((33,33))
        phi_eval_2 = phi_eval[:,:,1].reshape((33,33))
        phi_eval_3 = phi_eval[:,:,2].reshape((33,33))

        fig = plt.figure()
        axes = fig.add_subplot(projection = '3d')
        axes.plot_surface(phi_eval_1, phi_eval_2, phi_eval_3, alpha=alpha)


    def phi_push_curve(self, a_line, b_line):
        """
        This function evaluates the curve x(t)= phi_torus ( phi(t) )  R^3 at  nb_points=31 points
        generated along the curve phi(t) using line_linspaceLine.linspace with  tMin=0 and  tMax=1,
        and a, b as given in the input arguments.
        """
        # Generate points along the curve phi(t) using line_linspace
        theta_points = line_linspace(a_line, b_line, t_min=0, t_max=1, nb_points=31)
        x_points = self.phi(theta_points)
        return x_points

    def plot_curves(self, save_plot=False):
        """
        The function should iterate over the following four curves:
        - 3/4*pi, 0
        - 3/4*pi, 3/4*pi
        - -3/4*pi, 3/4*pi
        - 0 -3/4*pi  and  b=np.array([[-1],[-1]]).
        The function should show an overlay containing:
        - The output of Torus.plot;
        - The output of the functions Torus.phi_push_curve for each one of the curves.
        """
        a_lines = [
            np.array([[3 / 4 * math.pi], [0]]),
            np.array([[3 / 4 * math.pi], [3 / 4 * math.pi]]),
            np.array([[-3 / 4 * math.pi], [3 / 4 * math.pi]]),
            np.array([[0], [-3 / 4 * math.pi]])
        ]

        b_line = np.array([[-1], [-1]])

        axis = gca_3d()
        for a_line in a_lines:
            x_points = self.phi_push_curve(a_line, b_line)
            axis.plot(x_points[0, :], x_points[1, :], x_points[2, :], linewidth = 4.0)
        
        if save_plot:
            plt.savefig('assets/torus_curves.png')

    def phi_test(self):
        """
        Uses the function phi to plot two perpendicular rings
        """
        nb_points = 200
        theta_ring = np.linspace(0, (15/8 * np.pi), nb_points)
        theta_zeros = np.zeros((1, nb_points))
        data = [
            np.vstack((theta_ring, theta_zeros)),
            np.vstack((theta_zeros, theta_ring))
        ]
        axis = gca_3d()
        for theta in data:
            ring = np.zeros((3, nb_points))
            for idx in range(nb_points):
                ring[:, [idx]] = self.phi(theta[:, [idx]])
            axis.plot(ring[0, :], ring[1, :], ring[2, :])
        plt.show()

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
        x_1 = vertices[0,n_vert-1]
        y_1 = vertices[1,n_vert-1]
        x_2 = vertices[0,0]
        y_2 = vertices[1,0]
        plt.arrow(x_1,y_1,(x_2 - x_1),(y_2 - y_1), width=0.08, head_width = 0.16, **kwargs)
        for i in range (n_vert - 1):
            x_1 = vertices[0,i]
            y_1 = vertices[1,i]
            x_2 = vertices[0,i+1]
            y_2 = vertices[1,i+1]
            plt.arrow(x_1,y_1,(x_2-x_1),(y_2-y_1), width=0.08, head_width = 0.16, **kwargs)

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
            x_1= self.vertices[:, idx][0]
            y_1= self.vertices[:, idx][1]
            x_2 = self.vertices[:, (idx+1) % n_vertices][0]
            y_2 = self.vertices[:, (idx+1) % n_vertices][1]

            x_3 = self.vertices[:, (idx+2) % n_vertices][0]
            y_3 = self.vertices[:, (idx+2) % n_vertices][1]

            cross_product = (x_2 - x_1) * (y_3 - y_2) - (y_2 - y_1) * (x_3 - x_2)

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
        n_vertices = self.vertices.shape[1] # Number of vertices
        flag_points = [] # Initialize return array of Booleans

        vertex = self.vertices[:,idx_vertex].reshape(2,1)

        # Iterate over test points
        for point_idx in range(test_points.shape[1]):   # iterate through each test point
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
        return (0 < t_self < 1 and 0 < t_edge < 1)

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
