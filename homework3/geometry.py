"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import math
import numbers

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

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

def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental
    numerical types
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

class Grid:
    """
    A function to store the coordinates of points on a 2-D grid and evaluate
    arbitrary functions on those points.
    """
    def __init__(self, xx_ticks, yy_ticks):
        """
        Stores the input arguments in attributes.
        """
        self.xx_ticks = xx_ticks
        self.yy_ticks = yy_ticks

    def eval(self, fun):
        """
        This function evaluates the function  fun (which should be a function)
        on each point defined by the grid.
        """

        dim_domain = [numel(self.xx_ticks), numel(self.yy_ticks)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_ticks[idx_x]],
                                   [self.yy_ticks[idx_y]]])
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

        return np.meshgrid(self.xx_ticks, self.yy_ticks)

    def plot_threshold(self, f_handle, threshold=10, save_plot=False, outfile='images/output.png'):
        """
        The function evaluates the function  f_handle on points placed on the grid.
        """
        def f_handle_clip(val):
            return clip(f_handle(val), threshold)

        f_eval = self.eval(f_handle_clip)

        [xx_mesh, yy_mesh] = self.mesh()
        f_dim = numel(f_handle_clip(np.zeros((2, 1))))
        if f_dim == 1:
            # scalar field
            fig = plt.gcf()
            axis = fig.add_subplot(111, projection='3d')

            axis.plot_surface(xx_mesh,
                              yy_mesh,
                              f_eval.transpose(),
                              cmap=cm.gnuplot2)
            axis.set_zlim(0, threshold)
        elif f_dim == 2:
            # vector field

            # grid.eval gives the result transposed with respect to
            # what meshgrid expects
            f_eval = f_eval.transpose((1, 0, 2))
            # vector field
            plt.quiver(xx_mesh,
                       yy_mesh,
                       f_eval[:, :, 0],
                       f_eval[:, :, 1],
                       angles='xy',
                       scale_units='xy',
                       scale=1)
            axis = plt.gca()
        else:
            raise NotImplementedError(
                'Field plotting for dimension greater than two not implemented'
            )

        axis.set_xlim(-15, 15)
        axis.set_ylim(-15, 15)
        plt.xlabel('x')
        plt.ylabel('y')
        if save_plot:
            plt.savefig(outfile)



class Sphere:
    """ Class for plotting and computing distances to spheres (circles, in 2-D). """
    def __init__(self, center, radius, distance_influence):
        """
        Save the parameters describing the sphere as internal attributes.
        """
        self.center = center
        self.radius = radius
        self.distance_influence = distance_influence

    def plot(self, color, axes=None):
        """
        This function draws the sphere (i.e., a circle) of the given radius, 
        and the specified color,and then draws another circle in gray with 
        radius equal to the distance of influence.
        """
        # Get current axes
        if axes is None:
            axes = plt.gca()

        # Add circle as a patch
        if self.radius > 0:
            # Circle is filled in
            kwargs = {'facecolor': (0.3, 0.3, 0.3)}
            radius_influence = self.radius + self.distance_influence
        else:
            # Circle is hollow
            kwargs = {'fill': False}
            radius_influence = -self.radius - self.distance_influence

        center = (self.center[0, 0], self.center[1, 0])
        axes.add_patch(
            plt.Circle(center,
                       radius=abs(self.radius),
                       edgecolor=color,
                       **kwargs))

        axes.add_patch(
            plt.Circle(center,
                       radius=radius_influence,
                       edgecolor=(0.7, 0.7, 0.7),
                       fill=False))

    def distance(self, points):
        """
        Computes the signed distance between points and the sphere, while taking
        into account whether the sphere is hollow or filled in.
        """
        n_points = points.shape[1]
        d_points_sphere = np.zeros((1,n_points))

        for idx in range(n_points):
            point = points[:,idx].reshape((2,1))

            # Compute Euclidean distance between test point and center of sphere
            dist = np.linalg.norm(point - self.center)

            if self.radius > 0: # solid sphere
                d_points_sphere[:,idx] = dist - self.radius
            else: # hollow sphere
                d_points_sphere[:,idx] = abs(self.radius) - dist

        return d_points_sphere

    def distance_grad(self, points):
        """
        Computes the gradient of the signed distance between points and the
        sphere, consistently with the definition of Sphere.distance.
        """
        n_points = points.shape[1]
        grad_d_points_sphere = np.zeros((2,n_points))

        for idx in range(n_points):
            point = points[:,idx].reshape((2,1))
            if np.linalg.norm(point - self.center) == 0:
                grad = np.array([[0.0,0.0]])
            else:
                grad = (point - self.center)/np.linalg.norm(point - self.center)

            if self.radius < 0:
                grad *= -1

            grad_d_points_sphere[:,idx] = grad.reshape((2,))
        return grad_d_points_sphere.reshape((2,n_points))

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


def clip(val, threshold):
    """
    If val is a scalar, threshold its value; if it is a vector, normalized it
    """
    if isinstance(val, np.ndarray):
        val_norm = np.linalg.norm(val)
        if val_norm > threshold:
            val = val * threshold / val_norm
    elif isinstance(val, numbers.Number):
        if np.isnan(val):
            val = threshold
        else:
            val = min(val, threshold)
    else:
        raise ValueError('Numeric format not recognized')

    return val
