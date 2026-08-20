"""
Substitute the class Grid from the previous homework assignments with the new version below
"""
import math
import numbers
import numpy as np
import matplotlib.pyplot as plt

def rot2d(theta):
    """
    Create a 2-D rotation matrix from the angle  theta according to (1).
    """
    rot_theta = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
    return rot_theta

def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental numerical
    types
    """
    if isinstance(var, bool) or isinstance(var, numbers.Number) or isinstance(
            var, np.number) or isinstance(var, np.bool_):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size

class Grid:
    """ A class to store the coordinates of points on a 2-D grid and evaluate arbitrary functions on
those points. """
    def __init__(self, xx_grid, yy_grid):
        """
        Stores the input arguments in attributes.
        """
        def ensure_1d(val):
            """
            Ensure that the array is 1-D
            """
            if len(val.shape) > 1:
                val = np.reshape(val, (-1))
            return val

        self.xx_grid = ensure_1d(xx_grid)
        self.yy_grid = ensure_1d(yy_grid)
        self.fun_evalued = None

    def eval(self, fun):
        """
        This function evaluates the function  fun (which should be a function)
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

        self.fun_evalued = fun_eval
        return fun_eval

    def mesh(self):
        """
        Shorhand for calling meshgrid on the points of the grid
        """

        return np.meshgrid(self.xx_grid, self.yy_grid)

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

    def plot(self, axes=None, **kwargs):
        """
        Plot the polygon using Matplotlib. Accept an optional `axes` to draw into.
        """
        vertices = self.vertices
        if axes is None:
            axes = plt.gca()
        #plt.xlim(np.min(vertices[0,:]) * 2, np.max(vertices[0,:]) * 2)
        #plt.ylim(np.min(vertices[1,:]) * 2, np.max(vertices[1,:]) * 2)
        n_vert = vertices.shape[1]
        # Plot last vector
        x_1 = vertices[0,n_vert-1]
        y_1 = vertices[1,n_vert-1]
        x_2 = vertices[0,0]
        y_2 = vertices[1,0]
        axes.arrow(x_1, y_1, (x_2 - x_1), (y_2 - y_1), width=0.08, head_width=0.16, **kwargs)
        for i in range(n_vert - 1):
            x_1 = vertices[0,i]
            y_1 = vertices[1,i]
            x_2 = vertices[0,i+1]
            y_2 = vertices[1,i+1]
            axes.arrow(x_1, y_1, (x_2-x_1), (y_2-y_1), width=0.08, head_width=0.16, **kwargs)

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

