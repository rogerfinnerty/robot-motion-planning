"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import numpy as np
import geometry as geometry

class TwoLink:
    """ This class was introduced in a previous homework. """
    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the vertices of the links, all
    transformed according to  theta_1, theta_2.
        """
        # Extract angles 1 and 2
        theta_1 = theta[0]
        theta_2 = theta[1]

        # Define end effector position (expressed in B2 coordinates)
        p_eff_b2 = np.array([[5],[0]])

        # Define rigid body transformation matrices
        rot_b1_w = geometry.rot2d(theta_1)  # rotation b1 --> W
        rot_b2_b1 = geometry.rot2d(theta_2) # rotation b2 --> b1
        rot_b2_w = np.matmul(rot_b1_w, rot_b2_b1)   # rotation b2 --> w
        trans_b2_b1 = np.array([[5],[0]])   # translation b2 --> b1

        # Create manipulator links (in their local coordinates)
        polygon1_transf, polygon2_transf = polygons_generate()

        # Transform polygons
        polygon1_transf.vertices = np.matmul(rot_b1_w, polygon1_transf.vertices)
        polygon2_transf.vertices = np.matmul(rot_b2_w, polygon2_transf.vertices) + \
            np.matmul(rot_b1_w, trans_b2_b1)

        vertex_effector_transf = np.matmul(rot_b2_w, p_eff_b2) + np.matmul(rot_b1_w, trans_b2_b1)

        return vertex_effector_transf, polygon1_transf, polygon2_transf

    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        [_, polygon1_transf, polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(color=color)
        polygon2_transf.plot(color=color)

    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """
        flag_theta = []
        nb_theta = theta.shape[1]

        for theta_idx in range(nb_theta):
            # Extract single theta values
            single_theta = theta[:,theta_idx].reshape(2,1)
            # Transform manipulator links according to theta1, theta2
            _, link1_transf, link2_transf = self.kinematic_map(single_theta)
            # Check both links for collisions with the array of test points
            flags1 = link1_transf.is_collision(points)
            flags2 = link2_transf.is_collision(points)
            if (any(flags1) or any(flags2)):
                flag_theta.append(True)
            else:
                flag_theta.append(False)

        return flag_theta

    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.

        """
        nb_theta = theta.shape[1]
        # Determine if each configuration is a collision or not
        flag_collisions = self.is_collision(theta, points)
        for idx in range(nb_theta):
            single_theta = theta[:, idx]
            if flag_collisions[idx] is True:
                color = 'r'
            else:
                color = 'g'
            # Plot configuration based on color
            self.plot(single_theta, color)

    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """
        vertex_effector_dot = np.zeros((2,theta.shape[1]))
        p_eff_b2 = np.array([[5],[0]])

        trans_b2_b1 = np.array([[5],[0]])   # Translation from b2 --> b1

        for idx in range(theta.shape[1]):
            # B1 to W rotation matrix and derivative
            rot_b1_w = geometry.rot2d(theta[0,idx])
            theta1_dot_mat = np.array([[0, -theta_dot[0,idx]],[theta_dot[0,idx], 0]])
            rot_dot_b1_w = np.matmul(theta1_dot_mat, rot_b1_w)

            # B2 to B1 rotation matrix and derivative
            rot_b2_b1 = geometry.rot2d(theta[1,idx])
            theta2_dot_mat = np.array([[0, -theta_dot[1,idx]],[theta_dot[1,idx], 0]])
            rot_dot_b2_b1 = np.matmul(theta2_dot_mat, rot_b2_b1)

            # Derivative of rotation b2 --> w
            rot_dot_b2_w = np.matmul(rot_dot_b1_w, rot_b2_b1) + \
                np.matmul(rot_b1_w, rot_dot_b2_b1)

            p_eff_dot = np.matmul(rot_dot_b2_w, p_eff_b2) + \
                np.matmul(rot_dot_b1_w, trans_b2_b1)

            vertex_effector_dot[:,idx] = p_eff_dot.reshape(2)

        return vertex_effector_dot

def polygons_add_x_reflection(vertices):
    """
    Given a sequence of vertices, adds other vertices by reflection
    along the x axis
    """
    vertices = np.hstack([vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
    return vertices

def polygons_generate():
    """
    Generate the polygons to be used for the two-link manipulator
    """
    vertices1 = np.array([[0, 5], [-1.11, -0.511]])
    vertices1 = polygons_add_x_reflection(vertices1)
    vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                          [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
    vertices2 = polygons_add_x_reflection(vertices2)
    return (geometry.Polygon(vertices1), geometry.Polygon(vertices2))
