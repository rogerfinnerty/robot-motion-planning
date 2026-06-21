"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import geometry as geometry
import potential as pot
# import me570_robot as robot

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

def polygons_add_x_reflection(vertices):
    """
    Given a sequence of vertices, adds other vertices by reflection
    along the x axis
    """
    vertices = np.hstack([vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
    return vertices

class TwoLink:
    """ See description from previous homework assignments. """
    def jacobian_matrix(self, theta):
        """
        Compute the matrix representation of the Jacobian of the position of the end effector with
    respect to the joint angles as derived in Question~ q:jacobian-matrix.
        """
        # Extract angles 1 and 2
        theta_1 = theta[0]
        theta_2 = theta[1]
        j_theta = 5 * np.array([
            [
                -math.sin(theta_1) - math.sin(theta_1+theta_2),
                -math.sin(theta_1+theta_2)
            ],
            [
                math.cos(theta_1)+math.cos(theta_1+theta_2),
                math.cos(theta_1+theta_2)
            ]
        ])
        # Define each value in the Jacobian matrix separately
        # j_theta_0_0 = - 5 * (math.sin(theta_1)*math.cos(theta_2) +
        #                     math.sin(theta_1)*math.cos(theta_2) +
        #                     math.sin(theta_1))
        # j_theta_0_1 = - 5 * (math.sin(theta_1)*math.cos(theta_2) +
        #                      math.cos(theta_1)*math.cos(theta_2))
        # j_theta_1_0 = 5 * (math.cos(theta_1)*math.cos(theta_2)-
        #                    math.sin(theta_1)*math.sin(theta_2) +
        #                    math.cos(theta_1))
        # j_theta_1_1 = 5 * (math.cos(theta_1)*math.cos(theta_2)-
        #                    math.sin(theta_1)*math.sin(theta_2))
        # j_theta = np.array([
        #     [j_theta_0_0, j_theta_0_1],
        #     [j_theta_1_0, j_theta_1_1]
        #     ])

        return j_theta

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
        This function should use TwoLink.kinematic_map from the 
        previous question together with the method Polygon.plot 
        from Homework 1 to plot the manipulator.
        """
        [_, polygon1_transf, polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(color=color)
        polygon2_transf.plot(color=color)

    def animate(self, theta):
        """
        Draw the two-link manipulator for each column in theta 
        with a small pause between each drawing operation
        """
        theta_steps = theta.shape[1]
        for i_theta in range(0, theta_steps, 15):
            self.plot(theta[:, [i_theta]], 'k')


class TwoLinkPotential:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential

    def eval(self, theta_eval):
        """
        Compute the potential U pulled back through the kinematic 
        map of the two-link manipulator,i.e., U(Wp_eff(theta)), 
        where U is defined as in Question~q:total-potential, and
        Wp_ eff(theta) is the position of the end effector in the 
        world frame as a function of the joint angles   = _1\\ _2.
        """
        two_link = TwoLink()
        planner = pot.Total(self.world,self.potential)

        p_eff, _, _ = two_link.kinematic_map(theta_eval)

        # Evaluate total potential @ end effector coordinate
        u_eval_theta = planner.eval(p_eff)

        return u_eval_theta

    def grad(self, theta_eval):
        """
        Compute the gradient of the potential U pulled back through the kinematic map of the
        two-link manipulator, i.e., grad U(  Wp_ eff(  )).
        """
        two_link = TwoLink()
        planner = pot.Total(self.world, self.potential)

        p_eff, _, _ = two_link.kinematic_map(theta_eval)

        grad_u = planner.grad(p_eff)

        jacobian = two_link.jacobian_matrix(theta_eval)

        grad_u_eval_theta = np.matmul(jacobian.T, grad_u)

        return grad_u_eval_theta

    def run_plot(self, epsilon, nb_steps, save_gif=False):
        """
        This function performs the same steps as Planner.run_plot in
        Question~q:potentialPlannerTest, except for the following:
     - In step  it:grad-handle:  planner_parameters['U'] should be set to  @twolink_total, and
    planner_parameters['control'] to the negative of  @twolink_totalGrad.
     - In step  it:grad-handle: Use the contents of the variable  thetaStart instead of  xStart to
    initialize the planner, and use only the second goal  x_goal[:,1].
     - In step  it:plot-plan: Use Twolink.plotAnimate to plot a decimated version of the results of
    the planner. Note that the output  xPath from Potential.planner will really contain a sequence
    of join angles, rather than a sequence of 2-D points. Plot only every 5th or 10th column of
    xPath (e.g., use  xPath(:,1:5:end)). To avoid clutter, plot a different figure for each start.
        """
        sphere_world = pot.SphereWorld()

        nb_starts = sphere_world.theta_start.shape[1]

        def negative_grad(x):
            return -self.grad(x)

        planner = pot.Planner(function=self.eval,
                              control = negative_grad,
                              #control=self.grad,
                              epsilon=epsilon,
                              nb_steps=nb_steps)

        two_link = TwoLink()

        for start in range(0, nb_starts):
            # Run the planner
            theta_start = sphere_world.theta_start[:, [start]]
            theta_path, u_path = planner.run(theta_start)

            # Plots
            _, axes = plt.subplots(ncols=2)
            axes[0].set_aspect('equal', adjustable='box')
            plt.sca(axes[0])
            sphere_world.plot()
            two_link.animate(theta_path)
            axes[1].plot(u_path.T)
    


