"""
Classes to define potential and potential planner for the sphere world
"""
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as scio

import geometry as geometry
import qp as qp


class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different goal locations (one
    for each column).
        """
        data = scio.loadmat('sphereworld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = sphere_args[1].item()
            sphere_args[2] = sphere_args[2].item()
            self.world.append(geometry.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def plot(self, axes=None):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a  * marker at the goal
        location.
        """

        if axes is None:
            axes = plt.gca()

        for sphere in self.world:
            sphere.plot('r', axes)

        axes.scatter(self.x_goal[0, :], self.x_goal[1, :], c='g', marker='*')

        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.axis('equal')

class RepulsiveSphere:
    """ Repulsive potential for a sphere """
    def __init__(self, sphere):
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """
        Evaluate the repulsive potential from sphere at the location x= x_eval. 
        The function returns the repulsive potential as given by (  eq:repulsive  ).
        """
        distance = self.sphere.distance(x_eval.reshape((2,1)))

        distance_influence = self.sphere.distance_influence
        if distance > distance_influence:
            u_rep = 0
        elif distance_influence > distance > 0:
            u_rep = ((distance**-1 - distance_influence**-1)**2) / 2
            u_rep = u_rep.item()
        else:
            u_rep = math.nan
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by (eq:repulsive-gradient).
        """

        distance = self.sphere.distance(x_eval.reshape((2,1)))
        distance_influence = self.sphere.distance_influence
        grad_distance = self.sphere.distance_grad(x_eval.reshape((2,1)))

        if distance > distance_influence:
            grad_u_rep = np.array([[0.0],[0.0]])
        elif distance_influence > distance > 0:
            grad_u_rep = -(distance**-1 - distance_influence**-1) * distance**-2 * grad_distance
        else:
            grad_u_rep = np.array([[0.0],[0.0]])
            # grad_u_rep = np.array([[math.nan],[math.nan]])

        return grad_u_rep.reshape((2,1))

class Attractive:
    """ Attractive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        x_eval = x_eval.reshape((2,1))
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if shape == 'conic':
            expo = 1
        else:
            expo = 2
        u_attr = np.linalg.norm(x_eval - x_goal)**expo

        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient
        is given by the formula If  potential['shape'] is equal to 'conic', use p=1; if it is
        equal to 'quadratic', use p=2.
        """
        x_eval = x_eval.reshape((2,1))
        x_goal = self.potential['x_goal'].reshape((2,1))
        shape = self.potential['shape']
        if shape == 'conic':
            expo = 1
        else:
            expo = 2
        grad_u_attr = expo * np.linalg.norm(x_eval - x_goal)**(expo-2) * (x_eval - x_goal)

        return grad_u_attr.reshape((2,1))

class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world  # SphereWorld object
        self.potential = potential  # dict

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a 
        is given by the variable potential.repulsiveWeight
        """
        alpha = self.potential['repulsive_weight']

        # Attractive potential
        attr = Attractive(self.potential)
        u_att = attr.eval(x_eval.reshape((2,1)))

        sphere_world = self.world

        # Repulsive potential
        u_rep = 0
        for sphere in sphere_world.world:
            rep = RepulsiveSphere(sphere)
            u_rep += rep.eval(x_eval.reshape((2,1)))

        # Total potential
        u_eval = u_att + alpha * u_rep

        return u_eval

    def grad(self, x_eval):
        """
        Compute the gradient of the total potential,  U=U_ attr+a*U_rep,i, where a is given by
        the variable  potential.repulsiveWeight
        """
        alpha = self.potential['repulsive_weight']

        # Gradient of attractive potential
        attr = Attractive(self.potential)
        grad_u_attr = attr.grad(x_eval.reshape((2,1)))

        sphere_list = self.world

        # Gradient of repulsive potential
        grad_u_rep = np.zeros((2,1))
        for sphere in sphere_list.world:
            rep = RepulsiveSphere(sphere)
            rep_grad = rep.grad(x_eval.reshape((2,1)))
            grad_u_rep += rep_grad

        # Total gradient
        grad_u_eval = grad_u_attr + alpha * grad_u_rep

        return grad_u_eval.reshape((2,1))

class Planner:
    """
    A class implementing a generic potential planner and plot the results.
    """
    def __init__(self, function, control, epsilon, nb_steps):
        """
        Save the arguments to internal attributes
        """
        self.function = function    # function handle for computing value of potential function
        self.control = control  # function handle for computing direction planner should move
        self.epsilon = epsilon  # value for step size
        self.nb_steps = nb_steps # total number of steps

    def run(self, x_start, axes=None):

        """
        This function uses a given function (given by  control) to implement a
        generic potential-based planner with step size  epsilon, and evaluates
        the cost along the returned path. The planner must stop when either the
        number of steps given by  nb_steps is reached, or when the norm of the
        vector given by  control is less than 5 10^-3 (equivalently,  5e-3).
        """
        if axes is None:
            axes = plt.gca()

        x_path = np.full((2,self.nb_steps), np.nan)
        u_path = np.full((1,self.nb_steps), np.nan)

        threshold = 5e-3
        x_path[:,0] = x_start.reshape((2,))
        # axes.scatter(x_start[0], x_start[1], color='b', marker='*')

        # Compute trajectory generated by planner, x_path
        # Compute the potential at each point on the path
        for k in range(self.nb_steps-1):
            control_current = self.control(x_path[:,k].reshape((2,1)))
            if np.linalg.norm(control_current) > threshold:
                x_k_1 = x_path[:,k].reshape((2,1)) + (self.epsilon * control_current)
                x_path[:, k+1] = x_k_1.reshape((2,))
                u_path[:, k] = self.function(x_path[:, k].reshape((2,1)))
        # Populate last value value of u_path
        u_path[:, self.nb_steps-1] = self.function(x_path[:, self.nb_steps-1].reshape((2,1)))

        return x_path, u_path

    def run_plot(self):
        """
        This function performs the following steps:
         - Loads the problem data via an object world of class SphereWorld.
         - Uses the function Sphereworld.plot to plot the world in a first
        figure.
         - it:grad-handleCalls the method run. The function needs to be called
        five times, using each one of the initial locations given in  x_start
        (also provided in !70!DarkSeaGreen2 sphereworld.mat).
         - it:plot-plan After each call, plot the resulting trajectory
        superimposed to the world in the first subplot; in a second subplot,
        show  u_path (using the same color and using the  semilogy command).
        """
        sphere_world = SphereWorld()

        x_start = sphere_world.x_start
        nb_starts = x_start.shape[1]
        fig, axes = plt.subplots(ncols=3)
        for sub_axes in axes[0:2]:
            sub_axes.set_aspect('equal', adjustable='datalim')
            sphere_world.plot(sub_axes)
        axes[1].set_xlim(6.5, 8.5)

        x_steps = np.arange(self.nb_steps)
        for start in range(0, nb_starts):
            x_start = sphere_world.x_start[:, [start]]
            x_path, u_path = self.run(x_start)
            for sub_axes in axes[0:2]:
                sub_axes.plot(x_path[0, :], x_path[1, :])
            axes[-1].plot(x_steps, u_path.T)

        axes[-1].set_xlim(0, self.nb_steps - 1)
        axes[-1].set_ylim(0, 50)
        axes[-1].set_autoscale_on(False)
        axes[-1].set_title('Potential Function')

        # Make figure with good aspect ratio
        # fig.set_size_inches([8, 2])

class Clfcbf_Control:
    """
    A class implementing a CLF-CBF-based control framework.
    """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes, and create an attribute
        attractive with an object of class  Attractive using the argument
        potential.
        """
        self.world = world  # Sphereworld object
        self.potential = potential
        self.attractive = Attractive(potential)

    def function(self, x_eval):
        """
        Evaluate the CLF (i.e.,  self.attractive.eval()) at the given input.
        """
        return self.attractive.eval(x_eval)

    def control(self, x_eval):
        """
        Compute u^* according to      (  eq:clfcbf-qp  ).
        """
        x_eval = x_eval.reshape((2,1))
        c_h = self.potential['repulsive_weight']

        nb_spheres = len(self.world.world)

        a_barrier = np.zeros((nb_spheres,2))
        b_barrier = np.zeros((nb_spheres, 1))

        u_ref = - self.attractive.grad(x_eval)

        idx = 0
        for sphere in self.world.world:
            dist = sphere.distance(x_eval)
            grad_dist = sphere.distance_grad(x_eval)
            a_barrier[idx,:] = - np.transpose(grad_dist).reshape((2,))
            b_barrier[idx,0] = - c_h * dist
            idx += 1

        u_opt = qp.qp_supervisor(a_barrier, b_barrier, u_ref)

        return u_opt
