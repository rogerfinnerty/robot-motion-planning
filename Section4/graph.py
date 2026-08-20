"""
Classes and utility functions for working with graphs (plotting, search, initialization, etc.)
"""
import math
import pickle
from math import pi
from numbers import Number

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy import io as scio
import geometry
import queue
import imageio

class PriorityQueue:
    """ Implements a priority queue """
    def __init__(self):
        """
        Initializes the internal attribute  queue to be an empty list.
        """
        self.queue_list = []

    def check(self):
        """
        Check that the internal representation is a list of (key,value) pairs,
        where value is numerical
        """
        is_valid = True
        for pair in self.queue_list:
            if len(pair) != 2:
                is_valid = False
                break
            if not isinstance(pair[1], Number):
                is_valid = False
                break
        return is_valid

    def insert(self, key, cost):
        """
        Add an element to the queue.
        """
        self.queue_list.append( (key, cost) )

    def min_extract(self):
        """
        Extract the element with minimum cost from the queue.
        """
        if len(self.queue_list) == 0:
            key = None
            cost = None
        else:
            min_idx = 0
            for i in range(1, len(self.queue_list) ):
                if self.queue_list[i][1] < self.queue_list[min_idx][1]:
                    min_idx = i
            key, cost = self.queue_list[min_idx]

        return key, cost

    def is_member(self, key):
        """
        Check whether an element with a given key is in the queue or not.
        """
        flag = False
        for _, item in enumerate(self.queue_list):
            if item[0] == key:
                flag = True
        return flag

    def print_queue(self):
        """
        Print contents of the queue 
        """
        if len(self.queue_list) == 0:
            print("Queue is empty.")
        else:
            print("Current queue: ")
            for item in self.queue_list:
                print(f"Key: {item[0]}, Cost: {item[1]}.")

    def display_descending(self):
        """
        Displays the elements of the queue in descending order by cost. 
        """
        # First create a duplicate priority queue so that we can delete elements as they are printed
        queue_copy = PriorityQueue()
        for _, item in enumerate(self.queue_list):
            key = item[0]
            cost = item[1]
            queue_copy.insert(key, cost)

        # Modified version of extract_min function to find the index
        # of the maximum cost item, print that item,
        # then remove it from the list
        for _ in range(len(self.queue_list)):
            # Determine index of maximum value
            max_idx = 0
            for j in range(1, len(queue_copy.queue_list) ):
                if queue_copy.queue_list[j][1] > queue_copy.queue_list[max_idx][1]:
                    max_idx = j
            # Display the maximum key, value pair
            max_key = queue_copy.queue_list[max_idx][0]
            max_val = queue_copy.queue_list[max_idx][1]
            print(f"Key: {max_key},Cost: {max_val}")
            # Remove the maximum pair from the list
            queue_copy.queue_list.pop(max_idx)
 
    def remove(self, key_remove):
        """
        Removes an element from the priority queue
        """
        self.queue_list = [(key, cost) for key, cost in self.queue_list if key != key_remove]

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
        data = scio.loadmat('data/sphereworld.mat')

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
    """ Repulsive potential for a sphere """
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

def plot_arrows_from_list(arrow_list, scale=1.0, color=(0., 0., 0.)):
    """
    Plot arrows from a list of pairs of base points and displacements
    """
    x_edges, v_edges = [np.hstack(x) for x in zip(*arrow_list)]
    plt.quiver(x_edges[0, :],
               x_edges[1, :],
               v_edges[0, :],
               v_edges[1, :],
               angles='xy',
               scale_units='xy',
               scale=scale,
               color=color)


def plot_text(coord, str_label, color=(1., 1., 1.)):
    """
    Wrap plt.text to get a consistent look
    """
    plt.text(coord[0].item(),
             coord[1].item(),
             str_label,
             ha="center",
             va="center",
             fontsize='xx-small',
             bbox=dict(boxstyle="round", fc=color, ec=None))


class Graph:
    """
    A class collecting a graph_vector data structure and all the functions that operate on a graph.
    """
    def __init__(self, graph_vector):
        """
        Stores the arguments as internal attributes.
        """
        self.graph_vector = graph_vector
        for node in self.graph_vector:
            if not 'g' in node:
                node['g'] = None
            if not 'backpointer' in node:
                node['backpointer'] = None

    def _apply_neighbor_function(self, func):
        """
        Apply a function on each node and chain the result
        """
        list_of_lists = [func(n) for n in self.graph_vector]
        return [e for l in list_of_lists for e in l]

    def _neighbor_weights_with_positions(self, n_current):
        """
        Get all weights and where to display them
        """
        x_current = n_current['x']
        return [
            (weight_neighbor,
             self.graph_vector[idx_neighbor]['x'] * 0.25 + x_current * 0.75)
            for (weight_neighbor, idx_neighbor
                 ) in zip(n_current['neighbors_cost'], n_current['neighbors'])
        ]

    def _neighbor_displacements(self, n_current):
        """
        Get all displacements with respect to the neighbors for a given node
        """
        x_current = n_current['x']
        return [(x_current, self.graph_vector[idx_neighbor]['x'] - x_current)
                for idx_neighbor in n_current['neighbors']]

    def _neighbor_backpointers(self, n_current):
        """
        Get coordinates for backpointer arrows
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [
                (x_current,
                 0.5 * (self.graph_vector[idx_backpointer]['x'] - x_current))
            ]
        else:
            arrow = []
        return arrow

    def _neighbor_backpointers_cost(self, n_current):
        """
        Get value and coordinates for backpointer costs
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [(n_current['g'],
                      self.graph_vector[idx_backpointer]['x'] * 0.25 +
                      x_current * 0.75)]
        else:
            arrow = []
        return arrow

    def has_backpointers(self):
        """
        Return True if self.graph_vector has a "backpointer" field
        """
        return self.graph_vector is not None and len(
            self.graph_vector) > 0 and 'backpointer' in self.graph_vector[0]

    def plot(self,
             flag_edges=True,
             flag_labels=False,
             flag_edge_weights=False,
             flag_backpointers=False,
             flag_backpointers_cost=False,
             flag_heuristic=False,
             node_lists=None,
             idx_closed=None,
             idx_goal=None,
             idx_best=None):
        """
        The function plots the contents of the graph described by the
        graph_vector structure, alongside other related, optional data.
        """

        if flag_edges:
            displacement_list = self._apply_neighbor_function(
                self._neighbor_displacements)
            plot_arrows_from_list(displacement_list, scale=1.05)

        if flag_labels:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                plot_text(x_current, str(idx))

        if idx_closed is not None:
            for idx in idx_closed:
                x_current = self.graph_vector[idx]['x']
                plt.scatter(x_current[0],
                            x_current[1],
                            marker='s',
                            color=(0., 0., 1.))

        if idx_goal is not None:
            x_goal = self.graph_vector[idx_goal]['x']
            plt.plot(x_goal[0, :],
                     x_goal[1, :],
                     marker='d',
                     markersize=10,
                     color=(.8, .1, .1))

        if idx_best is not None:
            x_best = self.graph_vector[idx_best]['x']
            plt.plot(x_best[0, :],
                     x_best[1, :],
                     marker='d',
                     markersize=10,
                     color=(0., 1., 0.))

        if flag_heuristic and idx_goal is not None:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                h_current = self.heuristic(idx, idx_goal)
                plot_text(x_current, f'h={h_current:.2f}', color=(.8, 1., .8))
                if flag_heuristic and idx_goal is not None:
                    idx_backpointer = n_current.get('backpointer', None)
                    if idx_backpointer is not None:
                        cost = n_current['g'] + h_current
                        offset = np.array([[0], [.15]])
                        plot_text(x_current + offset,
                                  f'f={cost:.2f}',
                                  color=(.8, 1., .8))

        if flag_edge_weights:
            weight_list = self._apply_neighbor_function(
                self._neighbor_weights_with_positions)
            for (weight, coord) in weight_list:
                plot_text(coord, str(weight), color=(.8, .8, 1.))

        if flag_backpointers and self.has_backpointers():
            backpointer_arrow_list = self._apply_neighbor_function(
                self._neighbor_backpointers)
            plot_arrows_from_list(backpointer_arrow_list,
                                  scale=1.05,
                                  color=(0.1, .8, 0.1))

        if flag_backpointers_cost and self.has_backpointers:
            backpointer_cost_list = self._apply_neighbor_function(
                self._neighbor_backpointers_cost)
            offset = np.array([[0], [-.15]])
            for (cost, coord) in backpointer_cost_list:
                plot_text(coord + offset, f'g={cost:.2f}', color=(.8, 1., .8))

        if node_lists is not None:
            if not isinstance(node_lists[0], list):
                node_lists = [node_lists]
            markers = ['d', 'o', 's', '*', 'h', '^', '8']
            for i, lst in enumerate(node_lists):
                x_list = [self.graph_vector[e]['x'] for e in lst]
                coords = np.hstack(x_list)
                plt.plot(
                    coords[0, :],
                    coords[1, :],
                    markers[i % len(markers)],
                    markersize=10,
                )

    def nearest_neighbors(self, x_query, k_nearest):
        """
        Returns the k nearest neighbors in the graph for a given point.
        """
        x_graph = np.hstack([n['x'] for n in self.graph_vector])
        distances_squared = np.sum((x_graph - x_query)**2, 0)
        idx = np.argpartition(distances_squared, k_nearest)
        return idx[:k_nearest]

    def heuristic(self, idx_x, idx_goal):
        """
        Computes the heuristic  h given by the Euclidean distance between the nodes with indexes
        idx_x and  idx_goal.
        """
        node = self.graph_vector[idx_x]['x']
        neighbor = self.graph_vector[idx_goal]['x']
        h_val = np.linalg.norm(node - neighbor)

        return h_val

    def get_expand_list(self, idx_n_best, idx_closed):
        """
        Finds the neighbors of element  idx_n_best that are not in  idx_closed (line   in Algorithm~
        ).
        """
        # List of neighbors
        neighbors = self.graph_vector[idx_n_best]['neighbors']

        # Convert lists of neighbors and closed
        idx_expand = list(set(neighbors) - set(idx_closed))
        idx_expand.reverse()

        return idx_expand

    def expand_element(self, idx_n_best, idx_x, idx_goal, pq_open):
        """
        This function expands the vertex with index idx_x (which is a neighbor of the one with
        index idx_n_best) and returns the updated versions of graph_vector and pq_open.
        In pq_open, the index of the vertex is stored as the key and the f(n) is used as cost; 
        f(n) varies based on type of graph search method.
        """
        # Extract nodes of interest
        n_best = self.graph_vector[idx_n_best]
        node = self.graph_vector[idx_x]

        # Find index of idx_x in 'neighbors' list of n_best
        x_index = n_best['neighbors'].index(idx_x)

        # If node is not in the queue
        if not pq_open.is_member(idx_x):
            # Set backpointer cost
            node['g'] = n_best['g'] + n_best['neighbors_cost'][x_index]
            # Set backpointer
            node['backpointer'] = idx_n_best
            # Compute heuristic
            h = self.heuristic(idx_x, idx_goal)
            # Compute estimated cost
            f = h + node['g']
            # Add x to priority queue
            pq_open.insert(idx_x, f)
        # Else if a better path to x has been found
        elif (n_best['g'] + n_best['neighbors_cost'][x_index]) < node['g']:
            node['g'] = n_best['g'] + n_best['neighbors_cost'][x_index]
            node['backpointer'] = idx_n_best

        # If the element is in the queue and there is not a better path to x, do nothing

        return pq_open

    def path(self, idx_start, idx_goal):
        """
        This function follows the backpointers from the node with index  idx_goal in  graph_vector
        to the one with index  idx_start node, and returns the  coordinates (not indexes) of the
        sequence of traversed elements.
        """
        x_path = []
        idx_x = idx_goal
        # Starting with goal location, append the coordinates of each node,
        # follow backpointers until reaching start location
        while idx_x is not None:
            x_path.append(self.graph_vector[idx_x]['x'])
            idx_x = self.graph_vector[idx_x]['backpointer']
        # Reverse the list so that the start location is first
        x_path.reverse()
        # Convert list to a 2D array
        x_val = np.array([x[0].item() for x in x_path])
        y_val = np.array([x[1].item() for x in x_path])
        x_path = np.vstack((x_val, y_val))
        return x_path

    def search(self, idx_start, idx_goal) -> np.ndarray:
        """
        Implements the  A^* algorithm, as described by the pseudo-code in Algorithm~ .
        """
        # Initialize priority queue O, of opened vertices
        #pq_open = queue.PriorityQueue()
        pq_open = PriorityQueue()
        # Add start node, with cost=0
        pq_open.insert(idx_start, 0)

        # Set start node backpointer cost to 0
        self.graph_vector[idx_start]['g'] = 0
        # Set start node backpointer to empty
        self.graph_vector[idx_start]['backpointer'] = None

        # Initialize list of closed nodes, C
        closed = []

        its = 0 # iteration count
        # Repeat until queue (O) is empty
        while len(pq_open.queue_list) > 0:
            # Extract n_best from queue (node with lowest cost)
            n_best, _  = pq_open.min_extract()
            # Remove n_best from queue, add to C
            pq_open.remove(n_best)
            closed.append(n_best)
            # Version 1
            if n_best == idx_goal:
                x_path = self.path(idx_start, idx_goal)
                return x_path

            # Get list S (list of nodes expanded from n_best that are not closed)
            idx_expand = self.get_expand_list(n_best, closed)

            # Plotting (for debugging)
            # self.plot(flag_labels=False,idx_closed=closed,idx_goal=idx_goal,
            #           idx_best=n_best,flag_backpointers=False)

            # Expand each node in S that is not already closed
            for idx_x in idx_expand:
                pq_open = self.expand_element(n_best, idx_x, idx_goal, pq_open)
            its += 1
        
        # No path found, return empty array
        return np.array([]).reshape(2, 0)

    def search_start_goal(self, x_start, x_goal) -> np.ndarray:
        """
        This function performs the following operations:
         - Identifies the two indexes  idx_start,  idx_goal in  graph.graph_vector that are closest
        to  x_start and  x_goal (using Graph.nearestNeighbors twice, see Question~ -nearest).
         - Calls Graph.search to find a feasible sequence of points  x_path from  idx_start to
        idx_goal.
         - Appends  x_start and  x_goal, respectively, to the beginning and the end of the array
        x_path.
        """
        # Find indices in graph_vector that are closest to x_start, x_goal
        idx_start = int( self.nearest_neighbors(x_start, 1) )
        idx_goal = int( self.nearest_neighbors(x_goal, 1) )

        # Find path from idx_start to idx_goal
        x_path = self.search(idx_start, idx_goal)

        # Append x_start, x_goal to beginning/end of x_path
        # x_path = np.concatenate((x_start, x_path), axis=1)
        # x_path = np.concatenate((x_path, x_goal), axis=1)

        return x_path

class SphereWorldGraph:
    """
    A discretized version of the SphereWorld from Homework 3 with the addition of a search
    function.
    """
    def __init__(self, nb_cells):
        """
        The function performs the following steps:
         - Instantiate an object of the class  SphereWorld from Homework 3 to load the contents of
        the file sphereworld.mat. Store the object as the internal attribute  sphereworld.d
         - Initializes an object  grid from the class  Grid initialized with arrays  xx_grid and
        yy_grid, each one containing  nb_cells values linearly spaced values from  -10 to  10.
         - Use the method grid.eval to obtain a matrix in the format expected by grid2graph in
        Question~ , i.e., with a  true if the space is free, and a  false if the space is occupied
        by a sphere at the corresponding coordinates. The quickest way to achieve this is to
        manipulate the output of Total.eval (for checking collisions with the spheres) while using
        it in conjunction with grid.eval (to evaluate the collisions along all the points on the
        grid); note that the choice of the attractive potential here does not matter.
         - Call grid2graph.
         - Store the resulting  graph object as an internal attribute.
        """
        self.world = SphereWorld()
        xx_grid = np.linspace(-10, 10, nb_cells)
        grid = geometry.Grid(xx_grid, xx_grid)


        potential = {
            'x_goal': self.world.x_goal[:,0],
            'shape': 'conic',
            'repulsive_weight': 0.1
        }

        def is_occupied(x_eval):
            total = Total(self.world, potential)
            return total.eval(x_eval) < 100

        grid.eval(is_occupied)

        self.graph = grid2graph( grid )

    def plot(self):
        """
        Plots the graph attribute
        """
        self.graph.plot()

    def run_plot(self, nb_cells, save_img=False):
        """
        - Load the variables  x_start,  x_goal from the internal attribute  sphereworld.
        homework4_sphereworldPlot
        """
        # Extract x_start, x_goal variables
        x_start = self.world.x_start
        x_goal = self.world.x_goal

        if save_img:
            import os
            os.makedirs('assets', exist_ok=True)

        for goal_idx in range(x_goal.shape[1]):
            goal = x_goal[:, goal_idx].reshape((2, 1))
            sphere_graph = SphereWorldGraph(nb_cells)
            fig, ax = plt.subplots(figsize=(6, 6))
            sphere_graph.graph.plot(flag_edges=True, flag_labels=False, flag_backpointers=False)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)
            ax.set_aspect('equal', adjustable='box')

            canvas = FigureCanvas(fig)
            images = []
            colors = plt.cm.tab10(np.linspace(0, 1, x_start.shape[1]))

            for start_idx in range(x_start.shape[1]):
                start = x_start[:, start_idx].reshape((2, 1))
                x_path = sphere_graph.graph.search_start_goal(start, goal)
                color = colors[start_idx]

                line, = ax.plot([], [], '-', linewidth=3, color=color)
                point, = ax.plot([], [], 'o', markersize=5, color=color)

                for k in range(1, x_path.shape[1] + 1):
                    line.set_data(x_path[0, :k], x_path[1, :k])
                    point.set_data([x_path[0, k - 1]], [x_path[1, k - 1]])
                    fig.canvas.draw()
                    w, h = canvas.get_width_height()
                    buf = canvas.tostring_argb()
                    arr = np.frombuffer(buf, dtype='uint8').reshape((h, w, 4))
                    img = arr[:, :, 1:4].copy()
                    images.append(img)
                    plt.pause(0.05)

            if save_img and images:
                outfile = f'assets/sphereworld_{nb_cells}cells_goal{goal_idx}.gif'
                imageio.mimsave(outfile, images, duration=0.05, loop=0)
                print(f"Saved sphereworld animation to {outfile}")

            plt.close(fig)


def graph_test_data_load(variable_name):
    """
    Loads data from the file graph_test_data.pkl.
    """
    with open('data/graph_test_data.pkl', 'rb') as fid:
        saved_data = pickle.load(fid)
    return saved_data[variable_name]


def graph_test_data_plot(save_img=False):
    """
    Plot two solved graphs
    """

    graph = Graph(graph_test_data_load('graphVector_solved'))
    plt.figure()
    graph.plot(flag_heuristic=True,
               flag_backpointers=True,
               flag_backpointers_cost=True,
                 idx_goal=1)
    if save_img:
        plt.savefig('assets/graph_test_data_plot.png')
        print(f"Saved graph to assets/graph_test_data_plot.png")

    graph = Graph(graph_test_data_load('graphVectorMedium_solved'))
    plt.figure()
    graph.plot(flag_heuristic=True,
               flag_backpointers=True,
               flag_backpointers_cost=True,
                idx_goal=14)
    if save_img:
        plt.savefig('assets/graph_test_data_plot_medium.png')
        print(f"Saved graph to assets/graph_test_data_plot_medium.png")



def grid2graph(grid):
    """
    The function returns a  Graph object described by the inputs. See Figure~  for an example of the
    expected inputs and outputs.
    """

    # Make sure values in F are logicals
    fun_evalued = np.vectorize(bool)(grid.fun_evalued)

    # Get number of columns, rows, and nodes
    nb_xx, nb_yy = fun_evalued.shape
    nb_nodes = np.sum(fun_evalued)

    # Get indeces of non-zero entries, and assign a progressive number to each
    idx_xx, idx_yy = np.where(fun_evalued)
    idx_assignment = range(0, nb_nodes)

    # Lookup table from xx,yy element to assigned index (-1 means not assigned)
    idx_lookup = -1 * np.ones(fun_evalued.shape, 'int')
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        idx_lookup[i_xx, i_yy] = i_assigned

    def grid2graph_neighbors(idx_xx, idx_yy):
        """
        Finds the neighbors of a given element
        """

        displacements = [(idx_xx + dx, idx_yy + dy) for dx in [-1, 0, 1]
                         for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        neighbors = []
        for i_xx, i_yy in displacements:
            if 0 <= i_xx < nb_xx and 0 <= i_yy < nb_yy and idx_lookup[
                    i_xx, i_yy] >= 0:
                neighbors.append(idx_lookup[i_xx, i_yy].item())

        return neighbors

    # Create graph_vector data structure and populate 'x' and 'neighbors' fields
    graph_vector = [None] * nb_nodes
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        x_current = np.array([[grid.xx_grid[i_xx]], [grid.yy_grid[i_yy]]])
        neighbors = grid2graph_neighbors(i_xx, i_yy)
        graph_vector[i_assigned] = {'x': x_current, 'neighbors': neighbors}

    # Populate the 'neighbors_cost' field
    # Cannot be done in the loop above because not all 'x' fields would be filled
    for idx, n_current in enumerate(graph_vector):
        x_current = n_current['x']

        if len(n_current['neighbors']) > 0:
            x_neighbors = np.hstack(
                [graph_vector[idx]['x'] for idx in n_current['neighbors']])
            neighbors_cost_np = np.sum((x_neighbors - x_current)**2, 0)
            graph_vector[idx]['neighbors_cost'] = list(neighbors_cost_np)
        else:
            graph_vector[idx]['neighbors_cost'] = []

    return Graph(graph_vector)


def test_nearest_neighbors():
    """
    Tests Graph.nearest_neighbors by picking a random point and finding the 3 nearest neighbors
    in graphVectorMedium
    """
    graph = Graph(graph_load_test_data('graphVectorMedium'))
    x_query = np.array([[5], [4]]) * np.random.rand(2, 1)
    idx_neighbors = graph.nearest_neighbors(x_query, 3)
    graph.plot(node_lists=idx_neighbors)
    plt.scatter(x_query[[0]], x_query[[1]])


def test_grid2graph():
    """
    Tests grid2graph() by creating an arbitrary function returning bools
    """
    xx_grid = np.linspace(0, 2 * pi, 40)
    yy_grid = np.linspace(0, pi, 20)
    func = lambda x: (x[[1]] > pi / 2 or np.linalg.norm(x - np.ones(
        (2, 1))) < 0.75) and not np.linalg.norm(x - np.array([[4], [2.5]])
                                                ) < 0.5
    grid = geometry.Grid(xx_grid, yy_grid)
    grid.eval(func)
    graph = grid2graph(grid)
    graph.plot()

def test_sphereworld_graph_plot_solved(nb_cells=20, save_img=False):
    """
    Tests SphereWorldGraph.run_plot() by plotting the solved paths for all start/goal pairs
    """
    sphereworld_graph = SphereWorldGraph(nb_cells=nb_cells)
    sphereworld_graph.run_plot(nb_cells=nb_cells, save_img=save_img)

if __name__ == "__main__":
    # test_grid2graph()

    test_sphereworld_graph_plot_solved(nb_cells=20, save_img=True)
    plt.show()


