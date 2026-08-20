"""
Combine the classes below with the file me570_robot.py from previous assignments
"""
import math
import numpy as np
from scipy import io as scio
import matplotlib.pyplot as plt
import geometry as geometry
import graph as graph
import imageio
import tempfile
import shutil
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple

def polygons_generate() -> Tuple[geometry.Polygon, geometry.Polygon]:
    """
    Generate the polygons to be used for the two-link manipulator
    """
    vertices1 = np.array([[0, 5], [-1.11, -0.511]])
    vertices1 = polygons_add_x_reflection(vertices1)
    vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                          [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
    vertices2 = polygons_add_x_reflection(vertices2)
    return (geometry.Polygon(vertices1), geometry.Polygon(vertices2))

def polygons_add_x_reflection(vertices: np.ndarray) -> np.ndarray:
    """
    Given a sequence of vertices, adds other vertices by reflection
    along the x axis
    """
    vertices = np.hstack([vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
    return vertices


class TwoLink:
    """ See description from previous homework assignments. """
    def jacobian_matrix(self, theta: np.ndarray) -> np.ndarray:
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

        return j_theta

    def kinematic_map(self, theta: np.ndarray) -> Tuple[np.ndarray, geometry.Polygon, geometry.Polygon]:
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

    def plot(self, theta: np.ndarray, color: str, axes=None) -> None:
        """
        This function should use TwoLink.kinematic_map from the 
        previous question together with the method Polygon.plot 
        from Homework 1 to plot the manipulator.
        """
        [_, polygon1_transf, polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(axes=axes, color=color)
        polygon2_transf.plot(axes=axes, color=color)

    def animate(self, theta: np.ndarray, outname: str, axes=None) -> None:
        """
        Draw the two-link manipulator for each column in theta 
        with a small pause between each drawing operation
        """
        # Parameters
        theta_steps = theta.shape[1]
        frame_step = 5
        pause = 1.0

        # Load obstacle points once
        test_data = scio.loadmat('data/twolink_testData.mat')
        obstacle_points = test_data['obstaclePoints']

        # Create temporary directory for frames
        tmpdir = tempfile.mkdtemp(prefix='twolink_frames_')
        frame_paths = []

        try:
            # Determine plotting limits based on obstacle points and link reach
            xp = obstacle_points[0, :]
            yp = obstacle_points[1, :]
            pad = 6
            xmin, xmax = xp.min() - pad, xp.max() + pad
            ymin, ymax = yp.min() - pad, yp.max() + pad

            for i_theta in range(0, theta_steps, frame_step):
                if axes is None:
                    plt.clf()
                    plt.plot(obstacle_points[0, :], obstacle_points[1, :], 'r*')
                    self.plot(theta[:, [i_theta]], 'k')
                    plt.axis('equal')
                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)

                    frame_path = os.path.join(tmpdir, f'frame_{i_theta:04d}.png')
                    plt.savefig(frame_path, bbox_inches='tight')
                    frame_paths.append(frame_path)
                    plt.pause(pause)
                else:
                    # Draw on provided axes and capture via its figure canvas
                    fig = axes.figure
                    canvas = FigureCanvas(fig)

                    pre_patches = list(axes.patches)
                    pre_lines = list(axes.lines)
                    pre_collections = list(axes.collections)
                    pre_artists = list(axes.artists)

                    # Plot manipulator onto axes
                    self.plot(theta[:, [i_theta]], 'k', axes=axes)

                    # Draw and capture
                    canvas.draw()
                    frame_path = os.path.join(tmpdir, f'frame_{i_theta:04d}.png')
                    w, h = canvas.get_width_height()
                    buf = canvas.tostring_argb()
                    arr = np.frombuffer(buf, dtype='uint8').reshape((h, w, 4))
                    img = arr[:, :, 1:4].copy()

                    # Write PNG
                    imageio.v2.imwrite(frame_path, img)
                    frame_paths.append(frame_path)

                    # Remove added artists so the world remains unchanged
                    added = [p for p in axes.patches if p not in pre_patches]
                    added += [l for l in axes.lines if l not in pre_lines]
                    added += [c for c in axes.collections if c not in pre_collections]
                    added += [a for a in axes.artists if a not in pre_artists]
                    for art in added:
                        try:
                            art.remove()
                        except Exception:
                            pass
                    plt.pause(pause)

            # Assemble GIF
            images = []
            for fp in sorted(frame_paths):
                images.append(imageio.v2.imread(fp))
            if images:
                imageio.mimsave(outname, images, duration=pause, loop=0)
                print(f"Animation saved to {outname}")
        finally:
            # Clean up temporary frames
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

class TwoLinkGraph:
    """
    A class for finding a path for the two-link manipulator among given obstacle points using a grid
    discretization and  A^*.
    """
    # def load_free_space_graph(self):
    def __init__(self) -> None:
        """
        The function performs the following steps
         - Calls the method load_free_space_grid.
         - Calls grid2graph.
         - Stores the resulting  graph object of class  Grid as an internal attribute.
        """
        grid = load_free_space_grid()
        self.graph = graph.grid2graph(grid)


    def plot(self) -> None:
        """
        Use the method Graph.plot to visualize the contents of the attribute  graph.
        """
        self.graph.plot()

    def search_start_goal(self, theta_start: np.ndarray, theta_goal: np.ndarray) -> np.ndarray:
        """
        Use the method Graph.search to search a path in the graph stored in  graph.

        """
        
        theta_path = self.graph.search_start_goal(theta_start, theta_goal)
        return theta_path

def load_free_space_grid() -> geometry.Grid:
    """
Loads the contents of the file ! twolink_freeSpace_data.mat
    """
    test_data = scio.loadmat('data/twolink_freeSpace_data.mat')
    test_data = test_data['grid'][0][0]
    grid = geometry.Grid(test_data[0], test_data[1])
    grid.fun_evalued = test_data[2]
    return grid
