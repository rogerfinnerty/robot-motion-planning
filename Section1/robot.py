"""
Representation of a simple robot used in the assignments
"""

import numpy as np

import geometry


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


polygons = polygons_generate()

if __name__ == "__main__":
    # Plot both polygons and save the figure
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,1,1)
    polygons[0].plot()
    plt.subplot(2,1,2)
    polygons[1].plot()

    plt.savefig("assets/robot_polygons.png", dpi=300)
