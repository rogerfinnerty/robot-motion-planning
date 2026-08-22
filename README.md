# Robot Motion Planning

## Contents

- ### [Section 1](Section1/): Geometric Foundations - Two-Link Manipulator Polygons, Collision Tests, Visibility Checks, Priority Queue
  This section establishes the geometric objects and relationships that will be necessary later on in the course.

Two-link manipulator polygons:

![polygons](Section1/assets/robot_polygons.png)

Demonstrating the "visibility" of various test points with respect to the vertices of the "solid" and "hollow" versions of the two-link segments.

  <p float="left">
    <img src="Section1/assets/polygon_visibility_test_hollow_1.png" width="49%" alt="Workspace view"/>
    <img src="Section1/assets/polygon_visibility_test_solid_2.png" width="49%" alt="Configuration view"/>
  </p>

- ### [Section 2](Section2/): 3D Rotation, Torus Manifold

  This section presents the torus as a 2D manifold embedded in 3D space that models the configuration space of the two‑link manipulator, whose configurations are specified by the two joint angles.

  ![plot_torus output](Section2/assets/torus_curves.png)

  A line in the torus manifold space corresponds to a path for the two-link manipulator to follow.

  ![torus_twolink_plot_jacobian curve 2](Section2/assets/torus_twolink_plot_jacobian_curve2.gif)

- ### [Section 3](Section3/): Potential-Based Methods, CLF-CBF Framework

  The section introduces the `SphereWorld` environment, which consists of three solid "sphere" objects contained in a hollow "sphere". The CLF-CBF-QP framework is implemented for safe path planning between various start and goal locations.

  ![sphere-world-clf-cbf](Section3/assets/planner_path_animation_goal_0.gif)

The same control framework is used to actuate the two-link manipulator from a start location to a goal location. The repulsive potential of the end-effector coordinate is shown next the animation.

![two-link-clf-cbf1](Section3/assets/twolink_animation_start0.gif)

- ### [Section 4](Section4/): A\*

Use the A\* solve graphs of increasing complexity, ultimately a graph representation of the SphereWorld environment. Also use A\* planning to actuate the two-link manipulator to a goal location while avoiding obstacle points.

![sphereworld graph goal 0](Section4/assets/sphereworld_20cells_goal1.gif)

  <p float="left">
    <img src="Section4/assets/twolink_solved.gif" width="80%" alt="Workspace view"/>
    <img src="Section4/assets/twolink_configuration_space.png" width="70%" alt="Configuration view"/>
  </p>
