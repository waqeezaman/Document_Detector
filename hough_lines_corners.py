from math import sin, cos, atan
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from processors import Opener, Closer, EdgeDetector
from sklearn.cluster import KMeans
from itertools import combinations

class HoughLineCornerDetector:
  ...
  
  def _get_intersections(self, lines, width, height):
    """Finds the intersections between groups of lines."""
    # lines = self._lines
    intersections = []
    group_lines = combinations(range(len(lines)), 2)
    x_in_range = lambda x: 0 <= x <= width##self._image.shape[1]
    y_in_range = lambda y: 0 <= y <= height##self._image.shape[0]

    for i, j in group_lines:
      line_i, line_j = lines[i][0], lines[j][0]

      if 80.0 < self._get_angle_between_lines(line_i, line_j) < 100.0:
          int_point = self._intersection(line_i, line_j)

          if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]): 
              intersections.append(int_point)

    # if self.output_process: self._draw_intersections(intersections)

    return intersections
      
  def _get_angle_between_lines(self, line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(atan(abs(m2-m1) / (1 + m2 * m1))) * (180 / np.pi)

    
  def _intersection(self, line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
      [np.cos(theta1), np.sin(theta1)],
      [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]