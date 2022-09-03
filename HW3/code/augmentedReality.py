import numpy as np
import cv2
import matplotlib.pyplot as plt

from planarH import ransacH, computeH
from BRIEF import briefLite, briefMatch, plotMatches


def computeExtrinsics(K, H):
  phi_prime = np.dot(np.linalg.inv(K), H)
  u, l, vt = np.linalg.svd(phi_prime[:, :2])
  omega12 = np.dot(np.dot(u, np.array([[1, 0], [0, 1], [0, 0]])), vt)
  omega3 = np.cross(omega12[:, 0], omega12[:, 1])
  omega3 /= np.sum(omega3 ** 2)
  omega = np.hstack((omega12, np.array([omega3]).T))
  
  if np.linalg.det(omega) == -1:
    omega[:, 2] *= -1
  
  lamda_prime = np.sum(phi_prime[:, :2] / omega[:, :2]) / 6
  t = phi_prime[:, 2] / lamda_prime
  t = t.T
  
  return omega, t

def projectExtrinsics(K, W, R, t):
  # print("1: ", K.shape, R.shape, t.shape, W.shape)
  W = np.vstack((W, np.ones(W.shape[1])))
  # print("2: ", K.shape, R.shape, t.shape, W.shape)
  X = np.dot(np.dot(K, np.vstack((R[:, 0], R[:, 1], R[:, 2], t)).T), W)
  X /= X[-1, :]
  
  return X


if __name__ == '__main__':
  im = cv2.imread('../data/prince_book.jpeg')
  points3D = np.loadtxt('../data/sphere.txt')
  # data points 
  W = np.array([[0.0, 18.2, 18.2, 0.0], [0.0, 0.0, 26.0, 26.0], [0.0, 0.0, 0.0, 0.0]])
  D = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
  K = np.array([[3043.72, 0.0, 1196.00], [0.0, 3043.72, 1604.00], [0.0, 0.0, 1.0]])
  H2to1 = computeH(D, W[:2])
  
  o_xy = [830, 1645, 1]
  o_xyz = np.matmul(np.linalg.inv(H2to1), o_xy)
  o_xyz /= o_xyz[2]
  o_xyz[2] = 6.8581 / 2
  points3D += o_xyz[:, np.newaxis]
  
  R, t = computeExtrinsics(K, H2to1)
  X = projectExtrinsics(K, points3D, R, t)
  X = X.astype(int)
  
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  plt.imshow(im)
  plt.plot(X[0, :], X[1, :], 'y-', linewidth=0.4)
  plt.show()
  
