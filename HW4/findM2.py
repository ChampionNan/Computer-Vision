'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import helper
import submission

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
pts1, pts2 = data['pts1'], data['pts2']

M = max(im1.shape)
F = submission.eightpoint(pts1, pts2, M)

intric = np.load('../data/intrinsics.npz')
K1, K2 = intric['K1'], intric['K2']
E = submission.essentialMatrix(F, K1, K2)

M2s = helper.camera2(E)
M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
C1 = np.dot(K1, M1)

M2 = None
P = None

for index in range(M2s.shape[-1]):
  M22 = M2s[:, :, index]
  C22 = np.dot(K2, M22)
  p, err = submission.triangulate(C1, pts1, C22, pts2)
  
  if np.min(p[:, -1]) > 0:
    M2 = M22
    P = p
    break
  
C2 = np.dot(K2, M2)
np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
