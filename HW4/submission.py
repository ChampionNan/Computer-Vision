"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
from ctypes import pointer
from locale import MON_1
from turtle import shape

from cv2 import threshold
import numpy as np
import helper
import scipy
import scipy.stats as stats
import scipy.optimize as opt

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    x1, y1, x2, y2 = pts1[:, 0]/M, pts1[:, 1]/M, pts2[:, 0]/M, pts2[:, 1]/M
    # step 0
    normMat = np.array([[1./M, 0, 0], [0, 1./M, 0], [0, 0, 1]])
    # step 1
    A = np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(x1.shape))).T
    # step 2
    u, s, vt = np.linalg.svd(A)
    # step 3
    F = np.reshape(vt[-1, :], (3, 3))
    F = helper.refineF(F, pts1/M, pts2/M)
    # step 4
    F = helper._singularize(F)
    # step 5
    F = np.dot(normMat.T, np.dot(F, normMat))

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
  # Replace pass by your implementation
  x1, y1, x2, y2 = pts1[:, 0]/M, pts1[:, 1]/M, pts2[:, 0]/M, pts2[:, 1]/M
  normMat = np.array([[1./M, 0, 0], [0, 1./M, 0], [0, 0, 1]])
  A = np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(x1.shape))).T
  
  u, s, vt = np.linalg.svd(A)
  F1 = np.reshape(vt[-1, :], (3, 3))
  F2 = np.reshape(vt[-2, :], (3, 3))
  
  fun = lambda alpha: np.linalg.det(alpha * F1 + (1 - alpha) * F2)
  
  a0 = fun(0)
  a1 = 2 * (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
  a2 = (fun(1) + fun(-1)) / 2 - a0
  a3 = (fun(1) - fun(-1)) / 2 - a1
  
  alpha = np.roots([a3, a2, a1, a0])
  
  F = [a * F1 + (1 - a) * F2 for a in alpha]
  F = [helper.refineF(f, pts1/M, pts2/M) for f in F]
  F = [np.dot(normMat.T, np.dot(f, normMat)) for f in F]
  
  return F
  
'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.dot(K2.T, np.dot(F, K1))
    
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    x1, y1, x2, y2 = pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]
    a1 = np.vstack((C1[0, 0]-C1[2, 0]*x1, C1[0, 1]-C1[2, 1]*x1, C1[0, 2]-C1[2, 2]*x1, C1[0, 3]-C1[2, 3]*x1)).T
    a2 = np.vstack((C1[1, 0]-C1[2, 0]*y1, C1[1, 1]-C1[2, 1]*y1, C1[1, 2]-C1[2, 2]*y1, C1[1, 3]-C1[2, 3]*y1)).T
    a3 = np.vstack((C2[0, 0]-C2[2, 0]*x2, C2[0, 1]-C2[2, 1]*x2, C2[0, 2]-C2[2, 2]*x2, C2[0, 3]-C2[2, 3]*x2)).T
    a4 = np.vstack((C2[1, 0]-C2[2, 0]*y2, C2[1, 1]-C2[2, 1]*y2, C2[1, 2]-C2[2, 2]*y2, C2[1, 3]-C2[2, 3]*y2)).T
    
    N = pts1.shape[0]
    P = np.zeros((N, 3))
    for index in range(N):
      A = np.vstack((a1[index, :], a2[index, :], a3[index, :], a4[index, :]))
      u, s, vh = np.linalg.svd(A)
      P[index, :] = vh[-1, :][:3] / vh[-1, :][-1]
      
    W = np.hstack((P, np.ones((N, 1))))
    err = 0
    for i in range(N):
      p1 = np.dot(C1, W[i,:].T)
      p2 = np.dot(C2, W[i, :].T)
      p1 = (p1[:2] / p1[-1]).T
      p2 = (p2[:2] / p2[-1]).T
      err += np.sum((p1 - pts1[i])**2 + (p2-pts2[i])**2)
    return P, err
    
''' 
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x1, y1 = int(round(x1)), int(round(y1))
    windowSize = 41
    windowOffset = windowSize // 2
    center = windowSize // 2
    sigma = 5
    searchLength = 40
    
    # gaussian kernel
    x = np.linspace(-sigma, sigma, windowSize + 1)
    gk1d = np.diff(stats.norm.cdf(x))
    gk2d = np.outer(gk1d, gk1d)
    gk2d = gk2d / gk2d.sum()
    gk2d = np.dstack((gk2d, gk2d, gk2d))
    
    patchIm1 = im1[y1-windowOffset:y1+windowOffset+1, x1-windowOffset:x1+windowOffset+1]
    
    epiLine = np.dot(F, np.array([x1, y1, 1])).reshape((3, 1))
    a, b, c = epiLine[0], epiLine[1], epiLine[2]
    
    y2 = np.arange(y1-searchLength, y1+searchLength)
    x2 = np.round((-b*y2-c)/a).astype(int)
    validPoints = (x2 >= windowOffset) & (x2 < im2.shape[1] - windowOffset) & (y2 >= windowOffset) & (y2 < im2.shape[0] - windowOffset)
    x2, y2 = x2[validPoints], y2[validPoints]
    
    minDist = float('inf')
    x2Min, y2Min = 0, 0
    for i in range(x2.shape[0]):
      patchIm2 = im2[y2[i]-windowOffset:y2[i]+windowOffset+1, x2[i]-windowOffset:x2[i]+windowOffset+1]
      dist = np.sum(np.square(patchIm1-patchIm2)*gk2d)
      if dist < minDist:
        minDist = dist
        x2Min, y2Min = x2[i], y2[i]
        
    return x2Min, y2Min

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    points0 = pts1.shape[0]
    iteration = 1000
    threshold = 0.8
    
    homoPts1, homoPts2 = np.hstack((pts1, np.ones((points0, 1)))), np.hstack((pts2, np.ones((points0, 1))))
    maxF = np.zeros((3, 3))
    maxInliers = np.zeros((points0))
    
    for i in range(iteration):
      points = np.random.randint(0, pts1.shape[0], 7)
      points17, points27 = pts1[points, :], pts2[points, :]
      FList = sevenpoint(points17, points27, M)
      
      for F in FList:
        epiLines = np.dot(F, homoPts1.T).T
        col1, col2, col3 = epiLines[:, 0], epiLines[:, 1], epiLines[:, 2]
        
        dist = homoPts2 * epiLines
        dist = dist / np.sqrt(np.square(col1[:, np.newaxis]) + np.square(col2[:, np.newaxis]))
        dist = np.sum(dist, axis=-1)
        inliers = np.where(np.abs(dist) < threshold, True, False)
        inliers0 = np.sum(inliers)
        if inliers0 > np.sum(maxInliers):
          maxInliers = inliers
          maxF = F
          
    return maxF, maxInliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    if theta == 0:
      return np.eye(3)
    else:
      temp = r / theta
      tempCross = np.array([[0.0, -temp[2, 0], temp[1, 0]], [temp[2, 0], 0.0, -temp[0, 0]], [-temp[1, 0], temp[0, 0], 0.0]])
      R = np.eye(3) + np.sin(theta) * tempCross + (1 - np.cos(theta)) * (np.dot(temp, temp.T) - np.sum(np.square(temp) * np.eye(3)))
      return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    threshold = 1e-15
    A = (R - R.T) / 2
    rho = np.array(A[[2, 0, 1], [1, 2, 0]])[:, np.newaxis]
    s = np.float64(np.linalg.norm(rho))
    c = np.float64((np.trace(R) - 1) / 2)
    if s < threshold and (c - 1) < threshold:
      r = np.array([0.0, 0.0, 0.0])[:, np.newaxis]
      return r
    elif s < 1e-15 and (c + 1) < threshold:
      v = None
      for i in range(R.shape[-1]):
        v = (R + np.eye(3))[:, i]
        if np.count_nonzero(v) > 0:
          break
      u = v / np.linalg.norm(v)
      r = (u * np.pi)[:, np.newaxis]
      if np.linalg.norm(r) == np.pi and (r[0, 0] == r[1, 0] == 0 and r[2, 0] < 0.0) or (r[0, 0] == 0 and r[1, 0] < 0) or (r[0, 0] < 0):
        return -r
      else:
        return r
    else:
      u = rho / s
      theta = np.arctan2(s, c)
      r = u * theta
      return r
    
'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P, r2, t2 = x[:-6], x[-6:-3], x[-3:]

    N = P.shape[0]//3
    P = np.reshape(P, (N, 3))
    r2 = np.reshape(r2, (3, 1))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    P = np.vstack((np.transpose(P), np.ones((1, N))))
    p1_hat = np.dot(np.dot(K1, M1), P)
    p1_hat = np.transpose(p1_hat[:2, :]/p1_hat[2, :])
    p2_hat = np.dot(np.dot(K2, M2), P)
    p2_hat = np.transpose(p2_hat[:2, :]/p2_hat[2, :])

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])

    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    x_init = P_init.flatten()
    R2 = M2_init[:, 0:3]
    t2 = M2_init[:, 3]
    
    r2 = invRodrigues(R2)
    x_init = np.append(x_init, r2.flatten())
    x_init = np.append(x_init, t2.flatten())
    
    f = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    x_opt, _ = opt.leastsq(f, x_init)
    
    w_opt, r2_opt, t2_opt = x_opt[:-6], x_opt[-6:-3], x_opt[-3:]
    W_opt = w_opt.reshape((w_opt.shape[0] // 3, 3))
    r2_opt = r2_opt[:, np.newaxis]
    t2_opt = t2_opt[:, np.newaxis]
    
    R2_opt = rodrigues(r2_opt)
    M2_opt = np.hstack((R2_opt, t2_opt))
    
    return M2_opt, W_opt
    