import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    A = []
    for i in range(p1.shape[1]):
      ui, vi, xi, yi = p2[0, i], p2[1, i], p1[0, i], p1[1, i]
      A.append([ui, vi, 1, 0, 0, 0, -xi*ui, -xi*vi, -xi])
      A.append([0, 0, 0, -ui, -vi, -1, yi*ui, yi*vi, yi])
    
    A = np.array(A)
    u, s, vt = np.linalg.svd(A)
    H2to1 = vt[-1, :].reshape(3, 3)
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    p1, p2 = [], []
    for i in range(matches.shape[0]):
      p1.append(locs1[matches[i, 0]][:2])
      p2.append(locs2[matches[i, 1]][:2])
    
    p1, p2 = np.array(p1), np.array(p2)
    # print(p1)
    iterations = 0
    maxInliers = 0
    bestH = None
    
    while  iterations < num_iter:
      try:
        randIdx = np.random.random((1, 4)) * matches.shape[0]
        randIdx = np.squeeze(randIdx.astype(np.int32))
        randP1 = p1[randIdx].T
        randP2 = p2[randIdx].T
        H2to1 = computeH(randP1, randP2)
        # generate homography points' cordinates
        appendOnes = np.ones((p1.shape[0], 1))
        homoP1 = np.hstack((p1, appendOnes)).T
        homoP2 = np.hstack((p2, appendOnes)).T
        transP2 = np.dot(H2to1, homoP2)
        transP2 /= transP2[2, :]
        # compute dist
        distance = np.sum((homoP1-transP2) ** 2, axis=0) ** 0.5
        # cal maybeInliers
        numInliers = distance[distance <= tol].size
        if numInliers > maxInliers:
          maxInliers = numInliers
          bestH = H2to1
      except:
        print("Iter: ", iterations, " selecting error, continuing...")
      
      iterations += 1
    
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # print(matches)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    print(bestH)

