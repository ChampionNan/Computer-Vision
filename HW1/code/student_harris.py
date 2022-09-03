from ast import operator
import enum
from math import radians
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

np.seterr(divide='ignore', invalid='ignore')

def local_non_maximal_supression(harris_response, kernel_size=25):
  pad = int(kernel_size/2)
  harris_response = np.pad(harris_response, ((pad, pad), (pad, pad)), mode="constant")
  new_harris_response = np.zeros_like(harris_response)
  h, w = harris_response.shape
  
  for i in range(pad, h-pad):
    for j in range(pad, w-pad):
      max_val = np.amax(harris_response[i - pad: i + pad + 1, j - pad: j + pad + 1])
      if harris_response[i, j] == np.amax(harris_response[i - pad: i + pad + i, j - pad: j + pad + 1]):
        new_harris_response[i, j] = max_val
        
  
  new_harris_response = new_harris_response[pad:h-pad, pad:w-pad]
  
  return new_harris_response

def get_max_corners(harris_response, max_no_corners=150):
  r, c = np.where(harris_response)
  val = harris_response[r, c]
  
  corners_xy = np.vstack((c, r)).T
  
  if max_no_corners > val.size:
    index = np.argsort(val)[::-1]
  elif max_no_corners > 0:
    index = np.argsort(val)[::-1][:max_no_corners]
  else:
    index = np.argsort(val)[::-1]
    
  corners_xy = corners_xy[index]
  return corners_xy


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    I_x = cv2.Sobel(image, -1, 1, 0)
    I_y = cv2.Sobel(image, -1, 0, 1)
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)
    
    k = 0.04
    thresh = 0.02
    N = 1500
    detA = Ixx * Iyy - Ixy ** 2
    traceA = Ixx + Iyy
    
    harris_response = detA - k * traceA ** 2
    cornelList = []
    avg = np.mean(harris_response[harris_response > 0])
    new_thresh = thresh * avg
    
    localMaximum = ndi.rank_filter(harris_response, rank=-1, size=(6, 6))
    print(harris_response.shape, localMaximum.shape)
    # print(harris_response[0,:], localMaximun[0,:])
    
    for row, response in enumerate(harris_response):
      for col, r in enumerate(response):
        if r > new_thresh and r == localMaximum[row, col]:
          cornelList.append([col, row, r]) # x, y, r
          
    # harris_response[np.where(harris_response > thresh * avg)] = 255
    # harris_response[np.where(harris_response < thresh * avg)] = 0
    
    # 1. maximum supression
    # corners_response = local_non_maximal_supression(harris_response, kernel_size=25)
    # detected_corners = get_max_corners(corners_response, max_no_corners=N)
    # print(detected_corners)
    
    # cornelList.sort(key=lambda radius: radius[2])
    cornelList = np.array(cornelList)
    # randIdx = np.random.randint(len(cornelList),size=1500)
    x = cornelList[0:N, 0]
    y = cornelList[0:N, 1]
    # print(x[0:100], y[0:100])
    '''
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(x[0:50], y[0:50], '.r', markersize=3)
    '''
    # raise NotImplementedError('`get_interest_points` function in ' + '`student_harris.py` needs to be implemented')
    
    return x, y, confidences, scales, orientations
    
    

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    
    raise NotImplementedError('adaptive non-maximal suppression in ' +
    '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations


