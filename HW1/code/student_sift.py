import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    fv = np.zeros((128))
    px, py, nx, ny = 0, 1, 2, 3
    pxpy, pxny, nxpy, nxny = 4, 5, 6, 7
    pad_r, pad_c = 8, 8
    
    kernel = cv2.getGaussianKernel(16, 1)
    G = kernel * kernel.T
    
    assert len(x) == len(y), "Imbalanced location values"
    # print(x.size)
    for i in range(0, len(x)):
      x[i], y[i] = int(np.round(x[i])), int(np.round(y[i]))
      row1, row2 = int(x[i]-pad_r), int(x[i]+pad_r)
      col1, col2 = int(y[i]-pad_c), int(y[i]+pad_c)
      # print(row1, col1)
      if row1 >= 0 and row2 <= image.shape[0] and col1 >= 0 and col2 <= image.shape[1]:
        # print(image)
        temp_cell = image[row1:row2, col1:col2]
        H_keypoint = np.zeros((8))
        for j in range(0, 4):
          indexj = j * 4
          for k in range(0, 4):
            histogram = np.zeros((8))
            indexx = k * 4
            temp = temp_cell[indexj:indexj+4, indexx:indexx+4]
            temp_gaussian = G[indexj:indexj+4, indexx:indexx+4]
            [y_gradient, x_gradient] = np.gradient(temp, edge_order=2)
            # y_gradient, x_gradient = y_gradient * 10, x_gradient * 10
            # diag_gradient = np.sqrt(x_gradient * x_gradient + y_gradient * y_gradient)
            diag_gradient = np.sqrt(x_gradient * x_gradient + y_gradient * y_gradient)
            # print(x_gradient, y_gradient)
            # print(temp_cell, temp.shape, y[i], col1, col2)
            
            for g in range(0, temp.shape[0]):
              for h in range(0, temp.shape[1]):
                if x_gradient[g, h] >= 0 and y_gradient[g, h] >= 0:
                  # print(py, g, h, histogram.shape, x_gradient.shape)
                  histogram[py] += y_gradient[g, h]
                  histogram[px] += x_gradient[g, h]
                  histogram[pxpy] += diag_gradient[g, h]
                elif x_gradient[g, h] >= 0 and y_gradient[g, h] < 0:
                  histogram[ny] += y_gradient[g, h]
                  histogram[px] += x_gradient[g, h]
                  histogram[pxny] += diag_gradient[g, h]
                elif x_gradient[g, h] < 0 and y_gradient[g, h] >= 0:
                  histogram[py] += y_gradient[g, h]
                  histogram[nx] += x_gradient[g, h]
                  histogram[nxpy] += diag_gradient[g, h]
                else:
                  histogram[ny] += y_gradient[g, h]
                  histogram[nx] += x_gradient[g, h]
                  histogram[nxny] += diag_gradient[g, h]
                  
            # if fv.shape[0] == 11 or fv.shape[0] == 12:
            # print(fv.shape, H_keypoint.shape, j, k)
            H_keypoint = np.hstack((H_keypoint, histogram))

      # print(fv.shape, H_keypoint.shape)
        H_keypoint = H_keypoint[8:]
        fv = np.vstack((fv, H_keypoint))

    # print(fv)
    fv = fv[1:, :]
    fv = np.delete(fv,np.where(np.isnan(fv))[0],axis=0)
    fv_norm = np.linalg.norm(fv)
    # fv_norm = fv_norm[np.newaxis, :] 
    # fv_norm = fv_norm[:, np.newaxis]
    np.seterr(invalid='ignore')
    fv = np.true_divide(fv, fv_norm)
    
    # raise NotImplementedError('`get_features` function in ' + '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
