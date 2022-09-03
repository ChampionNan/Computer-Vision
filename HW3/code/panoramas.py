from turtle import width
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
from soupsieve import match

def generateMask(im1, im2, homo1, homo2, outputShape):
  height1, width1, _ = im1.shape
  height2, width2, _ = im2.shape
  im1_mask = np.zeros((height1, width1))
  im1_mask[0, :] = 1
  im1_mask[:, 0] = 1
  im1_mask[-1, :] = 1
  im1_mask[:, -1] = 1
  im1_mask = distance_transform_edt(1 - im1_mask)
  warp_mask1 = cv2.warpPerspective(im1_mask / np.max(im1_mask), homo1, outputShape)
  
  im2_mask = np.zeros((height2, width2))
  im2_mask[0, :] = 1
  im2_mask[:, 0] = 1
  im2_mask[-1, :] = 1
  im2_mask[:, -1] = 1
  im2_mask = distance_transform_edt(1 - im2_mask)
  warp_mask2 = cv2.warpPerspective(im2_mask / np.max(im2_mask), homo2, outputShape)
  
  sum_mask = warp_mask1 + warp_mask2
  warp_mask1 /= sum_mask
  warp_mask2 /= sum_mask
  warp_mask1 = np.stack((warp_mask1, warp_mask1, warp_mask1), axis=2)
  warp_mask2 = np.stack((warp_mask2, warp_mask2, warp_mask2), axis=2)
  
  return warp_mask1, warp_mask2
   

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    height1, width1, _ = im1.shape
    height2, width2, _ = im2.shape
    corners = np.array([[0, 0, 1], [0, height2-1, 1], [width2-1, 0, 1], [width2-1, height2-1, 1]])
    homo_corners = np.dot(H2to1, corners.T)
    homo_corners /= homo_corners[2, :]
    
    max_width, max_height = int(np.round(np.max(homo_corners[0, :]))), int(np.max(homo_corners[1, :]))
    
    warp_im2 = cv2.warpPerspective(im2, H2to1, (max_width, max_height))
    cv2.imwrite('../results/4_1.jpg', warp_im2)
    # generateMask
    # type: ignore
    homo1 = np.float32([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])  
    warp_mask1, warp_mask2 = generateMask(im1, im2, homo1, H2to1, (max_width, max_height))
    pano_width = max(width1, max_width)
    pano_height = max(height1, max_height)
    pano_im = np.zeros((pano_height, pano_width, 3), im1.dtype)
    pano_im[:height1, :width1, :] = im1
    
    warp_res = pano_im * warp_mask1 + warp_im2 * warp_mask2
    
    nzero_points = tuple(np.array(np.nonzero(warp_im2)[:2]))
    pano_im[nzero_points] = warp_im2[nzero_points]
    overlap = tuple(np.array(np.nonzero(pano_im * warp_im2)[:2]))
    pano_im[overlap] = warp_res[overlap]
    
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    height1, width1, _ = im1.shape
    height2, width2, _ = im2.shape 
    # print(2)
    corners = np.array([[0, 0, 1], [0, height2-1, 1], [width2-1, 0, 1], [width2-1, height2-1, 1]])
    homo_corners = np.dot(H2to1, corners.T)
    homo_corners = homo_corners / homo_corners[2, :]
    # print(3)
    min_width, min_height = int(np.round(np.min(homo_corners[0, :]))), int(np.round(np.min(homo_corners[1, :])))
    
    M = np.float32([[1, 0, max(-min_width, 0)], [0, 1, max(-min_height, 0)], [0, 0, 1]])  
    # print(4)
    width, height = max(-min_width, 0) + int(np.round(np.max(homo_corners[0, :]))), max(-min_height, 0) + int(np.round(np.max(homo_corners[1, :])))
    # print(5)
    warp_im1, warp_im2 = cv2.warpPerspective(im1, M, (width, height)), cv2.warpPerspective(im2, np.matmul(M, H2to1), (width, height))
    # print(6)
    warp_mask1, warp_mask2 = generateMask(im1, im2, M, np.matmul(M, H2to1), (width, height))
    pano_im = np.zeros((height, width, 3), im1.dtype)
    nonzero_im1 = tuple(np.array(np.nonzero(warp_im1)[:2]))
    pano_im[nonzero_im1] = warp_im1[nonzero_im1]
    # print(7)
    mixed = pano_im * warp_mask1 + warp_im2 * warp_mask2
    # print(8)
    nonzero_im2 = tuple(np.array(np.nonzero(warp_im2)[:2]))
    pano_im[nonzero_im2] = warp_im2[nonzero_im2]
    overlap_nonzero = tuple(np.array(np.nonzero(pano_im * warp_im2)[:2]))
    pano_im[overlap_nonzero] = mixed[overlap_nonzero]
    cv2.imwrite('../result/q4_2_pan.jpg', pano_im)
    
    return pano_im

def generatePanorama(im1, im2):
  locs_im1, desc_im1 = briefLite(im1)
  locs_im2, desc_im2 = briefLite(im2)
  matches = briefMatch(desc1, desc2)
  H2to1 = ransacH(matches, locs_im1, locs_im2)
  np.save('../results/q4_1.npy', H2to1)
  im3 = imageStitching_noClip(im1, im2, H2to1)
  cv2.imwrite("../results/q4_3.jpg", im3)
  return im3

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print(1)
    pano_im_noClip = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/panoImg_noClip.png', pano_im_noClip)
    cv2.imshow('panoramas', pano_im_noClip)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    im3 = generatePanorama(im1, im2)