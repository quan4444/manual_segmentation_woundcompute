import cv2
import math
import numpy as np
from scipy.spatial import distance
from skimage.transform import rotate
from scipy import ndimage
from typing import List, Tuple, Union
from woundcompute import compute_values as com
from woundcompute import segmentation as seg


def compute_distance(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> Union[int, float]:
    """Given two 2D points. Will return the distance between them."""
    dist = ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5
    return dist


def compute_unit_vector(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> np.ndarray:
    """Given two 2D points. Will return the unit vector between them"""
    dist = compute_distance(x1, x2, y1, y2)
    vec = np.asarray([(x2 - x1) / dist, (y2 - y1) / dist])
    return vec


def compute_distance_multi_point(coords_1: np.ndarray, coords_2: np.ndarray):
    """Find the shortest distance between points in two arrays.
    Each array is formatted idx_0 points, idx_1 points."""
    # maximum array size -- downsample
    upper_lim = 1000
    if coords_1.shape[0] > upper_lim:
        val = int(coords_1.shape[0] / upper_lim)
        coords_1 = coords_1[::val, :]
    if coords_2.shape[0] > upper_lim:
        val = int(coords_2.shape[0] / upper_lim)
        coords_2 = coords_2[::val, :]
    arr = distance.cdist(coords_1, coords_2, 'euclidean')
    ind = np.unravel_index(np.argmin(arr, axis=None), arr.shape)
    coords_1_idx = ind[0]
    coords_2_idx = ind[1]
    pt1_0_orig = coords_1[coords_1_idx, 1]
    pt1_1_orig = coords_1[coords_1_idx, 0]
    pt2_0_orig = coords_2[coords_2_idx, 1]
    pt2_1_orig = coords_2[coords_2_idx, 0]
    return pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig


def box_to_unit_vec(box: np.ndarray) -> np.ndarray:
    """Given the rectangular box. Will compute the unit vector of the longest side."""
    side_1 = compute_distance(box[0, 0], box[1, 0], box[0, 1], box[1, 1])
    side_2 = compute_distance(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
    if side_1 > side_2:
        # side_1 is the long axis
        vec = compute_unit_vector(box[1, 0], box[0, 0], box[1, 1], box[0, 1])
    else:
        # side_2 is the long axis
        vec = compute_unit_vector(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
    return vec


def box_to_center_points(box: np.ndarray) -> float:
    """Given the rectangular box. Will compute the center as the midpoint of a diagonal."""
    center_row = np.mean([box[0, 0], box[2, 0]])
    center_col = np.mean([box[0, 1], box[2, 1]])
    return center_row, center_col


def insert_borders(mask: np.ndarray, border: int = 10) -> np.ndarray:
    """Given a mask. Will make the borders around it 0."""
    mask[0:border, :] = 0
    mask[-border:, :] = 0
    mask[:, 0:border] = 0
    mask[:, -border:] = 0
    return mask


def ix_loop(val: int, num_pts_contour: int) -> int:
    """Given an index value. Will loop it around (for contours)."""
    if val < 0:
        val = num_pts_contour + val
    if val >= num_pts_contour:
        val = val - num_pts_contour
    else:
        val = val
    return val


def get_local_curvature(contour: np.ndarray, mask: np.ndarray, ix_center: int, sample_dist: int) -> Union[float, int]:
    """Given a contour and a specified location. Will return curvature."""
    sample_dist = int(sample_dist)
    num_pts_contour = contour.shape[0]
    x0 = []
    x1 = []
    for kk in range(-sample_dist, sample_dist):
        val = ix_center + kk
        val = ix_loop(val, num_pts_contour)
        x0.append(contour[val, 0])
        x1.append(contour[val, 1])
    # find the best fit circle, see:
    # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    # coordinate of the barycenter
    x0_m = np.mean(x0)
    x1_m = np.mean(x1)
    # find the sign of the curvature
    midpoint_0 = int(x0_m)
    midpoint_1 = int(x1_m)
    if mask[midpoint_0, midpoint_1] > 0:
        kappa_sign = 1.0
    else:
        kappa_sign = -1.0
    # calculate the reduced coordinates
    u = x0 - x0_m
    v = x1 - x1_m
    # linear system defining the center (uc, vc) in reduced coordinates:
    #   Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #   Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)
    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    if np.abs(np.linalg.det(A)) < 10 ** np.finfo(float).eps * 10:
        kappa_correct_sign = float('inf')
    else:
        uc, vc = np.linalg.solve(A, B)
        x0c_1 = x0_m + uc
        x1c_1 = x1_m + vc
        # Calculates the distance from the center (xc_1, yc_1)
        Ri_1 = np.sqrt((x0 - x0c_1)**2 + (x1 - x1c_1)**2)
        R_1 = np.mean(Ri_1)
        # residu_1 = np.sum((Ri_1-R_1) ** 2)
        kappa = 1 / R_1
        kappa_correct_sign = kappa * kappa_sign
    return kappa_correct_sign


def mask_to_box(mask: np.ndarray, border: int = 0) -> np.ndarray:
    """Given a mask. Will return the minimum area bounding rectangle."""    
    # insert borders to the mask
    if border > 0:
        mask_mod = insert_borders(mask, border)
    else:
        mask_mod = mask
    # find contour
    mask_mod_one = (mask_mod > 0).astype(np.float64)
    # mask_thresh_blur = ndimage.gaussian_filter(mask_mod_one, 1)
    # cnts = measure.find_contours(mask_thresh_blur, 0.75)[0].astype(np.int32)
    coordinates = np.column_stack(np.where(mask_mod_one > 0))
    # find minimum area bounding rectangle
    rect = cv2.minAreaRect(coordinates)
    box = np.int0(cv2.boxPoints(rect))
    return box


def axis_from_mask(mask: np.ndarray) -> np.ndarray:
    """Given a folder path. Will import the mask and determine its long axis."""
    box = mask_to_box(mask)
    vec = box_to_unit_vec(box)
    center_row, center_col = box_to_center_points(box)
    return center_row, center_col, vec


def rot_vec_to_rot_mat_and_angle(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Given a rotation vector. Will return a rotation matrix and rotation angle."""
    ang = np.arctan2(vec[0], vec[1])
    rot_mat = np.asarray([[np.cos(ang), -1.0 * np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return (rot_mat, ang)


def get_rotation_info(
    *,
    center_row_input: Union[float, int] = None,
    center_col_input: Union[float, int] = None,
    vec_input: np.ndarray = None,
    mask: np.ndarray = None
) -> Tuple[Union[float, int], Union[float, int], np.ndarray, Union[float, int]]:
    """Given either prescribed rotation or mask.
    Will compute rotation information (rotation matrix and angle).
    Prescribed rotation will override rotation computed by the mask."""
    if mask is not None:
        center_row, center_col, vec = axis_from_mask(mask)
    if center_row_input is not None:
        center_row = center_row_input
    if center_col_input is not None:
        center_col = center_col_input
    if vec_input is not None:
        vec = vec_input
    (rot_mat, ang) = rot_vec_to_rot_mat_and_angle(vec)
    return (center_row, center_col, rot_mat, ang, vec)


def rot_image(
    img: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int],
    ang: float
) -> np.ndarray:
    """Given an image and rotation information. Will return rotated image."""
    new_img = rotate(img, ang / (np.pi) * 180, center=(center_col, center_row))
    return new_img


def rotate_points(
    row_pts: np.ndarray,
    col_pts: np.ndarray,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given array vectors of points, rotation matrix, and point to rotate about.
    Will perform rotation and return rotated points"""
    row_pts_centered = row_pts - center_row
    col_pts_centered = col_pts - center_col
    pts = np.hstack((row_pts_centered.reshape((-1, 1)), col_pts_centered.reshape((-1, 1)))).T
    pts_rotated = rot_mat @ pts
    new_row_pts = pts_rotated[0, :] + center_row
    new_col_pts = pts_rotated[1, :] + center_col
    return new_row_pts, new_col_pts


def invert_rot_mat(rot_mat: np.ndarray) -> np.ndarray:
    inv_rot_mat = rot_mat.transpose()
    return inv_rot_mat


def get_tissue_width(tissue_mask_robust: np.ndarray, width_buffer: int = 5) -> float:
    """Given a mask of the tissue. Will compute the width of the tissue at the center."""
    center_row_orig, center_col_orig, rot_mat, ang, vec = get_rotation_info(center_row_input=None, center_col_input=None, vec_input=None, mask=tissue_mask_robust)
    rot_mask = rot_image(tissue_mask_robust, center_row_orig, center_col_orig, ang)
    mask_box = mask_to_box(rot_mask)
    center_row, center_col = box_to_center_points(mask_box)
    tissue_width_all = []
    min_row_all = []
    max_row_all = []
    for kk in range(int(-1.0 * width_buffer), int(width_buffer)):
        center_width = np.nonzero(rot_mask[:, int(center_col + kk)] > 0)
        min_row = np.min(center_width)
        max_row = np.max(center_width)
        tissue_width_all.append(max_row - min_row)
        min_row_all.append(min_row)
        max_row_all.append(max_row)
    tissue_width = np.mean(tissue_width_all)
    # add in points --> points in un-rotated coordinate system
    pt1_0 = center_col
    pt1_1 = np.mean(min_row_all)
    pt2_0 = center_col
    pt2_1 = np.mean(max_row_all)
    row_pts = np.asarray([pt1_1, pt2_1])
    col_pts = np.asarray([pt1_0, pt2_0])
    # rotate points back to original coordinate system
    inv_rot_mat = invert_rot_mat(rot_mat)
    new_row_pts, new_col_pts = rotate_points(row_pts, col_pts, inv_rot_mat, center_row_orig, center_col_orig)
    pt1_0_orig = new_col_pts[0]
    pt1_1_orig = new_row_pts[0]
    pt2_0_orig = new_col_pts[1]
    pt2_1_orig = new_row_pts[1]
    return tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig


def get_tissue_width_zoom(tissue_mask: np.ndarray, wound_mask: np.ndarray):
    border = 5
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask, border)
    border = 1
    tissue_mask_robust = seg.insert_borders(tissue_mask_robust, border, 1)
    background_mask = tissue_mask_robust < 0.5
    border = 10
    background_mask_borders = seg.insert_borders(background_mask, border)
    region_props = seg.get_region_props(background_mask_borders)
    num_regions = 2
    regions_list = seg.get_roundest_regions(region_props, num_regions)
    # if there aren't sufficient regions (i.e., fewer than 2) to compute tissue width
    if len(regions_list) < 2:
        tissue_width = 0
        pt1_0_orig = 0
        pt1_1_orig = 0
        pt2_0_orig = 0
        pt2_1_orig = 0
    else:
        coords_1 = regions_list[0].coords
        coords_2 = regions_list[1].coords
        pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = compute_distance_multi_point(coords_1, coords_2)
        tissue_width = compute_distance(pt1_0_orig, pt2_0_orig, pt1_1_orig, pt2_1_orig)
    return tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig


# def get_tissue_width_zoom(tissue_mask_robust: np.ndarray, width_buffer: int = 5):
#     border_buffer = 10
#     tissue_mask_robust_border = seg.insert_borders(tissue_mask_robust, border_buffer, 1)
#     non_tissue = tissue_mask_robust_border < 0.5
#     regions_list = seg.get_region_props(non_tissue)
#     largest_region = seg.get_largest_regions(regions_list, 1)[0]
#     ang = largest_region.orientation
#     if ang < 0:
#         ang += np.pi / 2.0
#     else:
#         ang -= np.pi / 2.0
#     rot_mat = np.asarray([[np.cos(ang), -1.0 * np.sin(ang)], [np.sin(ang), np.cos(ang)]])
#     # vec = [np.cos(ang), np.sin(ang)]
#     center_row = int(tissue_mask_robust.shape[0] / 2.0)
#     center_col = int(tissue_mask_robust.shape[1] / 2.0)
#     rot_mask = rot_image(tissue_mask_robust, center_row, center_col, ang)
#     tissue_width_all = []
#     min_row_all = []
#     max_row_all = []
#     for kk in range(int(-1.0 * width_buffer), int(width_buffer)):
#         center_width = np.nonzero(rot_mask[:, int(center_col + kk)] > 0)
#         min_row = np.min(center_width)
#         max_row = np.max(center_width)
#         tissue_width_all.append(max_row - min_row)
#         min_row_all.append(min_row)
#         max_row_all.append(max_row)
#     tissue_width = np.mean(tissue_width_all)
#     # add in points --> points in un-rotated coordinate system
#     pt1_0 = center_col
#     pt1_1 = np.mean(min_row_all)
#     pt2_0 = center_col
#     pt2_1 = np.mean(max_row_all)
#     row_pts = np.asarray([pt1_1, pt2_1])
#     col_pts = np.asarray([pt1_0, pt2_0])
#     # rotate points back to original coordinate system
#     inv_rot_mat = invert_rot_mat(rot_mat)
#     new_row_pts, new_col_pts = rotate_points(row_pts, col_pts, inv_rot_mat, center_row, center_col)
#     pt1_0_orig = new_col_pts[0]
#     pt1_1_orig = new_row_pts[0]
#     pt2_0_orig = new_col_pts[1]
#     pt2_1_orig = new_row_pts[1]
#     return tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig


def compute_dist_line_pt(pt0, pt1, line):
    dist_all = ((line[:, 0] - pt0) ** 2.0 + (line[:, 1] - pt1) ** 2.0) ** 0.5
    return dist_all


def tissue_parameters(tissue_mask: np.ndarray, wound_mask: np.ndarray):
    area = np.sum(tissue_mask)
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = get_tissue_width(tissue_mask_robust)
    tissue_contour = seg.mask_to_contour(tissue_mask_robust)
    sample_dist = np.min([100, tissue_contour.shape[0] * 0.1])
    dist_pt1 = compute_dist_line_pt(pt1_0_orig, pt1_1_orig, tissue_contour)
    dist_pt2 = compute_dist_line_pt(pt2_0_orig, pt2_1_orig, tissue_contour)
    contour_idx_0 = np.argmin(dist_pt1)
    contour_idx_1 = np.argmin(dist_pt2)
    kappa_1 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_0, sample_dist)
    kappa_2 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_1, sample_dist)
    return tissue_width, area, kappa_1, kappa_2, pt1_1_orig, pt1_0_orig, pt2_1_orig, pt2_0_orig, tissue_contour


def tissue_parameters_zoom(tissue_mask: np.ndarray, wound_mask: np.ndarray):
    area = np.sum(tissue_mask)
    border = 10
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask, border)
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = get_tissue_width_zoom(tissue_mask, wound_mask)
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_contour = seg.mask_to_contour(tissue_mask_robust)
    if tissue_width == 0:
        return tissue_width, area, 0, 0, pt1_1_orig, pt1_0_orig, pt2_1_orig, pt2_0_orig, tissue_contour
    else:
        sample_dist = np.min([100, tissue_contour.shape[0] * 0.1])
        dist_pt1 = compute_dist_line_pt(pt1_0_orig, pt1_1_orig, tissue_contour)
        dist_pt2 = compute_dist_line_pt(pt2_0_orig, pt2_1_orig, tissue_contour)
        contour_idx_0 = np.argmin(dist_pt1)
        contour_idx_1 = np.argmin(dist_pt2)
        kappa_1 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_0, sample_dist)
        kappa_2 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_1, sample_dist)
        return tissue_width, area, kappa_1, kappa_2, pt1_1_orig, pt1_0_orig, pt2_1_orig, pt2_0_orig, tissue_contour


# def tissue_parameters_zoom(tissue_mask: np.ndarray, wound_mask: np.ndarray) -> Union[float, int]:
#     """Given a tissue mask. Will compute and return key properties."""
#     area = np.sum(tissue_mask)
#     tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
#     # tm_c_0, tm_c_1 = get_mean_center(tissue_mask_robust)
#     tissue_contour = seg.mask_to_contour(tissue_mask_robust)
#     tissue_regions_all = seg.get_region_props(tissue_mask_robust)
#     tissue_region = seg.get_largest_regions(tissue_regions_all, 1)[0]
#     _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(tissue_region)
#     width, contour_idx_0, contour_idx_1 = get_contour_width(tissue_contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length, orientation)
#     sample_dist = np.min([100, tissue_contour.shape[0] * 0.1])
#     kappa_1 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_0, sample_dist)
#     kappa_2 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_1, sample_dist)
#     pt1_0 = tissue_contour[contour_idx_0, 0]
#     pt1_1 = tissue_contour[contour_idx_0, 1]
#     pt2_0 = tissue_contour[contour_idx_1, 0]
#     pt2_1 = tissue_contour[contour_idx_1, 1]
#     return width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour


def tissue_parameters_all(tissue_mask_list: List, wound_mask_list: List, zoom_fcn_idx: int) -> List:
    """Given tissue and wound masks. Will return tissue parameters."""
    #  parameter list has order:
    #  area, pt1_0, pt1_1, pt2_0, pt2_1, width, kappa_1, kappa_2
    parameter_list = []
    for kk in range(0, len(tissue_mask_list)):
        if zoom_fcn_idx == 1:
            width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, _ = com.tissue_parameters_zoom(tissue_mask_list[kk], wound_mask_list[kk])
        elif zoom_fcn_idx == 2:
            width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, _ = com.tissue_parameters(tissue_mask_list[kk], wound_mask_list[kk])
        param = [area, pt1_0, pt1_1, pt2_0, pt2_1, width, kappa_1, kappa_2]
        parameter_list.append(param)
    return parameter_list


def check_broken_tissue(tissue_mask: np.ndarray, tissue_mask_orig: np.ndarray = None) -> bool:
    """Given a tissue mask. Will return true if it's a broken tissue."""
    is_broken = False
    # test if broken via no segmented regions
    region_props = seg.get_region_props(tissue_mask)
    if len(region_props) == 0:
        return True
    largest_region = seg.get_largest_regions(region_props, 1)[0]
    # area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation
    area, _, _, centroid_row, centroid_col, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(largest_region)
    # test if broken via being on 1 or 2 pillars (short)
    pix_mask = tissue_mask.shape[0] * tissue_mask.shape[1]
    if area < pix_mask * 0.1:
        is_broken = True
        return is_broken
    # test if broken via being on 2 pillars (long)
    if tissue_mask_orig is None:
        mask_row_center = tissue_mask.shape[0] / 2.0
        mask_col_center = tissue_mask.shape[1] / 2.0
    else:
        region_props = seg.get_region_props(tissue_mask_orig)
        largest_region = seg.get_largest_regions(region_props, 1)[0]
        _, _, _, centroid_row_orig, centroid_col_orig, _, (_, _, _, _), _ = seg.extract_region_props(largest_region)
        mask_row_center = centroid_row_orig
        mask_col_center = centroid_col_orig
    row_fraction_offset = np.abs(centroid_row - mask_row_center) / tissue_mask.shape[0]
    col_fraction_offset = np.abs(centroid_col - mask_col_center) / tissue_mask.shape[1]
    if row_fraction_offset > 0.1 or col_fraction_offset > 0.1:
        is_broken = True
        return is_broken
    # test if broken via lack of quad symmetry (on 3 pillars)
    mid_row = int(min_row * 0.5 + max_row * 0.5)
    mid_col = int(min_col * 0.5 + max_col * 0.5)
    Q1_area = np.sum(tissue_mask[min_row:mid_row, min_col:mid_col])
    Q2_area = np.sum(tissue_mask[mid_row:max_row, min_col:mid_col])
    Q3_area = np.sum(tissue_mask[mid_row:max_row, mid_col:max_col])
    Q4_area = np.sum(tissue_mask[min_row:mid_row, mid_col:max_col])
    Q_list = [Q1_area, Q2_area, Q3_area, Q4_area]
    min_area = np.min(Q_list)
    max_area = np.max(Q_list)
    mean_area = np.mean(Q_list)
    if min_area / max_area < 0.25 or min_area / mean_area < 0.60:
        is_broken = True
        return is_broken
    return is_broken


def split_into_four_corners_with_pillars(
    tissue_mask:np.ndarray,
    pillar_mask_list:List[np.ndarray],
    pillar_mask_fill_border:int=10,
    )->List[np.ndarray]:
    '''Given a tissue mask and a list of pillar masks, split the tissue mask into
    four tissue quarter masks using the bounding box of the pillars.'''
    
    pillar_mask = seg.mask_list_to_single_mask(pillar_mask_list)
    box_raw = seg.pillar_mask_to_rotated_box(pillar_mask,pillar_mask_fill_border)
    box = np.zeros_like(box_raw)
    box[:,0],box[:,1]=box_raw[:,1],box_raw[:,0]

    # sort the box points to ensure consistent ordering
    box = box[np.argsort(box[:, 0])]
    left = box[:2]
    right = box[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    box = np.vstack((left, right))

    mid_top = (box[1] + box[3]) // 2
    mid_bottom = (box[0] + box[2]) // 2
    mid_left = (box[0] + box[1]) // 2
    mid_right = (box[2] + box[3]) // 2
    center = (mid_left + mid_right) // 2

    height, width = tissue_mask.shape
    mask_quarters = []
    for i in range(4):
        quarter_mask = np.zeros((height, width), dtype=np.uint8)
        
        if i == 0:  # Top-left quarter
            pts = np.array([box[1]+[2,1], mid_left+[2,0], center, mid_top+[-2,0]])
        elif i == 1:  # Top-right quarter
            pts = np.array([mid_top, center, mid_right, box[3]])
        elif i == 2:  # Bottom-left quarter
            pts = np.array([mid_left, box[0], mid_bottom, center])
        else:  # Bottom-right quarter
            pts = np.array([center, mid_bottom, box[2], mid_right])
        
        cv2.fillPoly(quarter_mask, [pts], 255)
        mask_quarters.append(quarter_mask)

    tissue_mask = tissue_mask.astype(np.uint8)
    tissue_quarter_masks = []
    for _,mask_quarter in enumerate(mask_quarters):
        # tissue_quarter = tissue_mask[mask_quarter>0]
        mask_quarter = mask_quarter.astype(np.uint8)
        tissue_quarter = cv2.bitwise_and(tissue_mask, mask_quarter)
        tissue_quarter_masks.append(tissue_quarter)
    return tissue_quarter_masks


def obtain_tissue_quarters_area(tissue_mask:np.ndarray,pillar_mask_list=List[np.ndarray],border:int=10)->float:
    '''Given tissue mask and the list of pillar masks, divide the tissue mask into four
    tissue quarter masks, and return the area of each tissue quarter mask.'''
    tissue_quarter_masks = split_into_four_corners_with_pillars(tissue_mask,pillar_mask_list,border)
    Q1_area = np.sum(tissue_quarter_masks[0])/np.amax(tissue_quarter_masks[0])
    Q2_area = np.sum(tissue_quarter_masks[1])/np.amax(tissue_quarter_masks[1])
    Q3_area = np.sum(tissue_quarter_masks[2])/np.amax(tissue_quarter_masks[2])
    Q4_area = np.sum(tissue_quarter_masks[3])/np.amax(tissue_quarter_masks[3])
    return [Q1_area,Q2_area,Q3_area,Q4_area],tissue_quarter_masks


def check_broken_tissue_with_pillars(
    tissue_mask: np.ndarray,
    pillar_mask_list:List[np.ndarray],
    tissue_mask_orig: np.ndarray = None
    ) -> bool:
    """Given a tissue mask and list of pillar masks. Will return true if it's a broken tissue."""
    is_broken = False
    # test if broken via no segmented regions
    region_props = seg.get_region_props(tissue_mask)
    if len(region_props) == 0:
        return True
    largest_region = seg.get_largest_regions(region_props, 1)[0]
    # area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation
    area, _, _, centroid_row, centroid_col, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(largest_region)
    # test if broken via being on 1 or 2 pillars (short)
    pix_mask = tissue_mask.shape[0] * tissue_mask.shape[1]
    if area < pix_mask * 0.1:
        is_broken = True
        return is_broken
    # test if broken via being on 2 pillars (long)
    if tissue_mask_orig is None:
        mask_row_center = tissue_mask.shape[0] / 2.0
        mask_col_center = tissue_mask.shape[1] / 2.0
    else:
        region_props = seg.get_region_props(tissue_mask_orig)
        largest_region = seg.get_largest_regions(region_props, 1)[0]
        _, _, _, centroid_row_orig, centroid_col_orig, _, (_, _, _, _), _ = seg.extract_region_props(largest_region)
        mask_row_center = centroid_row_orig
        mask_col_center = centroid_col_orig
    row_fraction_offset = np.abs(centroid_row - mask_row_center) / tissue_mask.shape[0]
    col_fraction_offset = np.abs(centroid_col - mask_col_center) / tissue_mask.shape[1]
    if row_fraction_offset > 0.1 or col_fraction_offset > 0.1:
        is_broken = True
        return is_broken
    # test if broken via lack of quad symmetry (on 3 pillars)
    Q_list,_ = obtain_tissue_quarters_area(tissue_mask,pillar_mask_list)
    min_area = np.amin(Q_list)
    max_area = np.amax(Q_list)
    mean_area = np.mean(Q_list)
    if min_area / max_area < 0.25 or min_area / mean_area < 0.60:
        is_broken = True
        return is_broken
    return is_broken


def binary_mask_IOU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou


def check_broken_tissue_zoom(tissue_mask: np.ndarray, wound_mask: np.ndarray, tissue_mask_orig: np.ndarray, wound_mask_orig: np.ndarray):
    # make comparison to original tissue
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_mask_robust_orig = seg.make_tissue_mask_robust(tissue_mask_orig, wound_mask_orig)
    # comparison metrics -- simple IOU
    iou_masks = binary_mask_IOU(tissue_mask_robust, tissue_mask_robust_orig)
    if iou_masks < 0.75:
        return True #, iou_masks
    else:
        return False #, iou_masks


# def check_broken_tissue_zoom(tissue_mask: np.ndarray) -> bool:
#     """Given a tissue mask. Will return true if it's a broken tissue."""
#     is_broken = False
#     # test if broken via no segmented regions
#     region_props = seg.get_region_props(tissue_mask)
#     if len(region_props) == 0:
#         return True
#     # test if broken via being on 2 pillars (long)
#     largest_region = seg.get_largest_regions(region_props, 1)[0]
#     # area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation
#     area, _, _, centroid_row, centroid_col, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(largest_region)
#     # test if broken via being on 1 or 2 pillars (short)
#     mask_row_center = tissue_mask.shape[0] / 2.0
#     mask_col_center = tissue_mask.shape[1] / 2.0
#     row_fraction_offset = np.abs(centroid_row - mask_row_center) / tissue_mask.shape[0]
#     col_fraction_offset = np.abs(centroid_col - mask_col_center) / tissue_mask.shape[1]
#     if row_fraction_offset > 0.25 or col_fraction_offset > 0.25:
#         is_broken = True
#         return is_broken
#     # test if broken via lack of quad symmetry (on 3 pillars)
#     # create four quadrants based on the center of the FOV
#     min_row = 0
#     min_col = 0
#     max_row = tissue_mask.shape[0]
#     max_col = tissue_mask.shape[1]
#     mid_row = int(tissue_mask.shape[0] / 2.0)
#     mid_col = int(tissue_mask.shape[1] / 2.0)
#     Q1_area = np.sum(tissue_mask[min_row:mid_row, min_col:mid_col])
#     Q2_area = np.sum(tissue_mask[mid_row:max_row, min_col:mid_col])
#     Q3_area = np.sum(tissue_mask[mid_row:max_row, mid_col:max_col])
#     Q4_area = np.sum(tissue_mask[min_row:mid_row, mid_col:max_col])
#     # compare Q1 to Q3 ---- and ------ Q2 to Q4
#     min_Q13 = np.min([Q1_area, Q3_area])
#     max_Q13 = np.max([Q1_area, Q3_area])
#     min_Q24 = np.min([Q2_area, Q4_area])
#     max_Q24 = np.max([Q2_area, Q4_area])
#     rat_Q13 = min_Q13 / max_Q13
#     rat_Q24 = min_Q24 / max_Q24
#     if rat_Q13 < 0.75 or rat_Q24 < 0.75:
#         is_broken = True
#         return is_broken
#     return is_broken


def check_broken_tissue_all(
    tissue_mask_list: List,
    wound_mask_list: List = [],
    compare_orig: bool = False,
    zoom_type: int = 2,
    pillar_mask_list:List=None) -> List:
    """Given a tissue mask list. Will return a list of booleans specifying if tissue is broken."""
    
    if pillar_mask_list and len(pillar_mask_list) == 0:
        pillar_mask_list=None
    
    is_broken_list = []
    for kk in range(0, len(tissue_mask_list)):
        tissue_mask = tissue_mask_list[kk]
        if len(wound_mask_list) > 0:
            wound_mask = wound_mask_list[kk]
        if zoom_type == 2 and pillar_mask_list==None:
            if compare_orig:
                is_broken = check_broken_tissue(tissue_mask, tissue_mask_list[0])
            else:
                is_broken = check_broken_tissue(tissue_mask)
        elif zoom_type == 2 and pillar_mask_list:
            if compare_orig:
                is_broken = check_broken_tissue_with_pillars(tissue_mask, pillar_mask_list,tissue_mask_list[0])
            else:
                is_broken = check_broken_tissue_with_pillars(tissue_mask, pillar_mask_list)
        elif zoom_type == 1:
            # check_broken_tissue_zoom(tissue_mask: np.ndarray, wound_mask: np.ndarray, tissue_mask_orig: np.ndarray, wound_mask_orig: np.ndarray)
            # idea for future -->
            # average the first two masks?? -- for frames 0 + 1
            is_broken = check_broken_tissue_zoom(tissue_mask, wound_mask, tissue_mask_list[0], wound_mask_list[0])
        is_broken_list.append(is_broken)
    return is_broken_list


def shrink_bounding_box(min_row: int, min_col: int, max_row: int, max_col: int, shrink_factor: Union[int, float]) -> tuple:
    """Will return a shrunken bounding box."""
    row_range = max_row - min_row
    col_range = max_col - min_col
    row_delta = int(row_range * shrink_factor * 0.5)
    col_delta = int(col_range * shrink_factor * 0.5)
    min_row_new = min_row + row_delta
    max_row_new = max_row - row_delta
    min_col_new = min_col + col_delta
    max_col_new = max_col - col_delta
    return (min_row_new, min_col_new, max_row_new, max_col_new)


def check_inside_box(region: object, bbox1: tuple, bbox2: tuple) -> bool:
    """Will check if a region is inside an admissible bounding box."""
    _, _, _, cr, cc, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(region)
    inside_bbox = (min_row > bbox1[0]) and (min_col > bbox1[1]) and (max_row < bbox1[2]) and (max_col < bbox1[3])
    centroid_inside_bbox = (cr > bbox2[0]) and (cc > bbox2[1]) and (cr < bbox2[2]) and (cc < bbox2[3])
    if inside_bbox and centroid_inside_bbox:
        return True
    else:
        return False


def check_wound_closed_zoom(tissue_mask: np.ndarray, wound_region: object) -> bool:
    if wound_region is None:
        return True
    tissue_object = seg.get_region_props(tissue_mask)[0]
    _, _, _, _, _, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(tissue_object)
    # contract the bounding box to include only the admissible wound area
    shrink_factor = 0.0
    # whole wound must be inside this box
    bbox_outer = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    shrink_factor = 0.5
    # centroid of the wound must be inside this
    bbox_inner = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    # make checks on the wound
    is_inside_box = check_inside_box(wound_region, bbox_outer, bbox_inner)
    min_area = (tissue_mask.shape[0] / 100) ** 2.0
    is_large_enough = seg.check_above_min_size(wound_region, min_area)
    if is_inside_box and is_large_enough:
        return False
    else:
        return True


# def check_wound_closed_zoom(tissue_mask: np.ndarray, wound_region: object) -> bool:
#     # use the tissue mask to define an admissible wound region
#     # check to make sure the wound is within that region
#     # check to make sure the wound is above a certain size
#     # get tissue mask bounding box
#     if wound_region is None:
#         return True
#     tissue_object = seg.get_region_props(tissue_mask)[0]
#     _, _, _, _, _, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(tissue_object)
#     # contract the bounding box to include only the admissible wound area
#     shrink_factor = 0.25
#     # whole wound must be inside this box
#     bbox_outer = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
#     shrink_factor = 0.5
#     # centroid of the wound must be inside this
#     bbox_inner = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
#     # make checks on the wound
#     is_inside_box = check_inside_box(wound_region, bbox_outer, bbox_inner)
#     min_area = (tissue_mask.shape[0] / 100) ** 2.0
#     is_large_enough = seg.check_above_min_size(wound_region, min_area)
#     if is_inside_box and is_large_enough:
#         return False
#     else:
#         return True


def check_wound_closed(tissue_mask: np.ndarray, wound_region: object):
    if wound_region is None:
        return True
    # convert wound_region to wound mask
    wound_coords = seg.region_to_coords([wound_region])
    wound_mask = seg.coords_to_mask(wound_coords, tissue_mask)
    # create tissue_mask_robust
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    # get rotation information
    center_row_orig, center_col_orig, rot_mat, ang, vec = get_rotation_info(center_row_input=None, center_col_input=None, vec_input=None, mask=tissue_mask_robust)
    # rotate tissue
    rot_tissue_mask = rot_image(tissue_mask_robust, center_row_orig, center_col_orig, ang)
    # rotate wound
    rot_wound_mask = rot_image(wound_mask, center_row_orig, center_col_orig, ang)
    # perform checks -- wound inside the center of the tissue + large enough
    tissue_object = seg.get_region_props(rot_tissue_mask)[0]
    wound_list = seg.get_region_props(rot_wound_mask)
    if len(wound_list) == 0:
        return True
    else:
        wound_object = wound_list[0]
    _, _, _, _, _, _, (min_row, min_col, max_row, max_col), _ = seg.extract_region_props(tissue_object)
    shrink_factor = 0.25
    bbox_outer = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    shrink_factor = 0.5
    bbox_inner = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    # make checks on the wound
    is_inside_box = check_inside_box(wound_object, bbox_outer, bbox_inner)
    min_area = mask_to_area(tissue_mask_robust) * 1.0 / 625.0
    is_large_enough = seg.check_above_min_size(wound_region, min_area)
    if is_inside_box and is_large_enough:
        return False
    else:
        return True


def check_wound_closed_all(tissue_mask_list: List, wound_region_list: List, zoom_fcn_idx: int) -> List:
    """Given tissue and wound lists. Will return a list if all tissues are closed."""
    check_wound_closed_list = []
    for kk in range(0, len(tissue_mask_list)):
        if zoom_fcn_idx == 1:
            is_closed = check_wound_closed_zoom(tissue_mask_list[kk], wound_region_list[kk])
        elif zoom_fcn_idx == 2:
            is_closed = check_wound_closed(tissue_mask_list[kk], wound_region_list[kk])
        check_wound_closed_list.append(is_closed)
    return check_wound_closed_list


# def wound_parameters_all(wound_region_list: List) -> List:
#     """Given a wound regions list. Will return wound properties list."""
#     area_list = []
#     axis_major_length_list = []
#     axis_minor_length_list = []
#     for wound_region in wound_region_list:
#         area, axis_major_length, axis_minor_length, _, _, _, _, _ = seg.extract_region_props(wound_region)
#         area_list.append(area)
#         axis_major_length_list.append(axis_major_length)
#         axis_minor_length_list.append(axis_minor_length)
#     return area_list, axis_major_length_list, axis_minor_length_list
def wound_parameters_all(img: np.ndarray, contour_list: List) -> List:
    area_list = []
    axis_major_length_list = []
    axis_minor_length_list = []
    for contour in contour_list:
        wound_region = seg.contour_to_region(img, contour)
        area, axis_major_length, axis_minor_length, _, _, _, _, _ = seg.extract_region_props(wound_region)
        if area is None:
            area_list.append(0)
        else:
            area_list.append(area)
        if axis_major_length is None:
            axis_major_length_list.append(0)
        else:
            axis_major_length_list.append(axis_major_length)
        if axis_minor_length is None:
            axis_minor_length_list.append(0)
        else:
            axis_minor_length_list.append(axis_minor_length)
    return area_list, axis_major_length_list, axis_minor_length_list


def mask_to_area(mask: np.ndarray, pix_to_microns: Union[float, int] = 1):
    """Given a mask and pixel to micron conversions. Returns wound area."""
    area = np.sum(mask)
    area_scaled = area * pix_to_microns * pix_to_microns
    return area_scaled

# convert tissue mask, pt1_0, pt1_1, pt2_0, pt2_1 --> area, kappa_1, kappa_2, tissue_contour
#       --> get closest point on the contour to compute kappa


def get_contour_distance_across(
    c_idx: int,
    contour: np.ndarray,
    num_pts_contour: int,
    include_idx: List,
    tolerence_check: Union[float, int] = 0.2
) -> Union[float, int]:
    """Given a contour point and associated information. Will return the distance across the contour."""
    opposite_point = c_idx + int(num_pts_contour / 2)
    min_opposite = opposite_point - int(tolerence_check * num_pts_contour)
    max_opposite = opposite_point + int(tolerence_check * num_pts_contour)
    x0 = []
    x1 = []
    val_list = []
    for val_ix in range(min_opposite, max_opposite):
        val = ix_loop(val_ix, num_pts_contour)
        x0.append(contour[val, 0])
        x1.append(contour[val, 1])
        val_list.append(val)
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    x0_pt = contour[c_idx, 0]
    x1_pt = contour[c_idx, 1]
    dist_list = []
    for kk in range(0, x0.shape[0]):
        # if math.isinf(x0[kk]) or math.isinf(x1[kk]) or math.isinf(x0_pt) or math.isinf(x1_pt):
        #     dist_list.append(math.inf)
        # else:
        #     dist_list.append((x0[kk] - x0_pt) ** 2.0 + (x1[kk] - x1_pt) ** 2.0) ** 0.5
        if val_list[kk] in include_idx and c_idx in include_idx:
            dist = seg.compute_distance(x0[kk], x1[kk], x0_pt, x1_pt)
            dist_list.append(dist)
        else:
            dist_list.append(math.inf)
    dist_array = np.asarray(dist_list)
    ix = np.argmin(dist_array)
    distance_opposite = dist_array[ix]
    ix_opposite = val_list[ix]
    return distance_opposite, ix_opposite


def get_contour_distance_across_all(contour: np.ndarray, include_idx: List) -> np.ndarray:
    """Given a contour and an include index. Will compute the distance across."""
    num_pts_contour = contour.shape[0]
    tolerence_check = 0.2
    distance_all = []
    ix_all = []
    for kk in range(0, num_pts_contour):
        if kk in include_idx:
            dist, ix = get_contour_distance_across(kk, contour, num_pts_contour, include_idx, tolerence_check)
        else:
            dist = math.inf
            ix = 0
        distance_all.append(dist)
        ix_all.append(ix)
    distance_all = np.asarray(distance_all)
    ix_all = np.asarray(ix_all)
    return distance_all, ix_all


def include_points_contour(
    contour: np.ndarray,
    centroid_row: Union[int, float],
    centroid_col: Union[int, float],
    tissue_axis_major_length: Union[int, float],
    tissue_axis_minor_length: Union[int, float]
) -> List:
    """Given information about the tissue contour. Will return included points for tissue width."""
    # radius = 0.25 * (tissue_axis_major_length + tissue_axis_minor_length)
    radius = 0.5 * tissue_axis_minor_length
    include_idx = []
    for kk in range(0, contour.shape[0]):
        dist = seg.compute_distance(contour[kk, 0], contour[kk, 1], centroid_row, centroid_col)
        if dist < radius:
            include_idx.append(kk)
    return include_idx


def get_contour_width(
    contour: np.ndarray,
    centroid_row: Union[int, float],
    centroid_col: Union[int, float],
    tissue_axis_major_length: Union[int, float],
    tissue_axis_minor_length: Union[int, float],
    orientation: Union[int, float]
) -> Union[float, int]:
    """Given a contour. Will compute minimum distance across and location of minimum. This is the width."""
    include_idx = include_points_contour(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length)
    # contour_clipped_0 = clip_contour(contour, centroid_row, centroid_col, orientation, tissue_axis_major_length, tissue_axis_minor_length)
    # contour_clipped = clip_contour(contour_clipped_0, centroid_row, centroid_col, orientation + np.pi / 2.0, tissue_axis_major_length, tissue_axis_minor_length)
    # contour_clipped_penalized = get_penalized(contour, contour_clipped)
    # distance_all, ix_all = get_contour_distance_across_all(contour_clipped_penalized)
    distance_all, ix_all = get_contour_distance_across_all(contour, include_idx)
    idx_a = np.argmin(distance_all)
    width = distance_all[idx_a]
    idx_b = ix_all[idx_a]
    return width, idx_a, idx_b


def select_zoom_function(
    input_dict: dict
) -> int:
    """Given setup information. Will return which segmentation function to run."""
    if input_dict["zoom_type"] == 1:
        return 1
    elif input_dict["zoom_type"] == 2:
        return 2
    elif input_dict["zoom_type"] == 3:
        return 3
