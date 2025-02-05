import alphashape
import cv2
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point
from skimage import exposure, img_as_ubyte
from typing import List, Union
from woundcompute import segmentation as seg


def get_tracking_param_dicts() -> dict:
    """Will return dictionaries specifying the feature parameters and tracking parameters.
    In future, these may vary based on version."""
    # feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=7, blockSize=7)
    # BEST: feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    # need to make these adaptive to the resolution of the image
    # feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=20, blockSize=20)
    feature_params = dict(maxCorners=10000, qualityLevel=0.01, minDistance=10, blockSize=10)
    # window = 50
    window = 100
    lk_params = dict(winSize=(window, window), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return feature_params, lk_params


def get_tracking_param_dicts_pillar() -> dict:
    """Will return dictionaries specifying the feature parameters and tracking parameters.
    In future, these may vary based on version."""
    # feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=7, blockSize=7)
    # BEST: feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    # need to make these adaptive to the resolution of the image
    # feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=20, blockSize=20)
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=3, blockSize=3)
    # window = 50
    window = 100
    lk_params = dict(winSize=(window, window), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return feature_params, lk_params


def bool_to_uint8(arr_bool: np.ndarray) -> np.ndarray:
    """Given a boolean array. Will return a uint8 array."""
    arr_uint8 = (1. * arr_bool).astype('uint8')
    return arr_uint8


def uint16_to_uint8(img_16: np.ndarray) -> np.ndarray:
    """Given a uint16 image. Will normalize + rescale and convert to uint8."""
    img_8 = img_as_ubyte(exposure.rescale_intensity(img_16))
    return img_8


def uint16_to_uint8_all(img_list: List) -> List:
    """Given an image list of uint16. Will return the same list all as uint8."""
    uint8_list = []
    for img in img_list:
        img8 = uint16_to_uint8(img)
        uint8_list.append(img8)
    return uint8_list


def mask_to_track_points(img_uint8: np.ndarray, mask: np.ndarray, feature_params: dict) -> np.ndarray:
    """Given an image and a mask. Will return the good features to track within the mask region."""
    # ensure that the mask is uint8
    mask_uint8 = bool_to_uint8(mask)
    track_points_0 = cv2.goodFeaturesToTrack(img_uint8, mask=mask_uint8, **feature_params)
    return track_points_0


def track_one_step(img_uint8_0: np.ndarray, img_uint8_1: np.ndarray, track_points_0: np.ndarray, lk_params: dict):
    """Given img_0, img_1, tracking points p0, and tracking parameters.
    Will return the tracking points new location. Note that for now standard deviation and error are ignored."""
    track_points_1, _, _ = cv2.calcOpticalFlowPyrLK(img_uint8_0, img_uint8_1, track_points_0, None, **lk_params)
    return track_points_1


def get_order_track(len_img_list: int, is_forward: bool) -> List:
    """Given the length of the image list. Will return the order of tracking frames"""
    if is_forward:
        return list(range(0, len_img_list))
    else:
        return list(range(len_img_list - 1, -1, -1))


def track_all_steps(img_list_uint8: List, mask: np.ndarray, order_list: List, is_pillar: bool = False) -> np.ndarray:
    """Given the image list, mask, and order. Will run tracking through the whole img list in order.
    Note that the returned order of tracked points will match order_list."""
    if is_pillar:
        feature_params, lk_params = get_tracking_param_dicts_pillar()
    else:
        feature_params, lk_params = get_tracking_param_dicts()
    img_0 = img_list_uint8[order_list[0]]
    track_points = mask_to_track_points(img_0, mask, feature_params)
    num_track_pts = track_points.shape[0]
    num_imgs = len(img_list_uint8)
    tracker_x = np.zeros((num_track_pts, num_imgs))
    tracker_y = np.zeros((num_track_pts, num_imgs))
    for kk in range(0, num_imgs - 1):
        tracker_x[:, kk] = track_points[:, 0, 0]
        tracker_y[:, kk] = track_points[:, 0, 1]
        ix_0 = order_list[kk]
        ix_1 = order_list[kk + 1]
        img_0 = img_list_uint8[ix_0]
        img_1 = img_list_uint8[ix_1]
        track_points = track_one_step(img_0, img_1, track_points, lk_params)
    tracker_x[:, kk + 1] = track_points[:, 0, 0]
    tracker_y[:, kk + 1] = track_points[:, 0, 1]
    return tracker_x, tracker_y


def get_unique(numbers):
    """Helper function for getting unique values in a list."""
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers


def wound_mask_from_points(
    frame_0_mask: np.ndarray,
    tracker_x: np.ndarray,
    tracker_y: np.ndarray,
    wound_contour: np.ndarray,
    alpha_assigned: bool = True,
    assigned_alpha: float = 0.01
) -> np.ndarray:
    """Given tracking results and frame 0 wound contour. Will create wound masks based on the alphashape of the close track points."""
    num_pts = tracker_x.shape[0]
    final_xy = np.zeros((num_pts, 2))
    final_xy[:, 0] = tracker_x[:, -1]
    final_xy[:, 1] = tracker_y[:, -1]
    initial_xy = np.zeros((num_pts, 2))
    initial_xy[:, 0] = tracker_x[:, 0]
    initial_xy[:, 1] = tracker_y[:, 0]
    # find the edge points -- initial tracking points closest to the edge of the wound
    edge_pts = []
    for kk in range(0, wound_contour.shape[0]):
        x = wound_contour[kk, 1]  # CHANGED
        y = wound_contour[kk, 0]  # CHANGED
        dist = distance.cdist(np.asarray([[x, y]]), initial_xy, 'euclidean')
        argmin = np.argmin(dist)
        edge_pts.append(argmin)
    edge_pts = get_unique(edge_pts)
    # convert the edge points into an alpha shape
    num_pts = len(edge_pts)
    points_2d_initial = []
    points_2d_final = []
    for kk in range(0, num_pts):
        ix = edge_pts[kk]
        points_2d_initial.append((initial_xy[ix, 0], initial_xy[ix, 1]))
        points_2d_final.append((final_xy[ix, 0], final_xy[ix, 1]))
    if alpha_assigned:
        alpha_shape_initial = alphashape.alphashape(points_2d_initial, assigned_alpha)
        alpha_shape_final = alphashape.alphashape(points_2d_final, assigned_alpha)
    else:  # this will automatically select alpha, however it can be slow
        alpha_shape_initial = alphashape.alphashape(points_2d_initial)
        alpha_shape_final = alphashape.alphashape(points_2d_final)
    # convert the alpha shape into wound masks
    mask_wound_initial = np.zeros(frame_0_mask.shape)
    mask_wound_final = np.zeros(frame_0_mask.shape)
    # convert to a mask
    for kk in range(0, frame_0_mask.shape[0]):
        for jj in range(0, frame_0_mask.shape[1]):
            if alpha_shape_initial.contains(Point(jj, kk)) is True:
                mask_wound_initial[kk, jj] = 1
            if alpha_shape_final.contains(Point(jj, kk)) is True:
                mask_wound_final[kk, jj] = 1
    mask_wound_initial = mask_wound_initial > 0
    mask_wound_final = mask_wound_final > 0
    return mask_wound_initial, mask_wound_final


def wound_areas_from_points(
    frame_0_mask: np.ndarray,
    tracker_x: np.ndarray,
    tracker_y: np.ndarray,
    wound_contour: np.ndarray,
    alpha_assigned: bool = True,
    assigned_alpha: float = 0.01
) -> np.ndarray:
    """Given tracking results and frame 0 wound contour. Will create wound masks based on the alphashape of the close track points."""
    num_pts = tracker_x.shape[0]
    initial_xy = np.zeros((num_pts, 2))
    initial_xy[:, 0] = tracker_x[:, 0]
    initial_xy[:, 1] = tracker_y[:, 0]
    wound_masks_all = []
    # find the edge points -- initial tracking points closest to the edge of the wound
    edge_pts = []
    for kk in range(0, wound_contour.shape[0]):
        x = wound_contour[kk, 1]  # CHANGED
        y = wound_contour[kk, 0]  # CHANGED
        dist = distance.cdist(np.asarray([[x, y]]), initial_xy, 'euclidean')
        argmin = np.argmin(dist)
        edge_pts.append(argmin)
    edge_pts = get_unique(edge_pts)
    # convert the edge points into an alpha shape
    num_edge_pts = len(edge_pts)
    wound_area_list = []
    for jj in range(0, tracker_x.shape[1]):
        final_xy = np.zeros((num_pts, 2))
        final_xy[:, 0] = tracker_x[:, jj]
        final_xy[:, 1] = tracker_y[:, jj]
        points_2d_final = []
        for kk in range(0, num_edge_pts):
            ix = edge_pts[kk]
            points_2d_final.append((final_xy[ix, 0], final_xy[ix, 1]))
        if alpha_assigned:
            alpha_shape_final = alphashape.alphashape(points_2d_final, assigned_alpha)
        else:  # this will automatically select alpha, however it can be slow
            alpha_shape_final = alphashape.alphashape(points_2d_final)
        # convert the alpha shape into wound masks
        mask_wound_final = np.zeros(frame_0_mask.shape)
        # convert to a mask
        for kk in range(0, frame_0_mask.shape[0]):
            for jj in range(0, frame_0_mask.shape[1]):
                if alpha_shape_final.contains(Point(jj, kk)) is True:
                    mask_wound_final[kk, jj] = 1
        mask_wound_final = mask_wound_final > 0
        wound_area_list.append(np.sum(mask_wound_final))
        wound_masks_all.append(mask_wound_final)
    return wound_area_list, wound_masks_all


def perform_tracking(frame_0_mask: np.ndarray, img_list: List, include_reverse: bool = True, wound_contour: np.ndarray = None):
    """Given an initial mask and all images. Will perform forward and reverse (optional) tracking."""
    # convert img_list to all uint8 images
    img_list_uint8 = uint16_to_uint8_all(img_list)
    len_img_list = len(img_list_uint8)
    # perform forward tracking
    is_forward = True
    order_list = get_order_track(len_img_list, is_forward)
    tracker_x, tracker_y = track_all_steps(img_list_uint8, frame_0_mask, order_list)
    # create wound mask
    _, frame_final_mask = wound_mask_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour)
    if include_reverse:
        # perform reverse tracking
        is_forward = False
        order_list = get_order_track(len_img_list, is_forward)
        tracker_x_reverse, tracker_y_reverse = track_all_steps(img_list_uint8, frame_final_mask, order_list)
        # reverse array
        tracker_x_reverse = np.flip(tracker_x_reverse, axis=1)
        tracker_y_reverse = np.flip(tracker_y_reverse, axis=1)
    else:
        tracker_x_reverse = None
        tracker_y_reverse = None
    wound_area_list, wound_masks_all = wound_areas_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour)
    return frame_final_mask, tracker_x, tracker_y, tracker_x_reverse, tracker_y_reverse, wound_area_list, wound_masks_all


def template_match_tracking(img_masked: np.ndarray, template: np.ndarray) -> Union[np.ndarray, float]:
    # mask the image based on quadrant
    h, w = template.shape
    res = cv2.matchTemplate(img_masked, template, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x0, y0 = top_left
    xf, yf = bottom_right
    center_pt = np.zeros((2,))
    center_pt[0] = (xf - x0) / 2 + x0
    center_pt[1] = (yf - y0) / 2 + y0
    return np.array([x0, y0, xf, yf]), center_pt


def template_track_all_steps(img_list_uint8, pillar_mask, order_list):
    time_0 = order_list[0]
    template = seg.mask_to_template(img_list_uint8[time_0], pillar_mask)
    tracker_x = []
    tracker_y = []
    for kk in range(0, len(img_list_uint8)):
        img = img_list_uint8[order_list[kk]]
        img_masked = seg.mask_img_for_pillar_track(img, pillar_mask)
        _, center_pt = template_match_tracking(img_masked, template)
        tracker_x.append(center_pt[0])
        tracker_y.append(center_pt[1])
    tracker_x = np.asarray(tracker_x)
    tracker_y = np.asarray(tracker_y)
    return tracker_x, tracker_y


def perform_pillar_tracking(pillar_mask_list: List, img_list: List, version: int = 2):
    # convert img_list to all uint8 images
    img_list_uint8 = uint16_to_uint8_all(img_list)
    len_img_list = len(img_list_uint8)
    is_forward = True
    order_list = get_order_track(len_img_list, is_forward)
    # for each pillar mask perform tracking
    num_pillars = len(pillar_mask_list)
    avg_disp_all_x = np.zeros((len_img_list, num_pillars))
    avg_disp_all_y = np.zeros((len_img_list, num_pillars))
    for kk in range(0, num_pillars):
        # old version
        if version == 1:
            tracker_x, tracker_y = track_all_steps(img_list_uint8, pillar_mask_list[kk], order_list)
        else:
            tracker_x, tracker_y = template_track_all_steps(img_list_uint8, pillar_mask_list[kk], order_list)
        # consolidate average displacement
        if version == 1:
            tracker_x_avg = np.mean(tracker_x, axis=0)
            tracker_y_avg = np.mean(tracker_y, axis=0)
            avg_disp_all_x[:, kk] = tracker_x_avg
            avg_disp_all_y[:, kk] = tracker_y_avg
        else:
            avg_disp_all_x[:, kk] = tracker_x
            avg_disp_all_y[:, kk] = tracker_y
    return avg_disp_all_x, avg_disp_all_y
