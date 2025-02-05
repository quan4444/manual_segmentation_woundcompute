import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, img_as_ubyte
from skimage.segmentation import clear_border
from skimage import measure, morphology
from skimage.filters import gabor_kernel, threshold_otsu, threshold_multiotsu, rank
from skimage.measure import label, regionprops
from typing import List, Union
from woundcompute import compute_values as com


def uint16_to_uint8(img_16: np.ndarray) -> np.ndarray:
    '''Given a uint16 image. Will normalize + rescale and convert to uint8.'''
    img_8 = img_as_ubyte(exposure.rescale_intensity(img_16))
    return img_8


def thresh_img_local(img: np.ndarray, radius: int = 135) -> np.ndarray:
    '''Given an uint16 image. Will return a binary image based on local otsu thresholding.'''
    img = img.astype("uint16")
    selem = morphology.disk(radius)
    img = uint16_to_uint8(img)
    local_otsu = rank.otsu(img, selem)
    binary_img = img > local_otsu
    return binary_img


def apply_median_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the median filter applied by scipy"""
    filtered_array = ndimage.median_filter(array, filter_size)
    return filtered_array


def apply_gaussian_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the gaussian filter applied by scipy"""
    filtered_array = ndimage.gaussian_filter(array, filter_size)
    return filtered_array


def compute_otsu_thresh(array: np.ndarray) -> Union[float, int]:
    """Given an image array. Will return the otsu threshold applied by skimage."""
    thresh = threshold_otsu(array)
    return thresh


def apply_otsu_thresh(array: np.ndarray) -> np.ndarray:
    """Given an image array. Will return a boolean numpy array with an otsu threshold applied."""
    thresh = compute_otsu_thresh(array)
    thresh_img = array > thresh
    return thresh_img


def get_region_props(array: np.ndarray) -> List:
    """Given a binary image. Will return the list of region props."""
    label_image = label(array)
    region_props = regionprops(label_image)
    return region_props


def get_largest_regions(region_props: List, num_regions: int = 3) -> List:
    """Given a list of region properties. Will return a list of the num_regions largest regions.
    If there are fewer than num_regions regions, will return all regions."""
    area_list = []
    for region in region_props:
        area_list.append(region.area)
    ranked = np.argsort(area_list)[::-1]
    num_to_return = np.min([len(ranked), num_regions])
    regions_list = []
    for kk in range(0, num_to_return):
        idx = ranked[kk]
        regions_list.append(region_props[idx])
    return regions_list


def get_regions_not_touching_bounds(region_props: List, img_shape: tuple) -> List:
    """Given a list of region properties. Will return a list of all region properties not touching the edges of the domain."""
    new_regions = []
    for region in region_props:
        coords = region.coords
        if 0 in coords:
            continue
        if img_shape[0] - 1 in coords[:, 0]:
            continue
        if img_shape[1] - 1 in coords[:, 1]:
            continue
        new_regions.append(region)
    return new_regions


def get_roundest_regions(region_props: List, num_regions: int = 3) -> List:
    """Given a list of region properties. Will return the num_regions roundest regions.
    If there are fewer than num_regions regions, will return all regions.
    For eccentricity, 0 = circle, 1 = more elliptical"""
    eccentricity_list = []
    for region in region_props:
        eccentricity = region.eccentricity
        eccentricity_list.append(eccentricity)
    ranked = np.argsort(eccentricity_list)
    num_to_return = np.min([len(ranked), num_regions])
    regions_list = []
    for kk in range(0, num_to_return):
        idx = ranked[kk]
        regions_list.append(region_props[idx])
    return regions_list


# def get_longest_regions(region_props: List, num_regions: int = 3) -> List:
#     """Given a list of region properties. Will return the num_regions longest regions.
#     If there are fewer than num_regions regions, will return all regions.
#     For eccentricity, 0 = circle, 1 = more elliptical"""
#     eccentricity_list = []
#     for region in region_props:
#         eccentricity = region.eccentricity
#         eccentricity_list.append(eccentricity)
#     print(eccentricity_list)
#     ranked = np.argsort(eccentricity_list)
#     num_to_return = np.min([len(ranked), num_regions])
#     regions_list = []
#     for kk in range(0, num_to_return):
#         idx = ranked[len(ranked) - kk - 1]
#         regions_list.append(region_props[idx])
#     return regions_list


def get_domain_center(array: np.ndarray) -> Union[int, float]:
    """Given an array. Will return center (ix_0, ix_1)"""
    center_0 = array.shape[0] / 2.0
    center_1 = array.shape[1] / 2.0
    return center_0, center_1


def compute_distance(
    a0: Union[int, float],
    a1: Union[int, float],
    b0: Union[int, float],
    b1: Union[int, float]
) -> Union[int, float]:
    """Given two points. Will return distance between them."""
    dist = ((a0 - b0)**2.0 + (a1 - b1)**2.0)**0.5
    return dist


def get_closest_region(
    regions_list: List,
    loc_0: Union[int, float],
    loc_1: Union[int, float]
) -> object:
    """Given a list of region properties. Will return the object closest to location."""
    center_dist = []
    for region in regions_list:
        centroid = region.centroid
        region_0 = centroid[0]
        region_1 = centroid[1]
        dist = compute_distance(region_0, region_1, loc_0, loc_1)
        center_dist.append(dist)
    ix = np.argmin(center_dist)
    return regions_list[ix]


def get_closest_regions(
    regions_list: List,
    loc_0: Union[int, float],
    loc_1: Union[int, float],
    num_closest_regions: int=1
) -> object:
    """Given a list of region properties. Will return the object closest to location."""
    dist_list = []
    for cur_region in regions_list:
        centroid = cur_region.centroid
        region_0,region_1 = centroid
        dist = compute_distance(region_0,region_1,loc_0,loc_1)
        dist_list.append(dist)

    dist_arr = np.array(dist_list)
    num_closest_regions = min(len(dist_arr)-1, num_closest_regions)
    indices = np.argpartition(dist_arr, num_closest_regions)[:num_closest_regions]

    closest_regions = []
    for ind in indices:
        closest_regions.append(regions_list[ind])
    return closest_regions


def extract_region_props(region_props: object) -> Union[float, np.ndarray]:
    """Given region properties from skimage.measure.regionprops.
    Will return the values of relevant properties.
    See: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    """
    if region_props is None:
        return None, None, None, None, None, None, (None, None, None, None), None
    else:
        area = region_props.area
        axis_major_length = region_props.axis_major_length
        axis_minor_length = region_props.axis_minor_length
        centroid = region_props.centroid
        centroid_row = centroid[0]
        centroid_col = centroid[1]
        coords = region_props.coords
        bbox = region_props.bbox
        orientation = region_props.orientation
        return area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation


def region_to_coords(regions_list: List) -> List:
    """Given regions list. Will return the coordinates of all regions in the list."""
    coords_list = []
    for region in regions_list:
        coords = extract_region_props(region)[5]
        coords_list.append(coords)
    return coords_list


def coords_to_mask(coords_list: List, array: np.ndarray) -> np.ndarray:
    """Given coordinates and template array. Will turn coordinates into a binary mask."""
    mask = np.zeros(array.shape)
    for coords in coords_list:
        for kk in range(0, coords.shape[0]):
            mask[coords[kk, 0], coords[kk, 1]] = 1
    return mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return an inverted mask."""
    invert_mask = mask == 0
    return invert_mask


def coords_to_inverted_mask(coords_list: List, array: np.ndarray) -> np.ndarray:
    """Given coordinates and template array. Will turn coordinates into an inverted binary mask."""
    mask = coords_to_mask(coords_list, array)
    inverted_mask = invert_mask(mask)
    return inverted_mask


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Given a mask of the wound. Will return contour around wound."""
    # filter_size = 1
    # blur = apply_gaussian_filter(mask, filter_size)
    # contours = measure.find_contours(blur, 0.95)
    contours = measure.find_contours(mask, 0.75)
    contour_list = []
    contour_leng = []
    for cont in contours:
        contour_list.append(cont)
        contour_leng.append(cont.shape[0])
    if len(contour_list) == 0:
        return None
    else:
        argmax = np.argmax(contour_leng)
        chosen_contour = contour_list[argmax]
        return chosen_contour


def close_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def dilate_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    dilated_array = morphology.binary_dilation(array, footprint)
    return dilated_array


def gabor_filter(array: np.ndarray, theta_range: int = 17, ff_max: int = 3, ff_mult: float = 0.4) -> np.ndarray:
    gabor_all = np.zeros(array.shape)
    for ff in range(0, ff_max):
        frequency = 0.2 + ff * ff_mult
        for tt in range(0, theta_range):
            theta = tt * np.pi / (theta_range - 1)
            # filt_real, _ = gabor(array, frequency=frequency, theta=theta)
            g_kernel = gabor_kernel(frequency=frequency,theta=theta)
            filt_real = ndimage.convolve(array,np.real(g_kernel),mode='reflect',cval=0)
            gabor_all += filt_real
    return gabor_all


def apply_thresh_multiotsu(array: np.ndarray):
    thresholds = threshold_multiotsu(array)
    regions = np.digitize(array, bins=thresholds)
    foreground = regions > 0
    return foreground


def threshold_array(array: np.ndarray, selection_idx: int) -> np.ndarray:
    """Given an image array. Will return a binary array where object = 0, background = 1."""
    if selection_idx == 1:
        """Given a brightfield image array. Will return a binary array where tissue = 0, background = 1."""
        median_filter_size = 5
        array_median = apply_median_filter(array, median_filter_size)
        gaussian_filter_size = 2
        array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
        thresh_img = apply_otsu_thresh(array_gaussian)
        return thresh_img
    elif selection_idx == 2:
        """Given a gfp image array. Will return a binary array where gfp = 0, background = 1."""
        median_filter_size = 5
        array_median = apply_median_filter(array, median_filter_size)
        gaussian_filter_size = 1
        array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
        thresh_img = apply_otsu_thresh(array_gaussian)
        thresh_img_inverted = invert_mask(thresh_img)
        return thresh_img_inverted
    elif selection_idx == 3:
        """Given a phase contrast ph1 image array. Will return a binary array where tissue = 0, background = 1."""
        gabor_all = gabor_filter(array)
        thresh_img = apply_otsu_thresh(gabor_all)
        thresh_img_inverted = invert_mask(thresh_img)
        return thresh_img_inverted
    elif selection_idx == 4:
        """Given a phase contrast ph1 image array. Will return a binary array where tissue = 0, background = 1."""
        gabor_all = gabor_filter(array)
        median_filter_size = 5
        median_applied = apply_median_filter(gabor_all, median_filter_size)
        gaussian_filter_size = 2
        gaussian_applied = apply_gaussian_filter(median_applied, gaussian_filter_size)
        thresh_img = apply_otsu_thresh(gaussian_applied)
        thresh_img_inverted = invert_mask(thresh_img)
        return thresh_img_inverted
    elif selection_idx == 5:
        """Given a gfp image array. Will return a binary array where gfp = 0, background = 1."""
        foreground = apply_thresh_multiotsu(array)
        thresh_img_inverted = invert_mask(foreground)
        return thresh_img_inverted
    else:
        raise ValueError("specified version is not supported")


def threshold_all(img_list: List, threshold_function_idx: int) -> List:
    """Given an image list and function index. Will apply threshold to all images."""
    thresholded_list = []
    for img in img_list:
        thresh_img = threshold_array(img, threshold_function_idx)
        thresholded_list.append(thresh_img)
    return thresholded_list


def preview_thresholding(img: np.ndarray) -> list:
    """Given an image array. Will run all thresholds on the array for preview."""
    thresh_list = []
    idx_list = []
    for kk in range(1, 5):
        thresh_list.append(threshold_array(img, kk))
        idx_list.append(kk)
    return thresh_list, idx_list


def get_mean_center(array: np.ndarray) -> Union[float, int]:
    """Given a mask, return the center of the mask."""
    coords = np.argwhere(array > 0)
    center_0 = np.mean(coords[:, 0])
    center_1 = np.mean(coords[:, 1])
    return center_0, center_1


def isolate_masks(array: np.ndarray, selection_idx: int) -> np.ndarray:
    """Given a binary mask where background = 1. Will return a mask where `tissue' = 1.
    Will return a mask where `wound' = 1."""
    if selection_idx == 1 or selection_idx == 2 or selection_idx == 3 or selection_idx == 4 or selection_idx == 5:
        # select the three largest "background" regions -- side, side, wound
        region_props = get_region_props(array)
        # new approach -> remove all regions that aren't touching the boundaries
        region_props_not_touching = get_regions_not_touching_bounds(region_props, array.shape)
        num_regions = 10  # changed from 3
        regions_largest = get_largest_regions(region_props, num_regions)
        num_regions = 1
        region_not_touching_largest = get_largest_regions(region_props_not_touching, num_regions)
        # identify the wound as the "background" region closest to the center
        array_inverted = invert_mask(array)
        center_0, center_1 = get_mean_center(array_inverted)
        # center_0, center_1 = get_domain_center(array)
        if len(region_not_touching_largest) > 0:
            # create the wound mask
            wound_region = get_closest_region(region_not_touching_largest, center_0, center_1)
            wound_region_coords = region_to_coords([wound_region])
            wound_mask_open = coords_to_mask(wound_region_coords, array)
            wound_mask = close_region(wound_mask_open)
        else:
            wound_mask = np.zeros(array.shape)
            wound_region = None
        # create the tissue mask
        regions_largest_coords = region_to_coords(regions_largest)
        tissue_mask_extra = coords_to_inverted_mask(regions_largest_coords, array)
        region_props = get_region_props(tissue_mask_extra)
        num_regions = 1
        regions_largest = get_largest_regions(region_props, num_regions)
        regions_largest_coords = region_to_coords(regions_largest)
        tissue_mask_open = coords_to_mask(regions_largest_coords, array)
        tissue_mask = close_region(tissue_mask_open)
        return tissue_mask, wound_mask, wound_region
    else:
        raise ValueError("specified version is not supported")


def mask_all(thresh_img_list: List, selection_idx: int) -> List:
    """Given a thresholded image list. Will return masks and wound regions."""
    tissue_mask_list = []
    wound_mask_list = []
    wound_region_list = []
    for thresh_img in thresh_img_list:
        if selection_idx == 4:
            tissue_mask, wound_mask, wound_region = isolate_masks(thresh_img, selection_idx)
        else:
            _, wound_mask, wound_region = isolate_masks(thresh_img, selection_idx)
            tissue_mask, _, _ = isolate_masks(thresh_img, 4)
        tissue_mask_list.append(tissue_mask)
        wound_mask_list.append(wound_mask)
        wound_region_list.append(wound_region)
    return tissue_mask_list, wound_mask_list, wound_region_list


def check_above_min_size(region: object, min_area: Union[int, float]):
    """Will check if region is above a minimum area."""
    wound_area, _, _, _, _, _, _, _ = extract_region_props(region)
    if wound_area > min_area:
        return True
    else:
        return False


def contour_all(wound_mask_list: List) -> List:
    """Given a wound mask list. Will return a contour list."""
    contour_list = []
    for wound_mask in wound_mask_list:
        contour = mask_to_contour(wound_mask)
        contour_list.append(contour)
    return contour_list


def fill_tissue_mask_reconstruction(mask: np.ndarray) -> np.ndarray:
    """Given a tissue mask. Will return a filled tissue mask w/ reconstruction."""
    gaussian_filter_size = 1
    mask_gaussian = apply_gaussian_filter(mask, gaussian_filter_size)
    new_mask = mask_gaussian > 0
    seed = np.copy(new_mask)
    seed[1:-1, 1:-1] = new_mask.max()
    reconstruction_mask = new_mask
    mask_filled = morphology.reconstruction(seed, reconstruction_mask, method='erosion')
    return mask_filled


def insert_borders(mask: np.ndarray, border: int = 10, border_val: int = 0) -> np.ndarray:
    """Given a mask. Will make the borders around it 0."""
    mask[0:border, :] = border_val
    mask[-border:, :] = border_val
    mask[:, 0:border] = border_val
    mask[:, -border:] = border_val
    return mask


def make_tissue_mask_robust(tissue_mask: np.ndarray, wound_mask: np.ndarray, border_val: int = 10) -> np.ndarray:
    """Given a tissue mask and wound mask. Will fill holes in the tissue mask and make it suitable
    for computing the tissue contour etc."""
    tissue_mask_filled_1 = tissue_mask + wound_mask
    tissue_mask_filled_2 = fill_tissue_mask_reconstruction(tissue_mask_filled_1)
    tissue_mask_filled_3 = apply_gaussian_filter(tissue_mask_filled_2, 1) > 0
    tissue_mask_borders = insert_borders(tissue_mask_filled_3, border_val)
    # return largest connected component -- TODO: make sure this doesn't mess up other tests
    regions = get_region_props(tissue_mask_borders)
    largest_regions = get_largest_regions(regions, 1)
    coords = region_to_coords(largest_regions)
    tissue_mask_robust = coords_to_mask(coords, tissue_mask_borders)
    return tissue_mask_robust


def select_threshold_function(
    input_dict: dict,
    is_brightfield: bool,
    is_fluorescent: bool,
    is_ph1: bool
) -> int:
    """Given setup information. Will return which segmentation function to run."""
    if is_brightfield and input_dict["seg_bf_version"] == 1:
        return 1
    elif is_fluorescent and input_dict["seg_fl_version"] == 1:
        return 2
    elif is_ph1 and input_dict["seg_ph1_version"] == 1:
        return 3
    elif is_ph1 and input_dict["seg_ph1_version"] == 2:
        return 4
    elif is_fluorescent and input_dict["seg_fl_version"] == 2:
        return 5
    else:
        raise ValueError("specified version is not supported")


# def get_pillar_mask_list(array: np.ndarray, selection_idx: int):
#     # given the tissue mask, will create pillar masks
#     tissue_mask, wound_mask, wound_region = isolate_masks(array, selection_idx)
#     tissue_mask_filled = tissue_mask + wound_mask
#     tissue_mask_robust = make_tissue_mask_robust(tissue_mask, wound_mask)
#     # isolate pillars
#     tmf_inverted = invert_mask(tissue_mask_filled)
#     region_props = get_region_props(tmf_inverted)
#     num_regions = 6
#     regions_list = get_largest_regions(region_props, num_regions)
#     # get box and rotation of robust tissue mask
#     # 4 corner points are ordered clockwise starting from the point with the highest y (tiebreaker: rightmost)
#     # box is 4 (points) x 2 (dims) numpy array
#     box = com.mask_to_box(tissue_mask_robust)
#     selected_regions = []
#     for kk in range(0, 4):
#         loc_0 = box[kk, 0]
#         loc_1 = box[kk, 1]
#         region = get_closest_region(regions_list, loc_0, loc_1)
#         selected_regions.append(region)
#     pillar_mask_list = []
#     for kk in range(0, 4):
#         coords_list = region_to_coords([selected_regions[kk]])
#         mask = coords_to_mask(coords_list, tissue_mask)
#         pillar_mask_list.append(mask)
#     return pillar_mask_list


def get_pillar_mask_list(img: np.ndarray, selection_idx: int, mask_seg_type: int = 1):
    # get tissue
    thresh_img_tissue = threshold_array(img, selection_idx)
    tissue_mask, wound_mask, wound_region = isolate_masks(thresh_img_tissue, selection_idx)
    tissue_mask_robust = make_tissue_mask_robust(tissue_mask, wound_mask)
    # pillars tend to be dark circles with bright inside
    # img_thresh = apply_otsu_thresh(img)
    if mask_seg_type == 1:
        img_thresh = thresh_img_local(img)
    else:
        img_thresh = apply_otsu_thresh(img)
    region_props = get_region_props(img_thresh)
    # remove larger regions e.g., background
    regions_list = get_regions_not_touching_bounds(region_props, img.shape)
    regions_list = get_largest_regions(regions_list, 20)
    regions_list = get_roundest_regions(regions_list, 6)
    box = com.mask_to_box(tissue_mask_robust)
    selected_regions = []
    for kk in range(0, 4):
        loc_0 = box[kk, 0]
        loc_1 = box[kk, 1]
        region = get_closest_region(regions_list, loc_0, loc_1)
        selected_regions.append(region)
    pillar_mask_list = []
    for kk in range(0, 4):
        coords_list = region_to_coords([selected_regions[kk]])
        mask = coords_to_mask(coords_list, tissue_mask)
        if np.sum(mask) < (img.shape[0] * 0.05) ** 2.0:
            continue
        # dilate mask a little bit to get more edges in there
        filter_size = 2
        mask = apply_gaussian_filter(mask, filter_size)
        mask = mask > 0
        pillar_mask_list.append(mask)
    return pillar_mask_list


# def mask_quadrants_img(frame: np.ndarray, quadrant: int) -> np.ndarray:
#     '''Given frame and quadrant number, mask all other quadrants except the given quadrant.'''
#     frame_h, frame_w = frame.shape
#     mask = np.zeros(frame.shape, dtype=np.uint8)
#     if quadrant == 0:
#         mask_h = int((frame_h//2) * 1.2)
#         mask_w = int((frame_w//2) * 1.2)
#         mask[:mask_h, :mask_w] = 255
#     elif quadrant == 1:
#         mask_h = int((frame_h//2) * 0.8)
#         mask_w = int((frame_w//2) * 1.2)
#         mask[mask_h:, :mask_w] = 255
#     elif quadrant == 2:
#         mask_h = int((frame_h//2) * 1.2)
#         mask_w = int((frame_w//2) * 0.8)
#         mask[:mask_h, mask_w:] = 255
#     elif quadrant == 3:
#         mask_h = int((frame_h//2) * 0.8)
#         mask_w = int((frame_w//2) * 0.8)
#         mask[mask_h:, mask_w:] = 255
#     masked_image = cv2.bitwise_and(frame, frame, mask=mask)
#     return masked_image


def pillar_mask_to_box(img: np.ndarray, pillar_mask: np.ndarray, buffer: int):
    r_min = np.max([0, np.min(np.argwhere(pillar_mask > 0)[:, 0]) - buffer])
    r_max = np.min([img.shape[0], np.max(np.argwhere(pillar_mask > 0)[:, 0]) + buffer + 1])
    c_min = np.max([0, np.min(np.argwhere(pillar_mask > 0)[:, 1]) - buffer])
    c_max = np.min([img.shape[1], np.max(np.argwhere(pillar_mask > 0)[:, 1]) + buffer + 1])
    return r_min, r_max, c_min, c_max


def mask_img_for_pillar_track(img: np.ndarray, pillar_mask: np.ndarray, buffer: int = 50) -> np.ndarray:
    r_min, r_max, c_min, c_max = pillar_mask_to_box(img, pillar_mask, buffer)
    mask = np.zeros(img.shape)
    mask[r_min:r_max, c_min:c_max] = 1
    img_mask = img * mask
    return img_mask.astype("uint8")


def mask_to_template(img: np.ndarray, pillar_mask: np.ndarray, buffer: int = 2):
    r_min, r_max, c_min, c_max = pillar_mask_to_box(img, pillar_mask, buffer)
    template = img[r_min:r_max, c_min:c_max]
    return template


def contour_to_mask(img: np.ndarray, contour: np.ndarray):
    mask = np.zeros(img.shape)
    if contour is None:
        return mask
    else:
        contour_flip_axis = np.flip(contour, axis=1)
        cv2.fillPoly(mask, pts=[np.int32(contour_flip_axis)], color=(255, 0, 0))
        mask = (mask > 0).astype("uint8")
        return mask


def contour_to_mask_all(img:np.ndarray, contour_list:List)->List:
    masks_list = []
    for contour in contour_list:
        mask = contour_to_mask(img,contour)
        masks_list.append(mask)
    return masks_list


def contour_to_region(img: np.ndarray, contour: np.ndarray):
    if contour is None:
        return None
    else:
        mask = contour_to_mask(img, contour)
        region_props = get_region_props(mask)
        wound_region = get_largest_regions(region_props, 1)[0]
        return wound_region


# def sequence_tissue_segment(img_n: np.ndarray, img_np1: np.ndarray):
#     img_sub = img_np1 - img_n
#     import matplotlib.pyplot as plt
#     plt.imshow(img_sub)
#     aa = 44
#     return aa

def pillar_mask_to_rotated_box(mask: np.ndarray,border:int=10) -> np.ndarray:
    """Given a mask. Will return the minimum area bounding rectangle."""
    # insert borders to the mask
    if border > 0:
        mask_mod = insert_borders(mask, border)
    else:
        mask_mod = mask
    # find contour
    mask_mod_one = (mask_mod > 0).astype(np.float64)
    # mask_thresh_blur = ndimage.gaussian_filter(mask_mod_one, 1)
    all_contours = measure.find_contours(mask_mod_one, 0.75)
    all_pts = []
    for cnt in all_contours:
        for jj in range(0, cnt.shape[0]):
            all_pts.append(cnt[jj, :])
    cnts = np.asarray(all_pts).astype(np.int32)
    # find minimum area bounding rectangle
    rect = cv2.minAreaRect(cnts)
    box = np.int0(cv2.boxPoints(rect))
    return box


def compute_unit_vector(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> np.ndarray:
    """Given two 2D points. Will return the unit vector between them"""
    dist = compute_distance(x1, y1, x2, y2)
    vec = np.asarray([(x2 - x1) / dist, (y2 - y1) / dist])
    return vec


def box_to_unit_vec_len_wid(box: np.ndarray) -> np.ndarray:
    """Given the rectangular box. Will compute the unit vector of the longest side."""
    side_1 = compute_distance(box[0, 0], box[0, 1], box[1, 0], box[1, 1])
    side_2 = compute_distance(box[1, 0], box[1, 1], box[2, 0], box[2, 1])
    if side_1 > side_2:
        # side_1 is the long axis
        vec = compute_unit_vector(box[0, 0], box[1, 0], box[0, 1], box[1, 1])
        length = side_1
        width = side_2
    else:
        # side_2 is the long axis
        vec = compute_unit_vector(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
        length = side_2
        width = side_1
    return vec, length, width


def box_to_center_points(box: np.ndarray) -> float:
    """Given the rectangular box. Will compute the center as the midpoint of a diagonal."""
    center_row = np.mean([box[0, 0], box[2, 0]])
    center_col = np.mean([box[0, 1], box[2, 1]])
    return center_row, center_col


def mask_list_to_single_mask(pillar_mask_list: List) -> np.ndarray:
    pillar_mask = pillar_mask_list[0].copy()
    for kk in range(1, len(pillar_mask_list)):
        pillar_mask += pillar_mask_list[kk]
    pillar_mask = (pillar_mask > 0).astype("uint8")
    return pillar_mask


def move_point_closer(pt_0, pt_1, c_0, c_1, factor):
    dist = compute_distance(pt_0, pt_1, c_0, c_1)
    move_dist = dist * factor
    new_pt_0 = pt_0 + (c_0 - pt_0) / dist * move_dist
    new_pt_1 = pt_1 + (c_1 - pt_1) / dist * move_dist
    return new_pt_0, new_pt_1


def shrink_box(box: np.ndarray, scale_factor: float) -> np.ndarray:
    c_0, c_1 = box_to_center_points(box)
    scale_box = np.zeros((4, 2))
    for kk in range(0, 4):
        pt_0 = box[kk, 0]
        pt_1 = box[kk, 1]
        new_pt_0, new_pt_1 = move_point_closer(pt_0, pt_1, c_0, c_1, scale_factor)
        scale_box[kk, 0] = new_pt_0
        scale_box[kk, 1] = new_pt_1
    return scale_box


def area_triangle_3_pts(x0, x1, x2, y0, y1, y2):
    area = np.abs((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2) + (x0 * y2 - x2 * y0)) / 2.0
    return area


def point_in_box(box: np.ndarray, pt_0: float, pt_1: float) -> bool:
    box_tri_1 = area_triangle_3_pts(box[0, 0], box[1, 0], box[2, 0], box[0, 1], box[1, 1], box[2, 1])
    box_tri_2 = area_triangle_3_pts(box[2, 0], box[3, 0], box[0, 0], box[2, 1], box[3, 1], box[0, 1])
    area_box = box_tri_1 + box_tri_2
    area_box_pt = 0
    for kk in range(0, 4):
        ix0 = kk
        if kk == 3:
            ix1 = 0
        else:
            ix1 = kk + 1
        area_box_pt += area_triangle_3_pts(box[ix0, 0], box[ix1, 0], pt_0, box[ix0, 1], box[ix1, 1], pt_1)
    tol = 0.001
    if area_box_pt > (area_box + tol):
        return False
    else:
        return True


def regions_in_box(box: np.ndarray, region: object) -> bool:
    region_coords = region_to_coords([region])[0]
    skip = 10
    region_coords_downsampled = region_coords[::skip, :]
    all_true = True
    for kk in range(0, region_coords_downsampled.shape[0]):
        pt_0 = region_coords_downsampled[kk, 0]
        pt_1 = region_coords_downsampled[kk, 1]
        all_true = all_true and point_in_box(box, pt_0, pt_1)
    return all_true


def regions_in_box_all(box: np.ndarray, region_list: List) -> List:
    regions_keep = []
    for region in region_list:
        check_region = regions_in_box(box, region)
        if check_region is True:
            regions_keep.append(region)
    return regions_keep


def get_most_similar_area_region(
    regions_list,
    prev_area,
    num_regions: int=1
) -> object:
    """Given a list of region properties. Will return the objects closest to area."""
    diff_list = []
    for cur_region in regions_list:
        cur_area = cur_region.area
        dist = abs(cur_area - prev_area)
        diff_list.append(dist)

    diff_arr = np.array(diff_list)
    indices = np.argpartition(diff_arr, num_regions)[:num_regions]

    most_similar_area_regions = []
    for ind in indices:
        most_similar_area_regions.append(regions_list[ind])
    return most_similar_area_regions


def leverage_pillars_for_wound_seg(
    pillar_mask: np.ndarray,
    background_mask: np.asarray,
    wound_region)->List:

    box = pillar_mask_to_rotated_box(pillar_mask)
    scale_factor = 0.35
    box_shrink = shrink_box(box, scale_factor)

    if wound_region:
        prev_frame_cent0,prev_frame_cent1 = wound_region.centroid
        prev_frame_area = wound_region.area
    else:
        center_0, center_1 = box_to_center_points(box)
        prev_frame_cent0 = center_0
        prev_frame_cent1 = center_1
        prev_frame_area = None

    region_props = get_region_props(background_mask)
    region_props_not_touching = get_regions_not_touching_bounds(region_props, background_mask.shape)
    num_regions = 10
    regions_largest = get_largest_regions(region_props_not_touching, num_regions)
    if len(regions_largest) > 0:
        # eliminate regions outside of admissible area
        allowable_regions = regions_in_box_all(box_shrink, regions_largest)
        num_allowable_reg = len(allowable_regions)
        # get region closest to wound center
        if num_allowable_reg > 0:
            if num_allowable_reg == 1: # 1 region, that region is the wound
                wound_region = allowable_regions[0]
            else: # if more than 1 region, find the region with most similar area and position to previous wound
                closest_wound_regions = get_closest_regions(allowable_regions,prev_frame_cent0,prev_frame_cent1,2)
                if len(closest_wound_regions) == 1:
                    wound_region = closest_wound_regions[0]
                elif prev_frame_area:
                    # wound_region_coords = region_to_coords(closest_wound_regions)
                    # combined_mask = coords_to_mask(wound_region_coords,background_mask)
                    # region_props_combined = regionprops(combined_mask.astype(np.uint8))
                    # regions_in_question = closest_wound_regions + region_props_combined
                    wound_region = get_most_similar_area_region(closest_wound_regions,prev_frame_area,1)[0]
                else:
                    wound_region = get_largest_regions(closest_wound_regions, 1)[0]
            wound_region_coords = region_to_coords([wound_region])
            wound_mask_open = coords_to_mask(wound_region_coords, background_mask) # first extraction
            wound_mask = close_region(wound_mask_open) # closing step
        else:
            wound_mask = np.zeros(background_mask.shape)
            wound_region = None
    else:
        wound_mask = np.zeros(background_mask.shape)
        wound_region = None
    # create the tissue mask
    num_regions = 10  # changed from 3
    regions_largest = get_largest_regions(region_props, num_regions)
    regions_largest_coords = region_to_coords(regions_largest)
    tissue_mask_extra = coords_to_inverted_mask(regions_largest_coords, background_mask)
    region_props = get_region_props(tissue_mask_extra)
    num_regions = 1
    regions_largest = get_largest_regions(region_props, num_regions)
    regions_largest_coords = region_to_coords(regions_largest)
    tissue_mask_open = coords_to_mask(regions_largest_coords, pillar_mask)
    tissue_mask = close_region(tissue_mask_open)
    return tissue_mask, wound_mask, wound_region


def mask_all_with_pillars(thresh_img_list: List, pillar_mask_list: List) -> List:
    """Given a thresholded image list. Will return masks and wound regions."""
    pillar_mask = mask_list_to_single_mask(pillar_mask_list)
    tissue_mask_list = []
    wound_mask_list = []
    wound_region_list = []
    num_images = len(thresh_img_list)
    wound_region = None
    for kk in range(0, num_images):
        background_mask = thresh_img_list[kk]
        tissue_mask, wound_mask, wound_region = leverage_pillars_for_wound_seg(
            pillar_mask, background_mask,wound_region
            )
        tissue_mask[pillar_mask==1] = 0
        tissue_mask_list.append(tissue_mask)
        wound_mask_list.append(wound_mask)
        wound_region_list.append(wound_region)
    return tissue_mask_list, wound_mask_list, wound_region_list