import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def get_wound_area(output_path: Path) -> np.ndarray:
    file_name = "wound_area_vs_frame"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    wound_area = np.loadtxt(str(file_path))
    return wound_area


def get_is_broken(output_path: Path) -> np.ndarray:
    file_name = "is_broken_vs_frame"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    is_broken = np.loadtxt(str(file_path))
    return is_broken


def get_is_closed(output_path: Path) -> np.ndarray:
    file_name = "is_closed_vs_frame"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    is_closed = np.loadtxt(str(file_path))
    return is_closed


def smooth_with_GPR(s: np.ndarray) -> np.ndarray:
    num_frames = s.shape[0]
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e1)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    xdata = np.arange(num_frames).reshape(-1, 1)
    xhat = xdata
    ydata = s.reshape(-1, 1)
    remove_indices = np.where(np.isnan(ydata.reshape(-1)))[0]
    xdata = np.delete(xdata, remove_indices, axis=0)
    ydata = np.delete(ydata, remove_indices, axis=0)
    model.fit(xdata, ydata)
    yhat = model.predict(xhat)
    return yhat.reshape(-1)


def is_broken_smooth(is_broken: np.ndarray) -> np.ndarray:
    # step 1: remove "blips"
    is_broken_no_blips = []
    is_broken_no_blips.append(0)
    for kk in range(1, is_broken.shape[0]-1):
        if is_broken[kk] == 1:
            if is_broken[kk-1] or is_broken[kk + 1]:
                is_broken_no_blips.append(1)
            else:
                is_broken_no_blips.append(0)
        else:
            is_broken_no_blips.append(0)
    is_broken_no_blips.append(is_broken[-1])
    is_broken_no_blips = np.asarray(is_broken_no_blips)
    # step 2: make irreversible
    is_broken_irreversible = []
    is_broken_irreversible.append(0)
    for kk in range(1, is_broken.shape[0]):
        if np.sum(is_broken_no_blips[0:kk]) > 0 or is_broken_no_blips[kk] == 1:
            is_broken_irreversible.append(1)
        else:
            is_broken_irreversible.append(0)
    return np.asarray(is_broken_irreversible)


def is_closed_smooth(is_closed: np.ndarray) -> np.ndarray:
    # step 1: remove "blips"
    is_closed_no_blips = []
    is_closed_no_blips.append(is_closed[0])
    is_closed_no_blips.append(is_closed[1])
    for kk in range(2, is_closed.shape[0]-2):
        if is_closed[kk] == 1:
            if is_closed[kk - 2] and is_closed[kk - 1] or is_closed[kk + 1] and is_closed[kk + 2]:
                is_closed_no_blips.append(1)
            elif is_closed[kk - 1] and is_closed[kk + 1]:
                is_closed_no_blips.append(1)
            else:
                is_closed_no_blips.append(0)
        else:
            is_closed_no_blips.append(0)
    is_closed_no_blips.append(is_closed[-2])
    is_closed_no_blips.append(is_closed[-1])
    is_closed_no_blips = np.asarray(is_closed_no_blips)
    return is_closed_no_blips


def wound_area_with_is_closed(wound_area: np.ndarray, is_closed: np.ndarray) -> np.ndarray:
    wound_area_new = []
    for kk in range(0, wound_area.shape[0]):
        if is_closed[kk] == 0:
            wound_area_new.append(wound_area[kk])
        else:
            wound_area_new.append(0)
    return np.asarray(wound_area_new)


def run_full_postproc_sequence(wound_area: np.ndarray, is_broken: np.ndarray, is_closed: np.ndarray) -> np.ndarray:
    is_broken_sm = is_broken_smooth(is_broken)
    is_closed_sm = is_closed_smooth(is_closed)
    wound_area_new = wound_area_with_is_closed(wound_area, is_closed_sm)
    wound_area_new_smooth = smooth_with_GPR(wound_area_new)
    wound_area_final = wound_area_with_is_closed(wound_area_new_smooth, is_closed_sm)
    if np.sum(is_broken_sm) > 0:
        frame_broken = np.min(np.argwhere(is_broken_sm == 1))
    else:
        frame_broken = None
    if np.sum(is_closed_sm) > 0:
        frame_closed = np.min(np.argwhere(is_closed_sm == 1))
    else:
        frame_closed = None
    # return:
    # wound_area_smooth, is_broken_smooth, is_closed_smooth
    # frame_broken (first frame when broken) -- None if it doesn't break
    # frame closed (first frame when closed) -- None if it doesn't close
    return wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed


def get_postproc_results(output_path: Path):
    wound_area = get_wound_area(output_path)
    is_broken = get_is_broken(output_path)
    is_closed = get_is_closed(output_path)
    wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed = run_full_postproc_sequence(wound_area, is_broken, is_closed)
    return wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed


def get_pillar_info(output_path: Path) -> np.ndarray:
    file_name = "pillar_tracker_x"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    pillar_disp_x = np.loadtxt(str(file_path))
    file_name = "pillar_tracker_y"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    pillar_disp_y = np.loadtxt(str(file_path))
    return pillar_disp_x, pillar_disp_y


def pos_to_disp(pos_arr: np.array):
    disp_arr = np.zeros(pos_arr.shape)
    for kk in range(0, pos_arr.shape[0]):
        disp_arr[kk, :] = pos_arr[kk, :] - pos_arr[0, :]
    return disp_arr


def get_drift_mat():
    mat = np.zeros((10, 10))
    for kk in range(0, 8):
        mat[kk, kk] = 1
    for kk in range(0, 4):
        mat[kk, 8] = 1
    for kk in range(4, 8):
        mat[kk, 9] = 1
    mat[8, :] = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    mat[9, :] = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    mat_inv = np.linalg.inv(mat)
    return mat, mat_inv


def get_defl_decoupled(defl_x: np.ndarray, defl_y: np.ndarray, mat_inv: np.ndarray) -> np.ndarray:
    vec = np.zeros((10, 1))
    vec[0:4, 0] = defl_x[:]
    vec[4:8, 0] = defl_y[:]
    new_disp = np.dot(mat_inv, vec)
    return new_disp


def drift_correct_pillar_track(output_path: Path):
    pillar_disp_x, pillar_disp_y = get_pillar_info(output_path)
    pillar_defl_x = pos_to_disp(pillar_disp_x)
    pillar_defl_y = pos_to_disp(pillar_disp_y)
    num_frames = pillar_disp_x.shape[0]
    num_pillars = pillar_disp_x.shape[1]
    rigid_x = np.zeros((pillar_disp_x.shape[0], 1))
    rigid_y = np.zeros((pillar_disp_y.shape[0], 1))
    if num_pillars == 4:
        pillar_defl_x_update = np.zeros(pillar_defl_x.shape)
        pillar_defl_y_update = np.zeros(pillar_defl_y.shape)
        _, mat_inv = get_drift_mat()
        for jj in range(0, num_frames):
            new_disp = get_defl_decoupled(pillar_defl_x[jj, :], pillar_defl_y[jj, :], mat_inv)
            pillar_defl_x_update[jj, :] = new_disp[0:4, 0]
            pillar_defl_y_update[jj, :] = new_disp[4:8, 0]
            rigid_x[jj, :] = new_disp[8, 0]
            rigid_y[jj, :] = new_disp[9, 0]
        return pillar_defl_x_update, pillar_defl_y_update, rigid_x, rigid_y
    else:
        return pillar_defl_x, pillar_defl_y, rigid_x, rigid_y
