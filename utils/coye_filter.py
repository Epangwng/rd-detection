import numpy as np
import cv2
import warnings
from skimage.color import rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects
from sklearn.decomposition import PCA

def isodata(I):
    I_uint8 = np.clip(np.round(I * 255.0), 0, 255).astype(np.uint8)

    counts, _ = np.histogram(I_uint8.flatten(), bins=256, range=(0, 256))
    N = np.arange(256, dtype=np.float64)

    mu = np.cumsum(counts)
    if mu[-1] == 0:
        return 0.0

    T_val = int(np.round(np.sum(N * counts) / mu[-1]))
    T = max(1, min(256, T_val + 1))

    for _ in range(10000):
        mu2 = np.cumsum(counts[0:T])
        MBT = np.sum(N[0:T] * counts[0:T]) / mu2[-1] if mu2[-1] != 0 else 0.0

        mu3 = np.cumsum(counts[T - 1 :])
        MAT = np.sum(N[T - 1 :] * counts[T - 1 :]) / mu3[-1] if mu3[-1] != 0 else 0.0

        T_next_val = int(np.round((MAT + MBT) / 2.0))
        T_next = max(1, min(256, T_next_val + 1))

        if abs(T_next - T) < 1:
            T = T_next
            break
        T = T_next

    Threshold = T
    level = (Threshold - 1) / (N[-1] - 1)
    return level

def imoverlay(in_img, mask, color=(0, 0, 0)):
    if in_img.dtype != np.uint8:
        in_uint8 = np.clip(np.round(in_img * 255.0), 0, 255).astype(np.uint8)
    else:
        in_uint8 = in_img.copy()

    color_arr = np.array(color)
    if color_arr.dtype != np.uint8:
        color_uint8 = np.clip(np.round(color_arr * 255.0), 0, 255).astype(np.uint8)
    else:
        color_uint8 = color_arr

    mask_bool = mask != 0
    out = in_uint8.copy()

    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)

    out[mask_bool] = color_uint8
    return out

def apply_coye_filter(image_rgb):
    """
    Applies the Tyler Coye Retinal Vessel Segmentation filter to an RGB image array.
    """
    # Resize image to match the pipeline expectation
    B = cv2.resize(image_rgb, (565, 584))
    im = B.astype(np.float64) / 255.0

    # Create FOV mask
    gray_for_mask = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_for_mask, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (565, 584)) > 0

    # RGB to Gray via PCA
    lab = rgb2lab(im)
    f = 0.0
    weights = np.array([1.0 - f, f / 2.0, f / 2.0])
    wlab = (lab * weights).reshape(-1, 3)

    pca = PCA(n_components=1)
    if np.any(mask):
        pca.fit(wlab[mask.flatten()])
    else:
        pca.fit(wlab)
        
    S_1 = pca.transform(wlab).reshape(lab.shape[:2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.corrcoef(S_1.flatten(), lab[:, :, 0].flatten())[0, 1]
    if not np.isnan(corr) and corr < 0:
        S_1 = -S_1

    s_range = S_1.max() - S_1.min()
    gray = (S_1 - S_1.min()) / s_range if s_range > 0 else np.zeros_like(S_1)

    # Sharpen image
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    gray = np.clip(gray, 0, 1)

    # CLAHE
    kernel_size = (gray.shape[0] // 8, gray.shape[1] // 8)
    J = equalize_adapthist(gray, kernel_size=kernel_size, nbins=128, clip_limit=0.01)

    # Background Exclusion
    h_kernel = np.ones((31, 31), dtype=np.float64) / (31.0 * 31.0)
    JF = cv2.filter2D(J, -1, h_kernel, borderType=cv2.BORDER_CONSTANT)
    Z = JF - J

    # IsoData Threshold
    level = isodata(Z)

    BW = Z > (level - 0.015)
    BW2 = remove_small_objects(BW, min_size=15)

    BW2_comp = ~BW2
    out = imoverlay(B, BW2_comp, color=(0, 0, 0))

    return out
