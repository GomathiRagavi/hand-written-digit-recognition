from PIL import Image, ImageOps
import numpy as np

def _binarize(img_np, thresh=None):
    """Simple binarization. If thresh None, pick from percentiles."""
    if thresh is None:
        # robust threshold near background/foreground split
        thresh = max(20, int(np.percentile(img_np, 80)))
    return (img_np > thresh).astype(np.uint8)

def _center_of_mass_shift(img_np):
    """Compute integer pixel shift to move mass to center (14,14)."""
    h, w = img_np.shape
    y = np.arange(h).reshape(-1, 1)
    x = np.arange(w).reshape(1, -1)
    mass = img_np.astype(np.float32)
    s = mass.sum()
    if s == 0:
        return 0, 0
    cy = (y * mass).sum() / s
    cx = (x * mass).sum() / s
    # target center is 13.5,13.5 approx for 28x28
    dy = int(round(13.5 - cy))
    dx = int(round(13.5 - cx))
    return dy, dx

def _shift_with_zeros(img_np, dy, dx):
    """Translate using roll + zero fill (no wrap artifacts)."""
    shifted = np.roll(img_np, dy, axis=0)
    if   dy > 0: shifted[:dy, :] = 0
    elif dy < 0: shifted[dy:, :] = 0

    shifted = np.roll(shifted, dx, axis=1)
    if   dx > 0: shifted[:, :dx] = 0
    elif dx < 0: shifted[:, dx:] = 0
    return shifted

def preprocess_pil_image_to_mnist(img: Image.Image):
    """
    Take a PIL image (any size), return a (1,28,28,1) float32 array in [0,1]
    matching MNIST style (white digit on black, centered).
    """
    # 1) grayscale
    img = img.convert("L")
    np_img = np.array(img).astype(np.uint8)

    # 2) invert if background seems light (MNIST expects dark bg)
    if np_img.mean() > 127:
        np_img = 255 - np_img

    # 3) binarize & crop to digit bounding box
    mask = _binarize(np_img)  # 0/1
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        # blank image
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    np_crop = np_img[y_min:y_max+1, x_min:x_max+1]

    # 4) scale longest side to 20, keep aspect ratio
    crop_pil = Image.fromarray(np_crop)
    h, w = np_crop.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    crop_pil = crop_pil.resize((new_w, new_h), Image.LANCZOS)

    # 5) paste into 28x28 with margins
    canvas = Image.new("L", (28, 28), color=0)
    top  = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas.paste(crop_pil, (left, top))

    np28 = np.array(canvas).astype(np.float32)

    # 6) center-of-mass shift (small integer translation)
    dy, dx = _center_of_mass_shift(np28)
    np28 = _shift_with_zeros(np28, dy, dx)

    # 7) normalize to [0,1] and add batch/channel dims
    np28 = np.clip(np28, 0, 255) / 255.0
    np28 = np28.reshape(1, 28, 28, 1).astype("float32")
    return np28
