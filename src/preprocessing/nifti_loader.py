import SimpleITK as sitk
import numpy as np

def load_nifti_volume(filepath: str) -> np.ndarray:
    """
    Load a NIfTI volume and return numpy array (D, H, W)
    """
    image = sitk.ReadImage(filepath)
    volume = sitk.GetArrayFromImage(image)

    return volume.astype(np.float32)