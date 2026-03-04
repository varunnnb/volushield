import numpy as np

from src.preprocessing.nifti_loader import load_nifti_volume
from src.preprocessing.normalizer import normalize_volume
from src.preprocessing.resize_volume import center_crop


INPUT_PATH = "data/raw/brats/BraTS-GLI-00467-000/BraTS-GLI-00467-000-t1c.nii.gz"
OUTPUT_PATH = "data/processed/brats_vol64.npy"


def main():

    print("Loading volume...")
    vol = load_nifti_volume(INPUT_PATH)

    print("Original shape:", vol.shape)

    print("Normalizing...")
    vol = normalize_volume(vol)

    print("Cropping to 64³...")
    vol = center_crop(vol, (64,64,64))

    print("New shape:", vol.shape)

    np.save(OUTPUT_PATH, vol)

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()