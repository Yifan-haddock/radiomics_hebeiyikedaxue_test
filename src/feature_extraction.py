import os
import numpy as np
import pydicom
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
import glob
from tqdm import tqdm
import joblib
import re

def load_rtstruct(rtstruct_path):
    """Load the RTSTRUCT DICOM file."""
    rtstruct = pydicom.dcmread(rtstruct_path)
    return rtstruct

def extract_contours(rtstruct, roi_pattern):
    """Extract 2D contours for a specific ROI (e.g., GTV)."""
    contours = []
    for roi, contour in zip(rtstruct.StructureSetROISequence, rtstruct.ROIContourSequence):
        if roi_pattern.match(roi.ROIName):
            print(f"Found ROI: {roi.ROIName}")
            for contour_sequence in contour.ContourSequence:
                contour_data = contour_sequence.ContourData
                points = np.array(contour_data).reshape(-1, 3)  # Convert to (x, y, z)
                contours.append(points)
    return contours
    # raise ValueError(f"ROI gtv region not found in RTSTRUCT file.")

def create_mask_from_contours(contours, ct_image):
    """Convert 2D contours into a 3D binary mask aligned with the CT scan."""
    # Get metadata from the CT image
    origin = np.array(ct_image.GetOrigin())  # Physical coordinates of the image origin
    spacing = np.array(ct_image.GetSpacing())  # Voxel size in each dimension (x, y, z)
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)  # Image orientation matrix

    # Initialize the mask
    mask_shape = sitk.GetArrayFromImage(ct_image).shape
    mask = np.zeros(mask_shape, dtype=np.uint8)

    for points in contours:
        for point in points:
            # Convert physical coordinates to voxel indices
            physical_point = np.array(point)
            voxel_indices = np.round(np.linalg.inv(direction).dot((physical_point - origin) / spacing)).astype(int)
            
            # Convert to (z, y, x) for mask indexing
            x, y, z = voxel_indices
            # if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            mask[z, y, x] = 1  # Assign 1 to points within the contour

    # Convert the mask to a SimpleITK image and inherit metadata from the CT image
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetOrigin(ct_image.GetOrigin())
    mask_image.SetSpacing(ct_image.GetSpacing())
    mask_image.SetDirection(ct_image.GetDirection())

    return mask_image

def extract_radiomics_features(ct_path, mask):
    """Extract radiomics features using pyradiomics."""
    # Convert CT and mask to NIfTI for pyradiomics compatibility
    ct_image = sitk.ReadImage(ct_path)
    mask_path = os.path.join(Path(ct_path).parent,"binary_mask.nii.gz")
    sitk.WriteImage(mask, mask_path)

    # Initialize pyradiomics extractor with specific feature categories and image filters
    features_name = {
        'firstorder': {},
        'shape': {},
        'glcm': {},
        'glrlm': {},
        'glszm': {},
        'gldm': {},
    }
    
    imageTypeSettings = {
        'Original': {},              
        'Wavelet': {},               
        'LoG': {'sigma': [1, 3, 5]},
        'Gradient': {},
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(additionalInfo=True)
    extractor.enableImageTypes(**imageTypeSettings)
    extractor.enableFeaturesByName(**features_name)

    # Set GLCM symmetry configuration
    # extractor.settings['symmetricalGLCM'] = True

    # Extract features
    features = extractor.execute(ct_path, mask_path)

    feature_dict = {key: value for key, value in features.items() if not key.startswith("diagnostics")}
    if len(feature_dict) < 300:
        raise ValueError(f"Insufficient features extracted: {len(feature_dict)}. Maybe increase more filters.")

    return feature_dict

if __name__ == "__main__":
    ## init params
    feature_names = []
    feature_matrix = []
    patient_dirs = glob.glob('../data/manifest-1603198545583/NSCLC-Radiomics/*/')
    feature_matrix_path = '../data/manifest-1603198545583/NSCLC-Radiomics/feature_matrix.joblib'
    feature_name_path = '../data/manifest-1603198545583/NSCLC-Radiomics/feature_names.txt'
    gtv_pat = re.compile(r'\b\w*gtv\w*\b', re.IGNORECASE)
    
    ## roundandround
    for i, patient_dir in tqdm(enumerate(patient_dirs[347+32:]), total = len(patient_dirs)-347-32):
        file_list = glob.glob(f"{patient_dir}/*/*")
        file_list_sorted = sorted(
                [file for file in file_list if (file.split('/')[-1][0].isdigit()) and not (re.search('Segmentation',file))],
                key=lambda x: float(x.split('/')[-1].split('.')[0])
            )
        
        # input paths
        ct_series_path = file_list_sorted[0]
        rtstruct_path = glob.glob(f"{file_list_sorted[1]}/*")[0]
        ct_path = os.path.join(Path(ct_series_path).parent, "CT_image.nii.gz")
        feature_path = os.path.join(Path(ct_series_path).parent, "CT_features.joblib")
        # preprocess ct image
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(ct_series_path)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        sitk.WriteImage(image, ct_path)
        # Load and extract RTSTRUCT
        rtstruct = load_rtstruct(rtstruct_path)
        contours = extract_contours(rtstruct, gtv_pat)
        ct_image = sitk.ReadImage(ct_path)
        mask = create_mask_from_contours(contours, ct_image)
        features = extract_radiomics_features(ct_path, mask)
        # save patient extracted features
        with open(feature_path, 'wb') as f:
            joblib.dump(features, f)
        
        if i == 0:
            feature_names = list(features.keys())
        feature_matrix.append(np.array([features[name] for name in feature_names], dtype=np.float64))
        
    ## save all features
    feature_matrix = np.stack(feature_matrix)
    with open(feature_matrix_path, 'wb') as f:
        joblib.dump(feature_matrix, f)
    with open(feature_name_path,'w') as f:
        f.writelines([i+'\n' for i in feature_names])
    