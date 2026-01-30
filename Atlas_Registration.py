import os
import numpy as np
import nibabel as nib
from nipype.interfaces import fsl


def atlas_registration(
    atlas_path: str,
    arr_seg: np.ndarray,
    image_path: str,
    image_t1: str,
    output_dir: str = "."
) -> str:
    """
    Register an atlas to T1-weighted MRI in MRA space using FSL FLIRT.

    Performs a multistep registration pipeline:
    1. Crops and extracts T1 brain using BET
    2. Resamples T1 to MRA space
    3. Registers atlas mask to T1 for initial alignment
    4. Registers full atlas using mask alignment as initialisation

    Parameters
    ----------
    atlas_path : str or Path
        Path to the atlas NIfTI file (.nii or .nii.gz) to be registered.
        Atlas should contain labelled regions or probability maps.
    arr_seg : np.ndarray
        Segmentation array (3D volume) used for preprocessing. Shape (H, W, D).
        Values are rounded to integers. Currently used for validation but not
        directly in the registration pipeline.
    image_path : str or Path
        Path to the reference MRA image (.nii or .nii.gz) defining target space.
        T1 will be resampled to this space before atlas registration.
    image_t1 : str or Path
        Path to the T1-weighted MRI image (.nii or .nii.gz) to register atlas to.
        This image will be cropped, skull-stripped, and resampled to MRA space.
    output_dir : str or Path, optional
        Directory where all intermediate and final outputs will be saved.
        Default is the current directory. Directory must exist or be creatable.

    Returns
    -------
    str
        Path to the registered atlas file in MRA space
        (filename: {T1_name}_registered_atlas.nii.gz).
    """
    # Round and cast to integer
    print(f"Registering atlas: {atlas_path} to T1: {image_t1} in MRA space: {image_path}")
    assert os.path.exists(atlas_path), f"Atlas file not found: {atlas_path}"
    assert os.path.exists(image_path), f"MRA image file not found: {image_path}"
    assert os.path.exists(image_t1), f"T1 image file not found: {image_t1}"
    
    arr_seg = np.round(arr_seg).astype(int)
    print(np.max(arr_seg))

    # Map values: 1 → 0, 2 → 1, everything else → 0
    # arr_seg = np.where(arr == 1, 0, np.where(arr_seg == 2, 1, 0)).astype(int)
    print(np.max(arr_seg))

    image_t1_name = os.path.basename(image_t1).split('.')[0]

    # Paths for outputs
    reg_atlas_path = os.path.join(output_dir, f"{image_t1_name}_registered_atlas.nii.gz")
    atlas_no_brain_mask = os.path.join(output_dir, f"{image_t1_name}_atlas_no_brain_mask.nii.gz")
    image_path_no_brain = os.path.join(output_dir, f"{image_t1_name}_brain.nii.gz")
    cropped_path = os.path.join(output_dir, f"{image_t1_name}_cropped.nii.gz")
    T1_in_MRA_path = os.path.join(output_dir, f"{image_t1_name}_in_MRA.nii.gz")

    # ============================================================================
    # 1. Extract and prepare T1 brain
    # ============================================================================

    # Crop T1
    fsl_roi = fsl.ExtractROI()
    fsl_roi.inputs.in_file = image_t1  # Use original image_T1
    fsl_roi.inputs.t_min = 0
    fsl_roi.inputs.t_size = -1
    fsl_roi.inputs.x_min = 0
    fsl_roi.inputs.x_size = -1
    fsl_roi.inputs.y_min = 50
    fsl_roi.inputs.y_size = -1
    fsl_roi.inputs.z_min = 0
    fsl_roi.inputs.z_size = -1
    result = fsl_roi.run()
    cropped_file = result.outputs.roi_file
    import shutil
    shutil.move(cropped_file, cropped_path)

    # Extract T1 brain with BET
    bet = fsl.BET()
    bet.inputs.in_file = cropped_path
    bet.inputs.mask = True
    bet.inputs.frac = 0.30
    bet.inputs.vertical_gradient = 0.0
    bet.inputs.out_file = image_path_no_brain
    bet.run()

    # Reorient T1 to the standard
    reorient = fsl.Reorient2Std()
    reorient.inputs.in_file = image_path_no_brain
    reorient.inputs.out_file = image_path_no_brain
    reorient.run()

    # ============================================================================
    # 2. Resample T1 to MRA space
    # ============================================================================

    print("Resampling T1 to MRA space...")
    flirt_T1_to_MRA = fsl.FLIRT()
    flirt_T1_to_MRA.inputs.in_file = image_path_no_brain
    flirt_T1_to_MRA.inputs.reference = image_path
    flirt_T1_to_MRA.inputs.out_file = T1_in_MRA_path
    flirt_T1_to_MRA.inputs.out_matrix_file = os.path.join(output_dir, f"{image_t1_name}_brain_flirtt.mat")
    flirt_T1_to_MRA.inputs.apply_xfm = True
    flirt_T1_to_MRA.inputs.interp = 'trilinear'
    flirt_T1_to_MRA.inputs.uses_qform = True
    flirt_T1_to_MRA.run()

    # ============================================================================
    # 3. Prepare atlas binary mask
    # ============================================================================

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    atlas_binary = (atlas_data > 0).astype(np.uint8)
    atlas_binary_img = nib.Nifti1Image(atlas_binary, affine=atlas_img.affine, header=atlas_img.header)
    nib.save(atlas_binary_img, atlas_no_brain_mask)

    # ============================================================================
    # 4. Step 1: Register atlas MASK to T1 (in MRA space)
    # ============================================================================

    print("Step 1: Registering atlas mask to T1...")
    flirt_atlas_mask_to_T1 = fsl.FLIRT()
    flirt_atlas_mask_to_T1.inputs.in_file = atlas_no_brain_mask
    flirt_atlas_mask_to_T1.inputs.reference = T1_in_MRA_path
    flirt_atlas_mask_to_T1.inputs.out_file = os.path.join(output_dir, f"{image_t1_name}_atlas_mask_to_T1.nii.gz")
    flirt_atlas_mask_to_T1.inputs.out_matrix_file = os.path.join(output_dir, f"{image_t1_name}_atlas_mask_to_T1.mat")
    flirt_atlas_mask_to_T1.inputs.dof = 12
    flirt_atlas_mask_to_T1.inputs.interp = 'nearestneighbour'
    flirt_atlas_mask_to_T1.inputs.uses_qform = True
    flirt_atlas_mask_to_T1.run()

    # ============================================================================
    # 5. Step 2: Register full atlas to T1 (using mask alignment as initial)
    # ============================================================================

    print("Step 2: Registering full atlas to T1...")
    flirt_atlas_to_T1 = fsl.FLIRT()
    flirt_atlas_to_T1.inputs.in_file = atlas_path
    flirt_atlas_to_T1.inputs.reference = T1_in_MRA_path
    flirt_atlas_to_T1.inputs.out_file = reg_atlas_path
    flirt_atlas_to_T1.inputs.out_matrix_file = os.path.join(output_dir, f"{image_t1_name}_atlas_to_T1.mat")
    flirt_atlas_to_T1.inputs.in_matrix_file = os.path.join(output_dir, f"{image_t1_name}_atlas_mask_to_T1.mat")
    flirt_atlas_to_T1.inputs.searchr_x = [-30, 30]  # Constrain rotation search
    flirt_atlas_to_T1.inputs.searchr_y = [-30, 30]
    flirt_atlas_to_T1.inputs.searchr_z = [-30, 30]
    flirt_atlas_to_T1.inputs.dof = 12
    flirt_atlas_to_T1.inputs.interp = 'nearestneighbour'
    flirt_atlas_to_T1.inputs.uses_qform = True
    flirt_atlas_to_T1.run()

    print(f"Registered atlas saved to: {reg_atlas_path}")

def process_registration(
    image_path: str,
    image_t1_path: str,
    mask_path: str,
    atlas_path: str,
    output_dir: str = ".",
):
    """
    Process vessel segmentation with atlas registration and metric computation.

    Registers an atlas to the image space, extracts vessel regions based on
    atlas labels, and computes morphological metrics for each region.

    Parameters
    ----------
    image_path : str or Path
        Path to the reference MRA image (.nii or .nii.gz) used for registration
        and metric computation. Defines the target space.
    image_t1_path : str or Path
        Path to the T1-weighted MRI image (.nii or .nii.gz) used as the intermediate
        registration target. T1 is registered to MRA, then atlas to T1.
    mask_path : str or Path
        Path to the binary vessel segmentation mask (.nii or .nii.gz).
        Expected to contain binary values (0=background, 1=vessel).
    atlas_path : str or Path
        Path to the atlas file (.nii or .nii.gz) with labelled regions.
        Each unique integer label represents a different anatomical region.
    selected_metrics : list of str
        List of metric names to compute for each region. Examples include:
        'volume', 'length', 'tortuosity', 'mean_radius', 'surface_area'.
    save_segment_masks : bool
        If True, saves individual segmentation masks for each atlas region.
        Masks are saved to the same directory as mask_path.
    save_conn_comp_masks : bool
        If True, saves connected component masks for each region.
        Useful for analysing disconnected vessel segments separately.

    Returns
    -------
    dict
        Results dictionary with structure:
        {
            'region_metrics': {
                label1: {metric1: value1, metric2: value2, ...},
                label2: {metric1: value1, metric2: value2, ...},
                ...
            }
        }
        where each label is an atlas region ID, and metrics are computed values.
        Regions with no vessel segmentation are excluded from the results.
"""
    # Load binary mask
    seg = nib.load(mask_path)
    arr_seg = seg.get_fdata()
    atlas_registration(atlas_path, arr_seg, image_path, image_t1_path, output_dir)