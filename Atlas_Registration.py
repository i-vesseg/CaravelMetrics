import logging
import os
import argparse
import subprocess
import sys

# Setup basic logging for standalone execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
#                             DEFAULT PATHS
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ATLAS_DIR = os.path.join(SCRIPT_DIR, "Atlas")

DEFAULT_MNI_T1_BRAIN = os.path.join(DEFAULT_ATLAS_DIR, "MNI152_T1_1mm_Brain.nii.gz")
DEFAULT_MNI_T1_WHOLE = os.path.join(DEFAULT_ATLAS_DIR, "MNI152_T1_1mm.nii.gz")
DEFAULT_PARCELLATION = os.path.join(DEFAULT_ATLAS_DIR, "ArterialAtlas.nii.gz") 

def run_cmd(cmd_list):
    """Helper function to run shell commands via subprocess."""
    cmd_str = ' '.join(cmd_list)
    logger.debug(f"Running: {cmd_str}")
    try:
        result = subprocess.run(
            cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd_str}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Command failed: {cmd_str}\n{e.stderr}")

def apply_ants_preprocessing(image_path: str, output_dir: str, do_denoise: bool, do_debias: bool, cleanup_list: list) -> str:
    """Unified ANTs preprocessing. Returns the path to the processed image."""
    base_name = os.path.basename(image_path).split('.')[0]
    current_img = image_path

    if do_denoise:
        logger.info(f"  [PREPROC] Denoising {base_name} using ANTs DenoiseImage...")
        denoised_img = os.path.join(output_dir, f"{base_name}_denoised.nii.gz")
        run_cmd(["DenoiseImage", "-d", "3", "-i", current_img, "-o", denoised_img, "-n", "Rician"])
        current_img = denoised_img

    if do_debias:
        logger.info(f"  [PREPROC] Debiasing {base_name} using ANTs N4BiasFieldCorrection...")
        debiased_img = os.path.join(output_dir, f"{base_name}_debiased.nii.gz")
        run_cmd(["N4BiasFieldCorrection", "-d", "3", "-i", current_img, "-o", debiased_img])
        
        if do_denoise:
            cleanup_list.append(current_img)
            
        current_img = debiased_img

    return current_img

def generate_qc_snapshot(mra_path: str, seg_path: str, output_path: str):
    """Generates a highly condensed 3-view QC snapshot without bloating dependencies."""
    try:
        import numpy as np
        import nibabel as nib
        import matplotlib.pyplot as plt
        from scipy.ndimage import maximum_filter, minimum_filter
    except ImportError:
        logger.error("  [QC] Snapshot failed: Missing libraries. Run: pip install nibabel matplotlib scipy")
        return

    logger.info(f"  [QC] Generating quality control snapshot: {os.path.basename(output_path)}...")
    
    # Load data and spacing
    mra_img = nib.load(mra_path)
    mra_data = mra_img.get_fdata()
    seg_data = nib.load(seg_path).get_fdata()
    dx, dy, dz = mra_img.header.get_zooms()[:3]

    # Fast Boundaries
    footprint = np.ones((3, 3, 3))
    boundaries = (maximum_filter(seg_data, footprint=footprint) != minimum_filter(seg_data, footprint=footprint)) & (seg_data > 0)
    contour_labels = np.where(boundaries, seg_data, 0)

    # Crop to brain bounding box for tight views
    coords = np.argwhere(mra_data > np.percentile(mra_data, 10))
    if coords.size > 0:
        min_c = np.maximum(coords.min(axis=0) - 5, 0)
        max_c = np.minimum(coords.max(axis=0) + 5, mra_data.shape)
        mra_crop = mra_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
        seg_crop = contour_labels[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    else:
        mra_crop, seg_crop = mra_data, contour_labels

    # Contrast scaling
    mra_crop = np.clip(mra_crop, *np.percentile(mra_crop, [1, 99.5]))

    # Grid setup
    mid = np.array(mra_crop.shape) // 2
    step, num_slices = 40, 5
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, num_slices, figsize=(4 * num_slices, 11), dpi=200)
    fig.suptitle(f"QC Panel: {os.path.basename(mra_path)}", fontsize=14, fontweight='bold', y=0.98)

    views = [
        ('Sagittal', 0, dz / dy),
        ('Coronal', 1, dz / dx),
        ('Axial', 2, dy / dx)
    ]

    cmap = plt.get_cmap('turbo')
    
    for row_idx, (v_name, axis, aspect) in enumerate(views):
        slices = np.linspace(mid[axis] - step, mid[axis] + step, num_slices, dtype=int)
        for col_idx, s in enumerate(slices):
            ax = axes[row_idx, col_idx]
            s_valid = np.clip(s, 0, mra_crop.shape[axis] - 1)
            
            # Use np.take to dynamically extract slices across arbitrary axes
            mra_s = np.rot90(mra_crop.take(s_valid, axis=axis))
            seg_s = np.rot90(seg_crop.take(s_valid, axis=axis))
            
            ax.imshow(mra_s, cmap='gray', origin='upper', aspect=aspect)
            ax.imshow(np.ma.masked_where(seg_s == 0, seg_s), cmap=cmap, alpha=1, interpolation='nearest', origin='upper', aspect=aspect)
            
            if col_idx == 0:
                ax.set_ylabel(v_name, fontsize=11, fontweight='bold', labelpad=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    plt.style.use('default')

def atlas_registration(
    mni_parcellation: str,
    mni_t1_brain: str,
    mni_t1_whole: str,
    image_path: str,
    output_dir: str = ".",
    image_t1: str = None,
    preprocess_mra: bool = False,
    preprocess_t1: bool = False,
    mra_stripped: bool = False,
    t1_stripped: bool = False,
    generate_qc: bool = False
) -> tuple:

    mra_prefix = os.path.basename(image_path).split('.')[0]
    files_to_cleanup = []

    # --- MRA PREPROCESSING (ANTs denoise + bias correction) ---
    if preprocess_mra:
        image_path = apply_ants_preprocessing(
            image_path, output_dir, do_denoise=True, do_debias=True, cleanup_list=files_to_cleanup
        )

    # --- T1 PREPROCESSING (ANTs denoise + bias correction) ---
    if image_t1 and os.path.exists(image_t1) and preprocess_t1:
        image_t1 = apply_ants_preprocessing(
            image_t1, output_dir, do_denoise=True, do_debias=True, cleanup_list=files_to_cleanup
        )

    # ---------------------------------------------------------
    # ROUTE A: T1 IS AVAILABLE
    # ---------------------------------------------------------
    if image_t1 and os.path.exists(image_t1):
        t1_prefix = os.path.basename(image_t1).split('.')[0]
        
        if not t1_stripped:
            logger.info("  [DEBUG] ROBUSTFOV: Removing neck tissue...")
            t1_fov = os.path.join(output_dir, f"{t1_prefix}_cropped_fov.nii.gz")
            run_cmd(["robustfov", "-i", image_t1, "-r", t1_fov, "-b", "180"])
            files_to_cleanup.append(t1_fov)

            logger.info("  [DEBUG] BET: Extracting brain...")
            t1_brain = os.path.join(output_dir, f"{t1_prefix}_brain.nii.gz")
            run_cmd(["bet", t1_fov, t1_brain, "-m", "-f", "0.4", "-R"])
            files_to_cleanup.append(t1_brain)
            
            t1_to_reorient = t1_brain
        else:
            logger.info("  [DEBUG] T1 is already skull-stripped. Skipping FOV crop and BET.")
            t1_to_reorient = image_t1

        logger.info("  [DEBUG] REORIENT: Aligning T1 orientation to standard...")
        t1_reoriented = os.path.join(output_dir, f"{t1_prefix}_reoriented.nii.gz")
        run_cmd(["fslreorient2std", t1_to_reorient, t1_reoriented])
        files_to_cleanup.append(t1_reoriented)

        logger.info("  [DEBUG] FLIRT: Registering Subject T1 to Subject MRA...")
        t1_in_mra_img = os.path.join(output_dir, f"{t1_prefix}_in_MRA.nii.gz")
        t1_to_mra_mat = os.path.join(output_dir, f"{t1_prefix}_to_MRA.mat")
        run_cmd([
            "flirt", "-in", t1_reoriented, "-ref", image_path, 
            "-out", t1_in_mra_img, "-omat", t1_to_mra_mat, 
            "-dof", "6", "-cost", "mutualinfo",
            "-searchrx", "-180", "180", "-searchry", "-180", "180", "-searchrz", "-180", "180"
        ])

        logger.info("  [DEBUG] FLIRT: Registering MNI Brain Template to Subject T1...")
        mni_in_t1_img = os.path.join(output_dir, f"MNI_in_{t1_prefix}.nii.gz")
        mni_to_t1_mat = os.path.join(output_dir, f"MNI_to_{t1_prefix}.mat")
        run_cmd([
            "flirt", "-in", mni_t1_brain, "-ref", t1_reoriented, 
            "-out", mni_in_t1_img, "-omat", mni_to_t1_mat, 
            "-dof", "12", "-cost", "corratio"
        ])
        files_to_cleanup.append(mni_in_t1_img)

        logger.info("  [DEBUG] CONVERT_XFM: Combining transforms (MNI -> T1 -> MRA)...")
        combined_mat = os.path.join(output_dir, f"{mra_prefix}_MNI_to_MRA_combined.mat")
        run_cmd([
            "convert_xfm", "-omat", combined_mat, 
            "-concat", t1_to_mra_mat, mni_to_t1_mat
        ])

    # ---------------------------------------------------------
    # ROUTE B: NO T1 (DIRECT MNI -> MRA)
    # ---------------------------------------------------------
    else:
        logger.info("  [DEBUG] No T1 provided. Routing direct MNI to MRA registration...")
        
        mni_template = mni_t1_brain if mra_stripped else mni_t1_whole
        template_type = "Brain" if mra_stripped else "Whole Head"
        
        logger.info(f"  [DEBUG] FLIRT: Registering MNI {template_type} Template to Subject MRA...")
        mni_in_mra_img = os.path.join(output_dir, f"MNI_in_{mra_prefix}.nii.gz")
        combined_mat = os.path.join(output_dir, f"{mra_prefix}_MNI_to_MRA_combined.mat")
        run_cmd([
            "flirt", "-in", mni_template, "-ref", image_path, 
            "-out", mni_in_mra_img, "-omat", combined_mat, 
            "-dof", "12", "-cost", "normmi", "-bins", "32",
            "-searchrx", "-180", "180", "-searchry", "-180", "180", "-searchrz", "-180", "180"
        ])

    # ---------------------------------------------------------
    # COMMON FINAL STEPS
    # ---------------------------------------------------------
    logger.info("  [DEBUG] CONVERT_XFM: Inverting combined matrix (MRA -> MNI)...")
    mra_to_mni_mat = os.path.join(output_dir, f"{mra_prefix}_MRA_to_MNI_inverse.mat")
    run_cmd(["convert_xfm", "-omat", mra_to_mni_mat, "-inverse", combined_mat])

    logger.info("  [DEBUG] APPLYXFM: Warping MNI Parcellation to MRA space...")
    reg_parcellation_path = os.path.join(output_dir, f"{mra_prefix}_registered_parcellation.nii.gz")
    run_cmd([
        "flirt", "-in", mni_parcellation, "-ref", image_path, 
        "-out", reg_parcellation_path, "-applyxfm", "-init", combined_mat, 
        "-interp", "nearestneighbour"
    ])

    logger.info("  [DEBUG] IMAGEMATHS: Dilating parcellation labels into the background...")
    dilated_parcellation_path = os.path.join(output_dir, f"{mra_prefix}_registered_parcellation_dilated.nii.gz")
    run_cmd([
        "fslmaths", reg_parcellation_path, 
        "-dilD", "-dilD", "-dilD", "-dilD", "-dilD", 
        dilated_parcellation_path
    ])

    # 8. CLEANUP
    for f in files_to_cleanup:
        if os.path.exists(f):
            os.remove(f)

    # 9. QUALITY CONTROL SNAPSHOT
    if generate_qc:
        qc_output = os.path.join(output_dir, f"{mra_prefix}_QC_snapshot.png")
        generate_qc_snapshot(image_path, reg_parcellation_path, qc_output)

    return dilated_parcellation_path, mra_to_mni_mat

def process_registration(
    image_path,
    output_dir,
    mni_parcellation=None,
    mni_t1_template=None,
    image_t1_path=None,
    preprocess_mra=False,
    preprocess_t1=False,
    mra_stripped=False,
    t1_stripped=False,
    generate_qc=False
):
    
    if mni_parcellation is None:
        mni_parcellation = DEFAULT_PARCELLATION

    if mni_t1_template is not None:
        mni_t1_brain = mni_t1_template
        mni_t1_whole = mni_t1_template
    else:
        mni_t1_brain = DEFAULT_MNI_T1_BRAIN
        mni_t1_whole = DEFAULT_MNI_T1_WHOLE

    required_files = [(mni_parcellation, "MNI Parcellation")]
    
    if image_t1_path and os.path.exists(image_t1_path):
        required_files.append((mni_t1_brain, "MNI T1 Brain Template"))
    else:
        if mra_stripped:
            required_files.append((mni_t1_brain, "MNI T1 Brain Template"))
        else:
            required_files.append((mni_t1_whole, "MNI T1 Whole Template"))

    for path, name in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}\n")

    os.makedirs(output_dir, exist_ok=True)

    return atlas_registration(
        mni_parcellation=mni_parcellation,
        mni_t1_brain=mni_t1_brain,
        mni_t1_whole=mni_t1_whole,
        image_path=image_path,
        output_dir=output_dir,
        image_t1=image_t1_path,
        preprocess_mra=preprocess_mra,
        preprocess_t1=preprocess_t1,
        mra_stripped=mra_stripped,
        t1_stripped=t1_stripped,
        generate_qc=generate_qc
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas to MRA Registration Module")
    
    parser.add_argument("--image", required=True, help="Path to the reference MRA image.")
    parser.add_argument("--t1", default=None, help="Path to the subject T1 image.")
    parser.add_argument("--parcellation", default=None, help="Path to custom MNI parcellation.")
    parser.add_argument("--template", default=None, help="Override path for the MNI template.")
    parser.add_argument("--output_dir", default=".", help="Directory to save outputs.")
    
    parser.add_argument("--preprocess_mra", action="store_true", help="Apply ANTs denoising + N4 bias correction to the MRA.")
    parser.add_argument("--preprocess_t1", action="store_true", help="Apply ANTs denoising + N4 bias correction to the T1.")

    parser.add_argument("--mra_stripped", action="store_true", help="Flag if the input MRA is already skull-stripped.")
    parser.add_argument("--t1_stripped", action="store_true", help="Flag if the input T1 is already skull-stripped.")

    parser.add_argument("--registration_snapshot", action="store_true",
                        help="Generate a PNG snapshot of the registered labels for Quality Control.")

    args = parser.parse_args()

    try:
        registered_atlas, inv_matrix = process_registration(
            image_path=args.image,
            output_dir=args.output_dir,
            mni_parcellation=args.parcellation,
            mni_t1_template=args.template,
            image_t1_path=args.t1,
            preprocess_mra=args.preprocess_mra,
            preprocess_t1=args.preprocess_t1,
            mra_stripped=args.mra_stripped,
            t1_stripped=args.t1_stripped,
            generate_qc=args.registration_snapshot
        )
        logger.info(f"Registration successful! Outputs saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        sys.exit(1)