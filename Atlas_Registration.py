import logging
import os
import argparse
import subprocess

# Setup basic logging for standalone execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
#                             DEFAULT PATHS
# =========================================================================
# Dynamically locate the 'Atlas' directory relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ATLAS_DIR = os.path.join(SCRIPT_DIR, "Atlas")

# Now we define explicit paths for both the Brain-only and Whole-head templates
DEFAULT_MNI_T1_BRAIN = os.path.join(DEFAULT_ATLAS_DIR, "MNI152_T1_1mm_Brain.nii.gz")
DEFAULT_MNI_T1_WHOLE = os.path.join(DEFAULT_ATLAS_DIR, "MNI152_T1_1mm.nii.gz")
DEFAULT_PARCELLATION = os.path.join(DEFAULT_ATLAS_DIR, "ArterialAtlas.nii.gz") 

def run_cmd(cmd_list):
    """Helper function to run shell commands via subprocess."""
    cmd_str = ' '.join(cmd_list)
    logger.debug(f"Running: {cmd_str}")
    try:
        result = subprocess.run(
            cmd_list, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd_str}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Command failed: {cmd_str}\n{e.stderr}")

def preprocess_mra(mra_path: str, output_dir: str, do_denoise: bool, do_debias: bool) -> str:
    """Applies Denoising and/or N4 Debiasing to the MRA image using ANTs."""
    base_name = os.path.basename(mra_path).split('.')[0]
    current_mra = mra_path

    # 1. DENOISE (ANTs DenoiseImage - Non-Local Means)
    if do_denoise:
        logger.info("  [PREPROC] Denoising MRA using ANTs DenoiseImage...")
        denoised_mra = os.path.join(output_dir, f"{base_name}_denoised.nii.gz")
        run_cmd(["DenoiseImage", "-d", "3", "-i", current_mra, "-o", denoised_mra, "-n", "Rician"])
        current_mra = denoised_mra

    # 2. DEBIAS (ANTs N4BiasFieldCorrection)
    if do_debias:
        logger.info("  [PREPROC] Debiasing MRA using ANTs N4BiasFieldCorrection...")
        debiased_mra = os.path.join(output_dir, f"{base_name}_debiased.nii.gz")
        run_cmd(["N4BiasFieldCorrection", "-d", "3", "-i", current_mra, "-o", debiased_mra])
        current_mra = debiased_mra

    return current_mra

def atlas_registration(
    mni_parcellation: str,
    mni_t1_brain: str,    
    mni_t1_whole: str,    
    image_path: str,      # MRA
    output_dir: str = ".",
    image_t1: str = None, 
    denoise_mra: bool = False,
    debias_mra: bool = False
) -> tuple:
    
    # 1. Explicitly separate MRA and T1 prefixes
    mra_prefix = os.path.basename(image_path).split('.')[0]
    
    # --- MRA PREPROCESSING ---
    if denoise_mra or debias_mra:
        image_path = preprocess_mra(image_path, output_dir, denoise_mra, debias_mra)

    # ---------------------------------------------------------
    # ROUTE A: T1 IS AVAILABLE
    # ---------------------------------------------------------
    if image_t1 and os.path.exists(image_t1):
        t1_prefix = os.path.basename(image_t1).split('.')[0]
        
        # 0. ROBUST FOV
        logger.info(f"  [DEBUG 0/6] ROBUSTFOV: Removing neck tissue...")
        t1_fov = os.path.join(output_dir, f"{t1_prefix}_cropped_fov.nii.gz")
        run_cmd(["robustfov", "-i", image_t1, "-r", t1_fov, "-b", "180"])

        # 1. BRAIN EXTRACTION
        logger.info(f"  [DEBUG 1/6] BET: Extracting brain...")
        t1_brain = os.path.join(output_dir, f"{t1_prefix}_brain.nii.gz")
        run_cmd(["bet", t1_fov, t1_brain, "-m", "-f", "0.4", "-R"])

        # 2. REORIENT TO STANDARD
        logger.info(f"  [DEBUG 2/6] REORIENT: Aligning T1 orientation to standard...")
        t1_reoriented = os.path.join(output_dir, f"{t1_prefix}_reoriented.nii.gz")
        run_cmd(["fslreorient2std", t1_brain, t1_reoriented])

        # 3. T1 -> MRA
        logger.info(f"  [DEBUG 3/6] FLIRT: Registering Subject T1 to Subject MRA...")
        t1_in_mra_img = os.path.join(output_dir, f"{t1_prefix}_in_MRA.nii.gz")
        t1_to_mra_mat = os.path.join(output_dir, f"{t1_prefix}_to_MRA.mat")
        run_cmd([
            "flirt", "-in", t1_reoriented, "-ref", image_path, 
            "-out", t1_in_mra_img, "-omat", t1_to_mra_mat, 
            "-dof", "6", "-cost", "mutualinfo",
            "-searchrx", "-180", "180", 
            "-searchry", "-180", "180", 
            "-searchrz", "-180", "180"
        ])

        # 4. MNI -> T1
        logger.info(f"  [DEBUG 4/6] FLIRT: Registering MNI Brain Template to Subject T1...")
        mni_in_t1_img = os.path.join(output_dir, f"MNI_in_{t1_prefix}.nii.gz")
        mni_to_t1_mat = os.path.join(output_dir, f"MNI_to_{t1_prefix}.mat")
        run_cmd([
            "flirt", "-in", mni_t1_brain, "-ref", t1_reoriented, 
            "-out", mni_in_t1_img, "-omat", mni_to_t1_mat, 
            "-dof", "12", "-cost", "corratio"
        ])

        # 5. CONCATENATE MATRICES
        logger.info(f"  [DEBUG 5/6] CONVERT_XFM: Combining transforms (MNI -> T1 -> MRA)...")
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
        
        # 1. MNI -> MRA
        logger.info(f"  [DEBUG 1/2] FLIRT: Registering MNI Whole Head Template to Subject MRA...")
        mni_in_mra_img = os.path.join(output_dir, f"MNI_in_{mra_prefix}.nii.gz")
        combined_mat = os.path.join(output_dir, f"{mra_prefix}_MNI_to_MRA_combined.mat")
        run_cmd([
            "flirt", "-in", mni_t1_whole, "-ref", image_path, 
            "-out", mni_in_mra_img, "-omat", combined_mat, 
            "-dof", "12", "-cost", "normmi", "-bins", "32",
            "-searchrx", "-180", "180", 
            "-searchry", "-180", "180", 
            "-searchrz", "-180", "180"
        ])

    # ---------------------------------------------------------
    # COMMON FINAL STEPS
    # ---------------------------------------------------------
    # 5.5 CREATE INVERSE MATRIX
    logger.info(f"  [DEBUG 5.5/6] CONVERT_XFM: Inverting combined matrix (MRA -> MNI)...")
    mra_to_mni_mat = os.path.join(output_dir, f"{mra_prefix}_MRA_to_MNI_inverse.mat")
    run_cmd(["convert_xfm", "-omat", mra_to_mni_mat, "-inverse", combined_mat])

    # 6. FINAL WARP: MNI PARCELLATION -> MRA
    logger.info(f"  [DEBUG 6/6] APPLYXFM: Warping MNI Parcellation to MRA space...")
    reg_parcellation_path = os.path.join(output_dir, f"{mra_prefix}_registered_parcellation.nii.gz")
    run_cmd([
        "flirt", "-in", mni_parcellation, "-ref", image_path, 
        "-out", reg_parcellation_path, "-applyxfm", "-init", combined_mat, 
        "-interp", "nearestneighbour"
    ])

    # 7. EXPAND propagated parcellation
    logger.info(f"  [DEBUG 7/7] IMAGEMATHS: Dilating parcellation labels into the background...")
    dilated_parcellation_path = os.path.join(output_dir, f"{mra_prefix}_registered_parcellation_dilated.nii.gz")
    run_cmd([
        "fslmaths", reg_parcellation_path, 
        "-dilD", "-dilD", "-dilD", "-dilD", "-dilD", 
        dilated_parcellation_path
    ])

    return dilated_parcellation_path, mra_to_mni_mat

def process_registration(
    image_path, 
    output_dir, 
    mni_parcellation=None, 
    mni_t1_template=None, 
    image_t1_path=None, 
    denoise=False, 
    debias=False
):
    
    # --- FALLBACK LOGIC ---
    if mni_parcellation is None:
        mni_parcellation = DEFAULT_PARCELLATION

    # If the user provides a single custom template via CLI, we override both logic paths with it.
    # Otherwise, we use our smart dual-template system.
    if mni_t1_template is not None:
        mni_t1_brain = mni_t1_template
        mni_t1_whole = mni_t1_template
    else:
        mni_t1_brain = DEFAULT_MNI_T1_BRAIN
        mni_t1_whole = DEFAULT_MNI_T1_WHOLE

    # --- VERIFICATION ---
    # Validate the files we are actively going to use based on the route
    required_files = [(mni_parcellation, "MNI Parcellation")]
    if image_t1_path and os.path.exists(image_t1_path):
        required_files.append((mni_t1_brain, "MNI T1 Brain Template"))
    else:
        required_files.append((mni_t1_whole, "MNI T1 Whole Template"))

    for path, name in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at: {path}\n"
                "Please provide it via CLI or ensure the file exists in the 'Atlas/' folder."
            )

    os.makedirs(output_dir, exist_ok=True)

    return atlas_registration(
        mni_parcellation=mni_parcellation,
        mni_t1_brain=mni_t1_brain,
        mni_t1_whole=mni_t1_whole,
        image_path=image_path,
        output_dir=output_dir,
        image_t1=image_t1_path,
        denoise_mra=denoise,
        debias_mra=debias
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas to MRA Registration Module")
    
    # Required Image
    parser.add_argument("--image", required=True, help="Path to the reference MRA image.")
    
    # Optional Arguments
    parser.add_argument("--t1", default=None, help="Path to the subject T1 image (if omitted, direct MNI->MRA is used).")
    parser.add_argument("--parcellation", default=None, help="Path to custom MNI parcellation (defaults to Atlas/ArterialAtlas.nii.gz).")
    parser.add_argument("--template", default=None, help="Override path for the MNI template (disables smart Whole/Brain selection).")
    parser.add_argument("--output_dir", default=".", help="Directory to save outputs.")
    
    # ANTs Preprocessing flags
    parser.add_argument("--denoise", action="store_true", help="Apply ANTs DenoiseImage (Non-Local Means) to the MRA.")
    parser.add_argument("--debias", action="store_true", help="Apply ANTs N4BiasFieldCorrection to the MRA.")

    args = parser.parse_args()

    try:
        registered_atlas, inv_matrix = process_registration(
            image_path=args.image,
            output_dir=args.output_dir,
            mni_parcellation=args.parcellation,
            mni_t1_template=args.template,
            image_t1_path=args.t1,
            denoise=args.denoise,
            debias=args.debias
        )
        logger.info(f"Registration successful! Outputs saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Registration failed: {e}")