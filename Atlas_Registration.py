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

# Set the default filenames expected inside the Atlas folder
DEFAULT_MNI_T1 = os.path.join(DEFAULT_ATLAS_DIR, "MNI152_T1_1mm_brain.nii.gz")
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
        run_cmd(["DenoiseImage", "-d", "3", "-i", current_mra, "-o", denoised_mra])
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
    mni_t1_template: str,
    image_path: str,      # MRA
    output_dir: str = ".",
    image_t1: str = None, # Optional T1
    denoise_mra: bool = False,
    debias_mra: bool = False
) -> tuple:
    
    # Base naming setup
    base_name = os.path.basename(image_path).split('.')[0]
    
    # --- MRA PREPROCESSING ---
    if denoise_mra or debias_mra:
        image_path = preprocess_mra(image_path, output_dir, denoise_mra, debias_mra)

    # ---------------------------------------------------------
    # ROUTE A: T1 IS AVAILABLE
    # ---------------------------------------------------------
    if image_t1 and os.path.exists(image_t1):
        t1_name = os.path.basename(image_t1).split('.')[0]
        base_name = t1_name # Override base name for consistency
        
        # 0. ROBUST FOV (NECK CROP)
        logger.info(f"  [DEBUG 0/6] ROBUSTFOV: Removing neck tissue...")
        t1_fov = os.path.join(output_dir, f"{t1_name}_cropped_fov.nii.gz")
        run_cmd(["robustfov", "-i", image_t1, "-r", t1_fov, "-b", "180"])

        # 1. BRAIN EXTRACTION
        logger.info(f"  [DEBUG 1/6] BET: Extracting brain...")
        t1_brain = os.path.join(output_dir, f"{t1_name}_BETted_brain.nii.gz")
        run_cmd(["bet", t1_fov, t1_brain, "-m", "-f", "0.4", "-R"])

        # 2. REORIENT TO STANDARD
        logger.info(f"  [DEBUG 2/6] REORIENT: Aligning T1 orientation to standard...")
        t1_reoriented = os.path.join(output_dir, f"{t1_name}_Reoriented.nii.gz")
        run_cmd(["fslreorient2std", t1_brain, t1_reoriented])

        # 3. T1 -> MRA (6-DOF)
        logger.info(f"  [DEBUG 3/6] FLIRT: Registering Subject T1 to Subject MRA...")
        t1_in_mra_img = os.path.join(output_dir, f"{t1_name}_T1_in_MRA.nii.gz")
        t1_to_mra_mat = os.path.join(output_dir, "t1_to_mra.mat")
        run_cmd([
            "flirt", "-in", t1_reoriented, "-ref", image_path, 
            "-out", t1_in_mra_img, "-omat", t1_to_mra_mat, 
            "-dof", "6", "-cost", "mutualinfo",
            "-searchrx", "-180", "180", 
            "-searchry", "-180", "180", 
            "-searchrz", "-180", "180"
        ])

        # 4. MNI -> T1 (12-DOF)
        logger.info(f"  [DEBUG 4/6] FLIRT: Registering MNI Template to Subject T1...")
        mni_in_t1_img = os.path.join(output_dir, f"{t1_name}_MNI_in_T1.nii.gz")
        mni_to_t1_mat = os.path.join(output_dir, "mni_to_t1.mat")
        run_cmd([
            "flirt", "-in", mni_t1_template, "-ref", t1_reoriented, 
            "-out", mni_in_t1_img, "-omat", mni_to_t1_mat, 
            "-dof", "12", "-cost", "corratio"
        ])

        # 5. CONCATENATE MATRICES
        logger.info(f"  [DEBUG 5/6] CONVERT_XFM: Combining transforms (MNI -> T1 -> MRA)...")
        combined_mat = os.path.join(output_dir, "combined_mni_to_mra.mat")
        run_cmd([
            "convert_xfm", "-omat", combined_mat, 
            "-concat", t1_to_mra_mat, mni_to_t1_mat
        ])

    # ---------------------------------------------------------
    # ROUTE B: NO T1 (DIRECT MNI -> MRA)
    # ---------------------------------------------------------
    else:
        logger.info("  [DEBUG] No T1 provided. Routing direct MNI to MRA registration...")
        
        # 1. MNI -> MRA (12-DOF)
        logger.info(f"  [DEBUG 1/2] FLIRT: Registering MNI Template to Subject MRA...")
        mni_in_mra_img = os.path.join(output_dir, f"{base_name}_MNI_in_MRA.nii.gz")
        combined_mat = os.path.join(output_dir, "combined_mni_to_mra.mat")
        run_cmd([
            "flirt", "-in", mni_t1_template, "-ref", image_path, 
            "-out", mni_in_mra_img, "-omat", combined_mat, 
            "-dof", "12", "-cost", "mutualinfo",
            "-searchrx", "-180", "180", 
            "-searchry", "-180", "180", 
            "-searchrz", "-180", "180"
        ])

    # ---------------------------------------------------------
    # COMMON FINAL STEPS
    # ---------------------------------------------------------
    # 5.5 CREATE INVERSE MATRIX (MRA -> MNI)
    logger.info(f"  [DEBUG 5.5/6] CONVERT_XFM: Inverting combined matrix (MRA -> MNI)...")
    mra_to_mni_mat = os.path.join(output_dir, "mra_to_mni.mat")
    run_cmd(["convert_xfm", "-omat", mra_to_mni_mat, "-inverse", combined_mat])

    # 6. FINAL WARP: MNI PARCELLATION -> MRA
    logger.info(f"  [DEBUG 6/6] APPLYXFM: Warping MNI Parcellation to MRA space...")
    reg_parcellation_path = os.path.join(output_dir, f"{base_name}_registered_parcellation.nii.gz")
    run_cmd([
        "flirt", "-in", mni_parcellation, "-ref", image_path, 
        "-out", reg_parcellation_path, "-applyxfm", "-init", combined_mat, 
        "-interp", "nearestneighbour"
    ])

    # 7. EXPAND propagated parcellation
    logger.info(f"  [DEBUG 7/7] IMAGEMATHS: Dilating parcellation labels into the background...")
    dilated_parcellation_path = os.path.join(output_dir, f"{base_name}_registered_parcellation_dilated.nii.gz")
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
    if mni_t1_template is None:
        mni_t1_template = DEFAULT_MNI_T1
    if mni_parcellation is None:
        mni_parcellation = DEFAULT_PARCELLATION

    # --- VERIFICATION ---
    if not os.path.exists(mni_t1_template):
        raise FileNotFoundError(
            f"MNI T1 Template not found at: {mni_t1_template}\n"
            "Please provide it via CLI or place the file in the 'Atlas/' folder."
        )
        
    if not os.path.exists(mni_parcellation):
        raise FileNotFoundError(
            f"MNI Parcellation not found at: {mni_parcellation}\n"
            "Please provide it via CLI or place the file in the 'Atlas/' folder."
        )

    os.makedirs(output_dir, exist_ok=True)

    return atlas_registration(
        mni_parcellation=mni_parcellation,
        mni_t1_template=mni_t1_template,
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
    
    # Defaults for MNI files
    parser.add_argument("--parcellation", default=None, help="Path to the MNI parcellation file (defaults to Atlas/ArterialAtlas.nii.gz).")
    parser.add_argument("--template", default=None, help="Path to the MNI T1 template (defaults to Atlas/MNI152_T1_1mm_brain.nii.gz).")
    
    # Optional Arguments
    parser.add_argument("--t1", default=None, help="Path to the subject T1 image (if omitted, direct MNI->MRA is used).")
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