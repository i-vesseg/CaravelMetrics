import os
import argparse
import logging
import sys
from pathlib import Path
from Atlas_Registration import process_registration
from Graph_extractor import extract_graph
from Metric_extractor import extract_metrics

# Setup console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Caravel - Single Subject Processing Pipeline")
    
    # Required inputs
    parser.add_argument("-i", "--image", required=True, help="Path to the subject MRA image.")
    parser.add_argument("-m", "--mask", required=True, help="Path to the binary vessel mask.")
    parser.add_argument("-o", "--output_dir", required=True, help="Base output directory.")
    
    # Optional registration arguments
    parser.add_argument("--t1", default=None, help="Path to the subject T1 image (optional).")
    parser.add_argument("--template", default=None, help="Path to custom MNI T1 template (optional).")
    parser.add_argument("--parcellation", default=None, help="Path to custom MNI parcellation (optional).")
    parser.add_argument("--label_map", default=None, help="Path to ArterialAtlasLables.txt for regional naming (optional).")
    parser.add_argument("--no_atlas", action="store_true",
                        help="Skip atlas registration; graph and metric steps still run, "
                             "but without regional metrics.")
    
    # Preprocessing flags
    parser.add_argument("--preprocess_mra", action="store_true",
                        help="Apply ANTs denoising + N4 bias correction to the MRA.")
    parser.add_argument("--preprocess_t1", action="store_true",
                        help="Apply ANTs denoising + N4 bias correction to the T1.")

    # Skip skull-stripping if inputs are already stripped
    parser.add_argument("--mra_stripped", action="store_true", help="Flag if the input MRA is already skull-stripped.")
    parser.add_argument("--t1_stripped", action="store_true", help="Flag if the input T1 is already skull-stripped.")

    # Graph extraction tuning
    parser.add_argument("--min_branch_len", type=float, default=2.0,
                        help="Prune terminal branches (leaf-to-bifurcation hairs) shorter than "
                             "this many mm as skeletonization spurs (default: 2.0).")
    parser.add_argument("--target_iso", type=float, default=None,
                        help="Isotropic spacing (mm) to resample every mask to (default: each "
                             "scan's finest axis). Set one value for a whole cohort so all "
                             "subjects are processed on the same grid. Note: it cannot undo "
                             "acquisition differences -- bifurcation counts still change "
                             "with scan resolution.")
    parser.add_argument("--no_extend_tips", action="store_true",
                        help="Do not extend skeleton tips that stop short of their vessel end (on by default).")
    parser.add_argument("--mask_smoothing", type=float, default=0.0,
                        help="Gaussian smoothing (mm) of the mask boundary before "
                             "skeletonisation, for noisy segmentations only (default: 0 = off; "
                             "0.25 validated; raised to at least 0.7 voxel; erodes vessels thinner "
                             "than ~1 voxel).")

    # QC reporting
    parser.add_argument("--html_qc", action="store_true",
                        help="Also write an interactive HTML QC report for the extracted graph.")
    parser.add_argument("--qc_dir", default=None,
                        help="Where to write the QC report (default: the subject's own output folder).")
    parser.add_argument("--registration_snapshot", action="store_true",
                        help="Also write a PNG snapshot of the registered atlas labels for QC.")
    
    args = parser.parse_args()

    # ---------------------------------------------------------
    # ATLAS COMPONENT RESOLUTION
    # ---------------------------------------------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_ATLAS_DIR = os.path.join(SCRIPT_DIR, "Atlas")

    if not args.no_atlas:
        if args.parcellation is None:
            args.parcellation = os.path.join(DEFAULT_ATLAS_DIR, "ArterialAtlas.nii.gz")

        if args.label_map is None:
            args.label_map = os.path.join(DEFAULT_ATLAS_DIR, "ArterialAtlasLables.txt")
        
    # Note: args.template is deliberately left as None if omitted. 
    # This allows Atlas_Registration.py to use its smart Brain/Whole head routing.
    
    # 1. Setup Patient-Specific Output Folder
    patient_name = os.path.basename(args.image)
    while True:
        name, ext = os.path.splitext(patient_name)
        if ext and name:
            patient_name = name
        else:
            break
            
    subject_out_dir = os.path.join(args.output_dir, patient_name)
    Path(subject_out_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup File Logger for this specific subject
    file_handler = logging.FileHandler(os.path.join(subject_out_dir, 'processing.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*60)
    logger.info(f"STARTING CARAVEL PIPELINE FOR: {patient_name}")
    logger.info(f"Output directory: {subject_out_dir}")
    if args.no_atlas:
        logger.info("--- ATLAS CONFIGURATION: DISABLED (--no_atlas) ---")
    else:
        logger.info("--- ATLAS CONFIGURATION ---")
        logger.info(f"Parcellation: {args.parcellation}")
        logger.info(f"Label Map:    {args.label_map}")
        logger.info(f"Template:     {args.template if args.template else 'Default Smart Routing (Brain/Whole)'}")
    logger.info("="*60)

    # ---------------------------------------------------------
    # STEP 1: ATLAS REGISTRATION (optional)
    # ---------------------------------------------------------
    registered_atlas_path = None
    if args.no_atlas:
        logger.info("--- STEP 1: ATLAS REGISTRATION SKIPPED (--no_atlas) ---")
    else:
        logger.info("--- STEP 1: ATLAS REGISTRATION ---")
        try:
            registered_atlas_path, _ = process_registration(
                image_path=args.image,
                output_dir=subject_out_dir,
                mni_parcellation=args.parcellation,
                mni_t1_template=args.template,
                image_t1_path=args.t1,
                preprocess_mra=args.preprocess_mra,
                preprocess_t1=args.preprocess_t1,
                mra_stripped=args.mra_stripped,
                t1_stripped=args.t1_stripped,
                generate_qc=args.registration_snapshot,
            )
            logger.info(f"Registration complete. Registered atlas saved to: {registered_atlas_path}")
        except Exception as e:
            logger.error(f"Pipeline failed during registration: {str(e)}")
            sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: GRAPH EXTRACTION
    # ---------------------------------------------------------
    logger.info("--- STEP 2: VESSEL GRAPH EXTRACTION ---")
    try:
        graph_pkl_path = extract_graph(
            nifti_path=args.mask,
            output_folder=subject_out_dir,
            min_branch_length_mm=args.min_branch_len,
            target_iso_spacing=args.target_iso,
            mask_smoothing_mm=args.mask_smoothing,
            extend_tips=not args.no_extend_tips
        )
        logger.info(f"Graph extraction complete. Data saved to: {graph_pkl_path}")
    except Exception as e:
        logger.error(f"Pipeline failed during graph extraction: {str(e)}")
        sys.exit(1)
        
    # ---------------------------------------------------------
    # STEP 3: METRIC COMPUTATION
    # ---------------------------------------------------------
    logger.info("--- STEP 3: METRIC COMPUTATION ---")
    try:
        extract_metrics(
            output_folder=subject_out_dir,
            graph_pkl_path=graph_pkl_path,
            label_map_path=args.label_map,
            mask_path=args.mask,             # Passes the mask optionally for True Volume!
            registered_atlas_path=registered_atlas_path
        )
        logger.info("Metric computation complete.")
    except Exception as e:
        logger.error(f"Pipeline failed during metric computation: {str(e)}")
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 4: QC REPORT (optional)
    # ---------------------------------------------------------
    if args.html_qc:
        logger.info("--- STEP 4: QC REPORT ---")
        # Imported here rather than at module scope so the pipeline still runs
        # for anyone without plotly installed, as long as they don't ask for QC.
        try:
            from QC_report import generate_qc_report
            qc_dir = args.qc_dir or subject_out_dir
            html_path, _ = generate_qc_report(
                graph_pkl_path=graph_pkl_path,
                output_dir=qc_dir,
                mask_path=args.mask,
                atlas_path=registered_atlas_path,
                label_map_path=args.label_map,
            )
            logger.info(f"QC report written to: {html_path}")
        except Exception as e:
            # A failed report must not invalidate the metrics already written.
            logger.error(f"QC report generation failed (metrics are unaffected): {str(e)}")

    logger.info("="*60)
    logger.info("PIPELINE FINISHED SUCCESSFULLY.")
    logger.info("="*60)

if __name__ == "__main__":
    main()