import os
import argparse
import logging
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
    
    # Preprocessing flags
    parser.add_argument("--denoise", action="store_true", help="Apply ANTs DenoiseImage to the MRA.")
    parser.add_argument("--debias", action="store_true", help="Apply ANTs N4BiasFieldCorrection to the MRA.")
    
    args = parser.parse_args()

    # ---------------------------------------------------------
    # ATLAS COMPONENT RESOLUTION
    # ---------------------------------------------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_ATLAS_DIR = os.path.join(SCRIPT_DIR, "Atlas")

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
    logger.info("--- ATLAS CONFIGURATION ---")
    logger.info(f"Parcellation: {args.parcellation}")
    logger.info(f"Label Map:    {args.label_map}")
    logger.info(f"Template:     {args.template if args.template else 'Default Smart Routing (Brain/Whole)'}")
    logger.info("="*60)
    
    # ---------------------------------------------------------
    # STEP 1: ATLAS REGISTRATION
    # ---------------------------------------------------------
    logger.info("--- STEP 1: ATLAS REGISTRATION ---")
    registered_atlas_path = None
    try:
        # process_registration only exposes a single combined MRA-preprocessing
        # toggle (denoise+debias together), so either flag enables both.
        registered_atlas_path, inv_matrix_path = process_registration(
            image_path=args.image,
            output_dir=subject_out_dir,
            mni_parcellation=args.parcellation,
            mni_t1_template=args.template,
            image_t1_path=args.t1,
            preprocess_mra=(args.denoise or args.debias)
        )
        logger.info(f"Registration complete. Registered atlas saved to: {registered_atlas_path}")
    except Exception as e:
        logger.error(f"Pipeline failed during registration: {str(e)}")
        return

    # ---------------------------------------------------------
    # STEP 2: GRAPH EXTRACTION
    # ---------------------------------------------------------
    logger.info("--- STEP 2: VESSEL GRAPH EXTRACTION ---")
    try:
        graph_pkl_path = extract_graph(
            nifti_path=args.mask, 
            output_folder=subject_out_dir
        )
        logger.info(f"Graph extraction complete. Data saved to: {graph_pkl_path}")
    except Exception as e:
        logger.error(f"Pipeline failed during graph extraction: {str(e)}")
        return
        
    # ---------------------------------------------------------
    # STEP 3: METRIC COMPUTATION
    # ---------------------------------------------------------
    logger.info("--- STEP 3: METRIC COMPUTATION ---")
    try:
        metrics_to_compute = [
            'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
            'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
            'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
            'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
            'fit_rmse', 'num_loops', 'num_abnormal_degree_nodes'
        ]
        
        extract_metrics(
            output_folder=subject_out_dir,
            graph_pkl_path=graph_pkl_path,
            selected_metrics=metrics_to_compute,
            label_map_path=args.label_map,
            mask_path=args.mask,             # Passes the mask optionally for True Volume!
            registered_atlas_path=registered_atlas_path
        )
        logger.info("Metric computation complete.")
    except Exception as e:
        logger.error(f"Pipeline failed during metric computation: {str(e)}")
        return

    logger.info("="*60)
    logger.info("PIPELINE FINISHED SUCCESSFULLY.")
    logger.info("="*60)

if __name__ == "__main__":
    main()