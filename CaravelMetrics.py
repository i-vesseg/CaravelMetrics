import os
import sys
import traceback
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from Atlas_Registration import process_registration
from Graph_extractor import extract_graph
from Metric_extractor import extract_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set paths
segmentation_folder = os.path.join('path/to/nnUNet')
image_folder = os.path.join('path/to/00_ORIGINAL_Dataset')
t1_image_folder = os.path.join('path/to/IXI_T1/sources')

atlas_path = 'ArterialAtlas.nii.gz'
label_map_path = 'ArterialAtlas_labels.txt'
output_base_folder = 'Results/'

# Selected metrics to compute
selected_metrics = [ 'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
        'fit_rmse', 'num_loops', 'num_abnormal_degree_nodes' ]

# Control whether to skip existing outputs or overwrite them
skip_existing = False  # Set to False to overwrite existing results (True to skip)

# Control whether to run in parallel or sequentially
run_parallel = True  # Set to False to run sequentially (useful for debugging)

# Number of parallel workers (None = auto-detect, use cpu_count() - 1)
num_workers = None  # Set to specific number to override auto-detection

# Chunk size for parallel processing (smaller = better progress updates, larger = less overhead)
chunk_size = 1  # Process files one at a time for better error isolation

# Control whether to use atlas registration step or treat segmentation as is (1 single region)
use_atlas = False  # Whether to use atlas registration step

###################################################################
#                   DO NOT EDIT BELOW THIS LINE       
###################################################################
# All available metric keys
all_keys = [ 'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
        'fit_rmse', 'num_loops', 'num_abnormal_degree_nodes' ]


invalid = set(selected_metrics) - set(all_keys)
if invalid: 
    raise ValueError(f"Invalid metrics requested: {invalid}")


def process_single_file(fname, image_folder, t1_image_folder, segmentation_folder, atlas_path, output_base_folder, skip_existing):
    """Process a single segmentation file with comprehensive error handling"""
    
    try:
        # Remove double extensions (e.g., .nii.gz, .tar.gz) or single extensions
        patient_name = fname
        while True:
            name, ext = os.path.splitext(patient_name)
            if ext and name:
                patient_name = name
            else:
                break
            
        # Define paths (image, mask, output folder)
        mask_path = os.path.join(segmentation_folder, fname)
        image_path = os.path.join(image_folder, fname)
        output_folder = os.path.join(output_base_folder, patient_name)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"Processing file: {fname} (OUTPUT: {output_folder})")
        logger.info("="*60)
        
        # Determine T1 image path
        t1_fname = fname.replace('MRA', 'T1') # File name of the corresponding T1 image (with extension)
        t1_patient_name = patient_name.replace('MRA', 'T1') # Patient name for T1 image (without extension)    
        t1_image_path = os.path.join(t1_image_folder, t1_fname)
        
        # Validate input files exist
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
        if not os.path.exists(t1_image_path):
            logger.warning(f"T1 image file not found: {t1_image_path}")
    
    
        #=====================================================================================================#
        #                                      1 - Atlas registration
        #=====================================================================================================#
        
        if use_atlas:
            registered_atlas_path = os.path.join(output_folder, f"{t1_patient_name}_registered_atlas.nii.gz")
            if skip_existing and os.path.exists(registered_atlas_path):
                logger.info(f"  Skipped ATLAS REGISTRATION: {fname} (output already exists)")
            else:
                logger.info(f"  Starting ATLAS REGISTRATION for: {fname}")
                process_registration(image_path, t1_image_path, mask_path, atlas_path, output_folder)
                logger.info(f"  REGISTERED ATLAS saved at: {registered_atlas_path}")
    
    
        #=====================================================================================================#
        #                                      2 - Vessel graph extraction
        #=====================================================================================================#
        
        graph_pkl_path = os.path.join(output_folder, "vessel_data.pkl")
        if skip_existing and os.path.exists(graph_pkl_path):
            logger.info(f"  Skipped VESSEL GRAPH EXTRACTION: {fname} (output already exists)")
        else:
            logger.info(f"  Starting VESSEL GRAPH EXTRACTION for: {fname}")
            extract_graph(mask_path, output_folder)
            logger.info(f"  Vessel graph extracted and saved in: {output_folder}")
        
        
        #=====================================================================================================#
        #                                      3 - Metrics computation
        #=====================================================================================================#
        parent, child = os.path.split(output_folder.rstrip('/'))
        output_global_folder = os.path.join(parent + ("_GLOBAL"), child)
        metrics_output_path = os.path.join(output_folder, "all_components_by_region.csv") if use_atlas else os.path.join(output_global_folder, "all_components_by_region.csv")
        
        if skip_existing and os.path.exists(metrics_output_path):
            logger.info(f"  Skipped METRICS COMPUTATION: {fname} (output already exists)")
        else:
            logger.info(f"  Starting METRICS COMPUTATION for: {fname}")
            extract_metrics(patient_name, output_folder, graph_pkl_path, selected_metrics, label_map_path, registered_atlas_path=registered_atlas_path if use_atlas else None)
            logger.info(f"  Metrics computed and saved in: {output_folder}")
        
        #=====================================================================================================#
        #                                      4 - Done
        #=====================================================================================================#
        logger.info(f"Successfully completed processing: {fname}")
        return {"status": "completed", "file": fname, "error": None}
        
    except Exception as e:
        error_msg = f"Failed to process {fname}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"status": "failed", "file": fname, "error": str(e)}

def main():
    # Get list of files to process
    files_to_process = [fname for fname in os.listdir(segmentation_folder) if fname.endswith(('.nii', '.nii.gz', '.npy', '.npz'))]
    
    if not files_to_process:
        logger.warning("No segmentation files found!")
        return
    
    logger.info(f"Found {len(files_to_process)} files to process")
    logger.info(f"Configuration: skip_existing={skip_existing}, run_parallel={run_parallel}")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_file,
        image_folder=image_folder,
        t1_image_folder=t1_image_folder,
        segmentation_folder=segmentation_folder,
        atlas_path=atlas_path,
        output_base_folder=output_base_folder,
        skip_existing=skip_existing
    )
    
    # Process files
    if run_parallel:
        # Determine number of workers
        if num_workers is None:
            # Use all CPUs except 1, but at least 1
            max_workers = max(1, cpu_count() - 1)
        else:
            max_workers = num_workers
        
        # Don't use more workers than files
        max_workers = min(max_workers, len(files_to_process))
        logger.info(f"Using {max_workers} parallel workers (total CPUs: {cpu_count()})")
        logger.info(f"Chunk size: {chunk_size}")
        
        # Process files in parallel with better error handling
        try:
            with Pool(processes=max_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, files_to_process, chunksize=chunk_size),
                    total=len(files_to_process),
                    desc="Processing files",
                    unit="file"
                ))
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            pool.terminate()
            pool.join()
            return
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            logger.error(traceback.format_exc())
            return
    else:
        logger.info("Running sequentially (parallel processing disabled)")
        
        # Process files sequentially
        results = []
        for fname in tqdm(files_to_process, desc="Processing files", unit="file"):
            try:
                result = process_func(fname)
                results.append(result)
            except KeyboardInterrupt:
                logger.warning("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {fname}: {e}")
                logger.error(traceback.format_exc())
                results.append({"status": "failed", "file": fname, "error": str(e)})
    
    # Print results summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY:")
    logger.info("="*60)
    
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    logger.info(f"Successfully completed: {completed}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")
    
    if completed > 0:
        success_rate = (completed / len(results)) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Show failed files if any
    failed_results = [r for r in results if r["status"] == "failed"]
    if failed_results:
        logger.error("\nFailed files:")
        for failure in failed_results:
            logger.error(f"  - {failure['file']}: {failure['error']}")
        logger.error(f"\nCheck processing.log for detailed error information")
    
    logger.info(f"\nAll results saved in: {os.path.abspath(output_base_folder)}")
    logger.info(f"Log file: processing.log")
    
    return completed, failed

if __name__ == "__main__":
    main()