python Caravel.py \
    --image /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/MRA/IXI023-Guys-0699-MRA.nii.gz \
    --mask /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/masks/IXI023-Guys-0699-MRA.nii.gz \
    --output_dir /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Results/ \
    --t1 /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/T1/IXI023-Guys-0699-T1.nii.gz 

python Caravel.py \
    --image /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/MRA/IXI002-Guys-0828-MRA.nii.gz \
    --mask /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/masks/IXI002-Guys-0828-MRA.nii.gz \
    --output_dir /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Results/ \
    --t1 /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/T1/IXI002-Guys-0828-T1.nii.gz 

python Caravel.py \
    --image /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/MRA/IXI015-HH-1258-MRA.nii.gz \
    --mask /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/masks/IXI015-HH-1258-MRA.nii.gz \
    --output_dir /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Results/ \
    --t1 /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/T1/IXI015-HH-1258-T1.nii.gz 


#python Metric_extractor.py \
#    --graph /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Results_without_t1/IXI002-Guys-0828-MRA/IXI002-Guys-0828-MRA_vessel_data.pkl \
#    --out_dir /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Results_without_t1/IXI002-Guys-0828-MRA \
#    --mask /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/masks/IXI002-Guys-0828-MRA.nii.gz \
#    --label_map /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Atlas/ArterialAtlasLables.txt \
#    --atlas /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/Results_without_t1/IXI002-Guys-0828-MRA/IXI002-Guys-0828-MRA_registered_parcellation_dilated.nii.gz \


#python Graph_extractor.py \
#    --mask /homes/cazzolla/EURECOM/repos/Caravel_Mattia/CaravelMetrics/IXI/masks/IXI002-Guys-0828-MRA.nii.gz \
#    --output_dir ./Results_without_t1/IXI002-Guys-0828-MRA \
