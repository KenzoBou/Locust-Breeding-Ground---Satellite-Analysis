# Locust-Breeding-Ground---Satellite-Analysis
This repository was originally created during the GEO AI Hackathon 2025, co-organised by Instadeep and Datacraft. It fine-tunes Prithvi to make segmentation prediction to early detect locust breeding grounds in Afriaca with HLS data. 


The code features a complete pipeline for processing HLS satellite imagery, computing spectral indices, fine-tuning InstaDeep’s PrithviSeg model via InstaGeo, and producing locust-presence predictions.

## Repository Structure



## Methodology

1. **Data Subsetting**  
   Selected the most recent 25% of training imagery chips and their segmentation maps to manage storage and preserve temporal order.

2. **Band Replacement**  
   I replaced the SWIR2 band (channel 5 of each time step) with NDVI to match the six-band input expected by Prithvi’s patch embedding.  
   ```python
   # NDVI = (NIR – Red) / (NIR + Red + 1e-6)
   ndvi = (nir - red) / (nir + red + 1e-6)
This in-place modification allowed training on NDVI without increasing file size or chip count.

## Train/Validation Split
Generated train_ds_subset.csv listing chip and seg_map paths, then created a 70/30 split into train_split.csv and validation_split.csv.

## Model Training
Fine-tuned the PrithviSeg backbone on the transformed six-band inputs (Blue, Green, Red, NIR, NDVI, SWIR1) using Hydra:

"python scripts/train_instageo.py \
  --config configs/locust.yaml \
  --root_dir . \
  --train_csv train_split.csv \
  --val_csv validation_split.csv \
  --epochs 10 \
  --batch_size 8"
Evaluation & Inference
Ran validation and generated test-set predictions:

bash
Copier
Modifier
## Validation
python scripts/inference_instageo.py \
  --config configs/locust.yaml \
  --root_dir . \
  --test_csv validation_split.csv \
  --checkpoint outputs/first_run/instageo_best_checkpoint.ckpt \
  --mode eval

## Test-set inference
python scripts/inference_instageo.py \
  --config configs/locust.yaml \
  --root_dir . \
  --test_csv test_ds.csv \
  --checkpoint outputs/first_run/instageo_best_checkpoint.ckpt \
  --output_dir predictions \
  --mode chip_inference
  
## Results
Training on 25% of data with NDVI in place of SWIR2 achieved performance close to the baseline trained on 100% of data with original six bands.

The slight drop in accuracy was expected given fewer samples and modified inputs.

### Next Steps

Experiment with combining NDVI and SWIR1 for richer spectral input.

Scale up to 50–100% of the dataset now that in-place transforms avoid duplication.

Tune hyperparameters and add data augmentations.

Apply cross-validation and ensembling to improve stability.
