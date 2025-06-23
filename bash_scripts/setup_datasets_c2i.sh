echo "Converting class conditional dataset for Oxford Flowers102"

python3 helper_scripts/convert_hf_c2i_dataset.py \
    --hf_id dpdl-benchmark/oxford_flowers102 \
    --splits train,test,validation \
    --out_dir out/data/processed/flowers102 \
    --img_col image \
    --label_col label \
    --num_workers 8

mv out/data/processed/flowers102/validation out/data/processed/flowers102/val

echo "Converting class conditional dataset for Stanford Cars196"

python3 helper_scripts/convert_hf_c2i_dataset.py \
    --hf_id tanganke/stanford_cars \
    --splits train,test \
    --out_dir out/data/processed/cars196 \
    --img_col image \
    --label_col label \
    --num_workers 8 \
    --split_val_from_train

echo "Converting class conditional dataset for Oxford IIIT Pet"

python3 helper_scripts/convert_hf_c2i_dataset.py \
    --hf_id timm/oxford-iiit-pet \
    --splits train,test \
    --out_dir out/data/processed/pet \
    --img_col image \
    --label_col label \
    --num_workers 8 \
    --split_val_from_train

echo "Converting class conditional dataset for Food 101"

python3 helper_scripts/convert_hf_c2i_dataset.py \
    --hf_id ethz/food101 \
    --splits train,validation \
    --out_dir out/data/processed/food101 \
    --img_col image \
    --label_col label \
    --num_workers 16

mv out/data/processed/food101/validation out/data/processed/food101/val

echo "Converting class conditional dataset for CUB-200-2011"

python3 helper_scripts/convert_hf_c2i_dataset.py \
    --hf_id Donghyun99/CUB-200-2011 \
    --splits train,test \
    --out_dir out/data/processed/cub200 \
    --img_col image \
    --label_col label \
    --num_workers 16 \
    --split_val_from_train

echo "All Dataset Preparation Done!"