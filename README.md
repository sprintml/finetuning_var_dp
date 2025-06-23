# Implementing Adaptations for Vision AutoRegressive Model

Codebase for implementing Private & Non-Private Finetuning adaptations for Vision AutoRegressive Model.

## Setup Environment

```bash
conda env create -f environment.yaml --solver=classic
conda activate dp_var
pip install ninja
pip install flash-attn --no-build-isolation
python3 -m pip install 'tensorflow[and-cuda]' # Evaluation requires TensorFlow
pip install pyarrow -U
pip install numpy -U
```

### Initilize Submodules

```bash
git submodule init
git submodule update
```

## Downloading Datasets

**List of Datasets**

- Stanford Cars 196
- CUB-200-2011
- Oxford Flowers
- Food101
- Oxford-IIIT Pet

```bash
mkdir out
mkdir out/data
mkdir out/data/processed/
bash bash_scripts/setup_datasets_c2i.sh
```

## Run Finetuning

```bash
bash bash_scripts/finetune_fft.sh # FFT
bash bash_scripts/finetune_lora.sh # LoRA
bash bash_scripts/finetune_lntuning.sh # LNTuning
```

## Running Evaluation

Steps for Evaluation:

- Generate samples with the finetuned model.
- Compile Reference and Generated sample into .npz files.
- Compute FID, sFID, Precision and Recall.

```bash
bash bash_scripts/generate_samples.sh
bash bash_scripts/scores_computation.sh
```

## Running Profiler Script

You can compute how many FLOPS/TFLOPs is each adaptation for a given model and dataset is using for a given set of hyperparameters.

```bash
for ft in var_fft var_lora var_lntuning; do
    python3 -u train.py \
        +model=var_16 \
        +finetuning=$ft \
        +trainer=var_profiler \
        +dataset=flowers102 +wandb.disable=True
done
```
