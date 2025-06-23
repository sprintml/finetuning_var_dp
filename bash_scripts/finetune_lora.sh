model=var_16 # var_20, var_24, #var_30
ft=var_lora
trainer=var_finetuning # dp-eps-10
ep=50
ckpt=10
dataset=flowers102 # cub200, cars196, food101, pet

python3 -u train.py \
    +model=$model \
    +finetuning=$ft \
    +trainer=$trainer trainer.epochs=$ep trainer.checkpoint=$ckpt \
    +dataset=$dataset # +wandb.disable=True # +wandb.run_name="custom_run_name_here"