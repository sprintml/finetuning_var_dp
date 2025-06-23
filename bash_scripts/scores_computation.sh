model=var_16 # var_20, var_24, #var_30
ft=var_fft # var_lora, var_lntuning
trainer=var_finetuning # dp-eps-10
ep=50
ckpt=10
dataset=flowers102 # cub200, cars196, food101, pet
split=test # val for food101

python3 -u main.py \
    +action=scores_computation \
    +model=$model \
    +finetuning=$ft \
    +trainer=$trainer trainer.checkpoint=$ckpt \
    +dataset=$dataset dataset.split=$split