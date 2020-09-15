set -ex

cuda=0
ex_name=eot
# model=attack_et_tar_gan
crop_size=256
load_size=286
ATTACK=10
niter=100
niter_decay=100

target=800

input_label=215
k_spt=1
k_qry=1
finetune_step=500

python train.py --meta_dataroot "image" --update_step 10 --gpu_ids $cuda --lambda_ATTACK_B $ATTACK --k_spt $k_spt --k_qry $k_qry --ori $input_label --target $target --name $ex_name --crop_size $crop_size --load_size $load_size --finetune_step $finetune_step
