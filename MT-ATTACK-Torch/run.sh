set -ex

cuda=5,1,6,7
ex_name=eot
# model=attack_et_tar_gan

ATTACK=30
niter=100
niter_decay=100

target=800

input_label=215

python train.py --meta_dataroot "image" --update_step 50 --gpu_ids $cuda --lambda_ATTACK_B 1 --k_spt 1 --k_qry 1 --ori $input_label --target $target --name $ex_name
