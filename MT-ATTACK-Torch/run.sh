set -ex

cuda=0
ex_name=eot1
# model=attack_et_tar_gan

ATTACK=30
niter=100
niter_decay=100

# target=800

# input_label=215

python train.py --meta_dataroot "image" --update_step 2
