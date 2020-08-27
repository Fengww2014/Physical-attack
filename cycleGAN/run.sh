set -ex

cuda=0
ex_name=eot1
model=attack_et_tar_gan
ATTACK=30
niter=100
niter_decay=100

target=800

input_label=215

CUDA_VISIBLE_DEVICES=$cuda no_proxy=localhost python train.py --dataroot ../trainImages/aligned_imgs/${input_label}*/ --model $model  --name ${ex_name}_ones_v6_${input_label}t${target}_b${ATTACK}_2 --target ${target} --ori ${input_label} --lambda_ATTACK_B ${ATTACK}  --netG resnet_6blocks    --save_epoch_freq 100  --niter ${niter} --niter_decay ${niter_decay} 
CUDA_VISIBLE_DEVICES=$cuda no_proxy=localhost python test.py --dataroot ../trainImages/aligned_imgs/${input_label}*/ --model $model  --name ${ex_name}_ones_v6_${input_label}t${target}_b${ATTACK}_2   --netG resnet_6blocks   
