set -ex

cuda=0
ex_name=eot
# model=attack_et_tar_gan
crop_size=256
load_size=286
ATTACK=20 # 就是临界值30不清晰能攻击成功、20攻击不成功
niter=100
niter_decay=100

target=362

input_label=215
k_spt=1
k_qry=1
finetune_step=4000 # 5、6000不行
dist=20 # 10易产生模糊、15能在3500~4000较好
update_step=100
save_latest_freq=100
task_num=1
load_iter=100

python test.py --meta_dataroot "image" --update_step $update_step --gpu_ids $cuda --lambda_ATTACK_B $ATTACK  \
--k_spt $k_spt --k_qry $k_qry --ori $input_label --target $target --name $ex_name --crop_size $crop_size  \
--load_size $load_size --finetune_step $finetune_step --lambda_dist $dist --task_num $task_num \
--save_latest_freq $save_latest_freq --load_iter $load_iter 
