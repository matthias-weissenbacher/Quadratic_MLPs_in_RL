#!/bin/bash

# Script to reproduce results

num_devices=2 # Enter the number of available GPU's

for ((i=0;i<5;i+=1))
do

	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Ant-v3"  \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
    &>  Ant_nohup.out &


	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "BipedalWalker-v3"  \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
    &>  BPWalker_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "HalfCheetah-v3"  \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
    &>  HalfCheetah_nohup.out &
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Hopper-v3"  \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
    &>  Hopper_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Humanoid-v3"  \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
    &>  Humanoid_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Walker2d-v3"  \
	--policy "SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
    &>  Walker2d_nohup.out &
    


done
