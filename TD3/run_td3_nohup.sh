#!/bin/bash

# Script to reproduce results


num_devices=2 # Enter the number of available GPU's

for ((i=0;i<5;i+=1))
do

	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Ant-v3"  \
	--policy "TD3" \
	--seed $i \
	--cuda_device $rand   \
    &>  Ant_nohup.out &


	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "BipedalWalker-v3"  \
	--policy "TD3" \
	--seed $i \
	--cuda_device $rand   \
    &>  BPWalker_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "HalfCheetah-v3"  \
	--policy "TD3" \
	--seed $i \
	--cuda_device $rand   \
    &>  HalfCheetah_nohup.out &
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Hopper-v3"  \
	--policy "TD3" \
	--seed $i \
	--cuda_device $rand   \
    &>  Hopper_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Humanoid-v3"  \
	--policy "TD3" \
	--seed $i \
	--cuda_device $rand   \
    &>  Humanoid_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Walker2d-v3"  \
	--policy "TD3" \
	--seed $i \
	--cuda_device $rand   \
    &>  Walker2d_nohup.out &
    


done


