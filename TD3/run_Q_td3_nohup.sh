#!/bin/bash

# Script to reproduce results


num_devices=2 # Enter the number of available GPU's

for ((i=0;i<5;i+=1))
do



	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Ant-v3"  \
	--policy "Q_TD3" \
	--seed $i \
	--cuda_device $rand   \
	--hidden_dim 64 \
	--nf_fac 0.1 \
	--init_zeros "False" \
    &>  Ant_nohup.out &


	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "BipedalWalker-v3"  \
	--policy "Q_TD3" \
	--seed $i \
	--cuda_device $rand   \
	--hidden_dim 192 \
	--nf_fac 1.0 \
	--init_zeros "False" \
    &>  BPWalker_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "HalfCheetah-v3"  \
	--policy "Q_TD3" \
	--seed $i \
	--cuda_device $rand   \
	--hidden_dim 192 \
	--nf_fac 1.0 \
	--init_zeros "False" \
    &>  HalfCheetah_nohup.out &
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Hopper-v3"  \
	--policy "Q_TD3" \
	--seed $i \
	--cuda_device $rand   \
	--hidden_dim 128 \
	--nf_fac 1.0 \
	--init_zeros "False" \
    &>  Hopper_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Humanoid-v3"  \
	--policy "Q_TD3" \
	--seed $i \
	--cuda_device $rand   \
	--hidden_dim 192 \
	--nf_fac 0.02 \
	--init_zeros "True" \
    &>  Humanoid_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_td3.py \
	--env "Walker2d-v3"  \
	--policy "Q_TD3" \
	--seed $i \
	--cuda_device $rand   \
	--hidden_dim 64 \
	--nf_fac 1.0 \
	--init_zeros "False" \
    &>  Walker2d_nohup.out &
    


done


