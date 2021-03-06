#!/bin/bash

# Script to reproduce results

num_devices=2 # Enter the number of available GPU's

for ((i=0;i<5;i+=1))
do



	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Ant-v3"  \
	--policy "Q_SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
	--hidden_dim 128 \
	--nf_fac 0.1   \
	--init_zeros "False"   \
    &>  AntQ_nohup.out &


	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "BipedalWalker-v3"  \
	--policy "Q_SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
	--hidden_dim 192 \
	--nf_fac 1.0   \
	--init_zeros "False"   \
    &>  BPWalkerQ_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "HalfCheetah-v3"  \
	--policy "Q_SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
	--hidden_dim 192 \
	--nf_fac 1.0   \
	--init_zeros "False"   \
    &>  HalfCheetahQ_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Hopper-v3"  \
	--policy "Q_SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
	--hidden_dim 128 \
	--nf_fac 1.0   \
	--init_zeros "False"   \
    &>  HopperQ_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Humanoid-v3"  \
	--policy "Q_SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
	--hidden_dim 224 \
	--nf_fac 0.02   \
	--init_zeros "True"   \
    &>  HumanoidQ_nohup.out &
    
    
	rand=$(( ( RANDOM % $num_devices )  )) # 
	nohup python3 main_sac.py \
	--env "Walker2d-v3"  \
	--policy "Q_SAC" \
	--seed $i \
	--automatic_entropy_tuning True \
	--batch_size 100 \
	--cuda_device $rand   \
	--hidden_dim 128 \
	--nf_fac 1.0   \
	--init_zeros "False"   \
    &>  Walker2dQ_nohup.out &
    


done
