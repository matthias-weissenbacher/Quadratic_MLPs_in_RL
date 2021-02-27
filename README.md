# Quadratic MLPs in Reinforcement Learning

Source code and data for our accompanying paper "A  Quadratic  Actor  Network  for  Model-Free  Reinforcement  Learning"
TD3 and SAC algorithms with Quadratic - MLP (Q-MLP) as  actor policy network.  If you use our code or data please cite the paper.

# Rquirements
TD3 and SAC are tested on [Mujoco](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://gym.openai.com/). 
Neural Netorks are trained using Pytorch 1.4 and Python 3.7



# Usage
To run the test - with hyperparameter settings of the paper - run teh shell scripts in the SAC and TD3 folder, respectively.
```
./run_td3_nohup.sh
./run_sac_nohup.sh
```

# Acknowledgements
The TD3 and SAC code was based on ()
The TD3 code was based on [TD3](https://github.com/sfujim/TD3)
The SAC code was based on [SAC1](https://github.com/denisyarats/pytorch_sac) and [SAC2](https://github.com/vitchyr/rlkit).
