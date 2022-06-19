# Robust Deep Reinforcement Learning through Bootstrapped Opportunistic Curriculum

This repository contains the code for our paper:
_Junlin Wu and Yevgeniy Vorobeychik, "Robust Deep Reinforcement Learning through Bootstrapped Opportunistic Curriculum", ICML 2022_

We propose **Bootstrapped Opportunistic Adversarial Curriculum Learning (BCL)**, a novel flexible adversarial curriculum learning framework for robust reinforcement learning.

Our framework combines two ideas: conservatively bootstrapping each curriculum phase with highest quality solutions obtained from multiple runs of the previous phase, and opportunistically skipping forward in the curriculum.

In our experiments we show that the proposed BCL framework enables dramatic improvements in robustness of learned policies to adversarial perturbations.
The greatest improvement is for Pong, where our framework yields robustness to perturbations of up to 25/255; in contrast, the best existing approach can only tolerate adversarial noise up to 5/255.

In summary, we make the following contributions:

1. A novel flexible adversarial curriculum learning framework for reinforcement learning (BCL), in which bootstrapping each phase from multiple executions of previous phase plays a key role,
2. A novel opportunistic adaptive generation variant that opportunistically skips forward in the curriculum,
3. An approach that composes interval bound propagation and FGSM-based adversarial input generation as a part of adaptive curriculum generation, and
4. An extensive experimental evaluation using **OpenAI Gym Atari games (DQN-style)** and **Procgen (PPO-style)** that demonstrates significant improvement in robustness due to the proposed BCL framework.


# Requirements
To install requirements:

```sh
pip install -r requirements.txt
```

Python 3.7+ is required.

# Curriculum Learning for DQN models
For DQN-style models we experiment on four Atari-2600 games: Pong, Freeway, BankHeist and RoadRunner. Our implementation is based on [RADIAL-DQN](https://github.com/tuomaso/radial_rl_v2/tree/main/Atari). The code is in the `DQN_Atari/` folder and runs on GPU by default.

## Pre-trained Models
Pre-trained models are available [here](https://drive.google.com/drive/folders/1oxvjRiylpCjiqTXra5NnmcJqDn1qAVxr?usp=sharing). It contains 
1. DQN (Vanilla) -- named as "XXX-natural_sadqn.pt". 
2. RADIAL-DQN -- named as "XXX_radial_dqn.pt". 
3. Median results of our BCL models (i.e., the ones reported in Table 2 of the main paper) -- named as BCL-XXX, the same as the model names in Table 2.

Put the pre-trained models in the corresponding `trained_models/` folder for curriculum training or evaluation.

## Training
To perform curriculum learning, update `--load-path` with the model we want to bootstrap from. The starting epsilon of the curriculum can be set through `--attack-epsilon-start`, and the increment of the curriculum can be set through `--attack-epsilon-end` (default 1.0/255). Throughout the training, the adversarial perturbation size will increase from `attack-epsilon-start`/255 to `attack-epsilon-start`/255+`attack-epsilon-end`.

```sh
# AT-DQN/NCL-AT-DQN (Pong/Freeway/BankHeist, BCL_DQN_Atari/BCL_AT folder)
python main.py --env PongNoFrameskip-v4 --robust --load-path "trained_models/Pong-natural_sadqn.pt" --linear-kappa --kappa-end 0.5 --attack-epsilon-start 0.0
python main.py --env FreewayNoFrameskip-v4 --robust --load-path "trained_models/Freeway-natural_sadqn.pt" --linear-kappa --kappa-end 0.5 --attack-epsilon-start 0.0
python main.py --env BankHeistNoFrameskip-v4 --robust --load-path "trained_models/BankHeist-natural_sadqn.pt" --linear-kappa --kappa-end 0.5 --attack-epsilon-start 0.0

# AT-DQN/NCL-AT-DQN (RoadRunner, BCL_DQN_Atari/BCL_AT_v1 folder)
python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunnerNoFrameskip-v4_trained.pt" --attack-epsilon-start 0.0

# NCL/BCL-RADIAL-DQN (Pong/Freeway/BankHeist/RoadRunner, BCL_DQN_Atari/BCL_RADIAL folder)
python main.py --env PongNoFrameskip-v4 --robust --load-path "trained_models/Pong_radial_dqn.pt" --attack-epsilon-start 1.0
python main.py --env FreewayNoFrameskip-v4 --robust --load-path "trained_models/Freeway_radial_dqn.pt" --attack-epsilon-start 1.0
python main.py --env BankHeistNoFrameskip-v4 --robust --load-path "trained_models/BankHeist_radial_dqn.pt" --attack-epsilon-start 1.0
python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunner_radial_dqn.pt" --adam-eps 0.00015 --attack-epsilon-start 1.0

# BCL-C/MOS-AT-DQN (Pong/Freeway/BankHeist, BCL_DQN_Atari/BCL_AT folder)
python main.py --env PongNoFrameskip-v4 --robust --load-path "trained_models/Pong_radial_dqn.pt" --attack-epsilon-start 3.0
python main.py --env FreewayNoFrameskip-v4 --robust --load-path "trained_models/Freeway_radial_dqn.pt" --attack-epsilon-start 3.0
python main.py --env BankHeistNoFrameskip-v4 --robust --load-path "trained_models/BankHeist_radial_dqn.pt" --attack-epsilon-start 3.0

# BCL-C/MOS-AT-DQN (RoadRunner, BCL_DQN_Atari/BCL_AT_v1 folder)
python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/RoadRunnerNoFrameskip-v4_robust.pt" --attack-epsilon-start 3.0

# BCL-RADIAL+AT-DQN (AT part for BankHeist/RoadRunner, BCL_DQN_Atari/BCL_AT folder)
python main.py --env BankHeistNoFrameskip-v4 --robust --load-path "trained_models/BCL_RADIAL_DQN_BankHeist.pt" --attack-epsilon-start 13.0
python main.py --env RoadRunnerNoFrameskip-v4 --robust --load-path "trained_models/BCL_RADIAL_DQN_RoadRunner.pt" --lr 0.000000125 --adam-eps 0.00015 --attack-epsilon-start 12.0
```

Notice that for RoadRunner environment with AT-DQN/NCL-AT-DQN and BCL-C/MOS-AT-DQN, our implementation is based on [RADIAL-DQN (v1)](https://github.com/tuomaso/radial_rl/tree/master/DQN), as they yield better results.

## Evaluation
As mentioned in the paper, we conduct evaluation with four different adversarial attack methods: 
1. 30-step PGD: 30-step untargeted PGD attack with step size 0.1;
2. RI-FGSM (alpha = 0.375): FGSM with random initialization;
3. RI-FGSM (Multi): sample N = 1000 random starts for RI-FGSM, and takes the first sample where the resulting adversarial example alters the action;
4. RI-FGSM (Multi-T): sample N = 1000 random starts for RI-FGSM, and takes the sample which results the agent taking the action corresponding to the lowest Q value among those N samples.

We report the lowest reward obtained after running those four attacks. We observe that with obfuscated gradients, RI-FGSM (Multi-T) results in the strongest attack in many cases, while 30-step PGD is typically stronger otherwise.

Below is an evaluation example:
```
python evaluate.py --env PongNoFrameskip-v4 --load-path "trained_models/BCL_C_AT_DQN_Pong.pt" --pgd --RI_FGSM --RI_FGSM_Multi --RI_FGSM_Multi_T --eps 10 20 25 --nominal
```
In this example, the environment is Pong, attacker's adversarial perturbation sizes are {10, 20, 25}/255.0. `--pgd` is to run 30-step PGD (step size 0.1), `--RI_FGSM` is to run RI-FGSM, `--RI_FGSM_Multi` is to run RI-FGSM (Multi) and `--RI_FGSM_Multi_T` is to run RI-FGSM (Multi-T). 

Note that we fixed the seeds of the environment during evaluation, meaning the results are the same across different runs. However, we noticed that when the code is run on different machines, sometimes the results might be slightly different (they still remain on the same magnitude level). 

# Curriculum Learning for PPO models
For PPO-style models we experiment on two Procgen games: FruitBot and Jumper. Our implementation is based on [RADIAL-PPO](https://github.com/tuomaso/radial_rl_v2/tree/main/Procgen). The code is in the `PPO_Procgen/` folder and runs on GPU by default.

## Pre-trained Models
Pre-trained models are available under folders `BCL_PPO_Procgen/trained_models` and `BCL_PPO_Procgen/log`. It contains 
1. PPO (Vanilla) -- named as "XXX_ppo.pt". 
2. RADIAL-PPO -- named as "XXX_radial_ppo.pt". 
3. Median results of our BCL models (i.e., the ones reported in Table 7 in the Appendix) -- named as `model_final.pt`, the name of the folder corresponds to the model name in Table 7. For example: [this](https://github.com/jlwu002/BCL/tree/main/BCL_PPO_Procgen/log/fruitbot/nlev_200_easy/BCL_MOS_V_AT_PPO_FruitBot) is the pre-trained model for BCL-MOS(V)-AT-PPO (FruitBot).

## Training
To perform curriculum learning, update `--model-file` with the model we want to bootstrap from. The starting epsilon of the curriculum can be set through `--eps-start`, and the increment of the curriculum can be set through `--epsilon-end`. Throughout the training, the adversarial perturbation size will increase from `eps-start`/255 to `eps-start`/255+`epsilon-end`.

```sh
# FruitBot
python train_procgen.py --exp-name new_exp_fruitbot --env-name=fruitbot --eps-start 0.0 --epsilon-end 3.92e-3 --model-file=trained_models/FruitBot_ppo.pt

# Jumper
python train_procgen.py --exp-name new_exp_jumper --env-name=jumper --eps-start 1.0 --epsilon-end 3.92e-3 --model-file=trained_models/Jumper_ppo.pt

```

For FruitBot environment, adversarial examples are generated using RI-FGSM; for Jumper environment, adversarial examples are generated using 10-step PGD.

## Evaluation
We evaluate the models using 30-step PGD with step size 0.1. Below is an evaluation example:
```sh
python eval_procgen.py --env-name=jumper --model-file=log/jumper/nlev_200_easy/BCL_MOS_V_AT_PPO_Jumper/model_final.pt --standard --pgd --deterministic
```
The evaluation results will be printed as well as stored under the `log/jumper/nlev_200_easy/BCL_MOS_V_AT_PPO_Jumper/eval` folder.
