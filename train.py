import os
import warnings
from omegaconf import OmegaConf
from src.config import config
from src.env import FlappyBirdEnv
from stable_baselines3 import PPO

def train(args):
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok = True)

    logdir = args.logdir
    os.makedirs(logdir, exist_ok = True)

    env = FlappyBirdEnv(
        img_size = args.img_size,
        penalty = args.penalty
    )
    model = PPO(
        policy = "CnnPolicy", 
        env = env, 
        verbose = 1,
        tensorboard_log = logdir,
    )

    iter = args.timesteps // args.save_step
    for i in range(iter):
        model.learn(
            total_timesteps = args.save_step,
            reset_num_timesteps = False,
            tb_log_name = args.log_name
        )

        name = f"{(i+1)*args.save_step:07d}.zip"
        model.save(os.path.join(models_dir, name))
        
    env.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = OmegaConf.structured(config)
    train(args)