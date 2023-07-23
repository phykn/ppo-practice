import os
import time
import warnings
from glob import glob
from omegaconf import OmegaConf
from src.config import config
from src.env import FlappyBirdEnv
from stable_baselines3 import PPO

def test(args):  
    env = FlappyBirdEnv(
        img_size = args.img_size,
        penalty = args.penalty
    )

    model = PPO(
        policy = "CnnPolicy", 
        env = env
    )

    files = glob(os.path.join(args.models_dir, "*.zip"))
    file = files[-1]
    model.load(path = file, device = args.test_device)
    print(f"Load: {file}, Device: {args.test_device}")
    
    episodes = 5
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            env.render()
            time.sleep(1 / 30)

            action, state = model.predict(obs, deterministic = False)
            obs, reward, terminated, truncated, info = env.step(action)

    env.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = OmegaConf.structured(config)
    test(args)