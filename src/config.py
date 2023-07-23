from dataclasses import dataclass

@dataclass
class config:
    models_dir: str = "models/PPO"
    logdir: str = "logs"
    log_name: str = "PPO"
    img_size: int = 128
    penalty: int = 1

    timesteps: int = 1000000
    save_step: int = 10000

    test_device: str = "cpu"