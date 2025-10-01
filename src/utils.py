import os
from typing import Dict
from omegaconf import DictConfig
import torch
from dotenv import load_dotenv


def setup_env(cfg: DictConfig):
    """
    Setup the environment for passage retrieval using the given config
    """
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.environment.cuda_device)
    # os.environ["HTTP_PROXY"] = cfg.environment.http_proxy
    # os.environ["HTTPS_PROXY"] = cfg.environment.https_proxy
    os.environ["HF_HOME"] = cfg.environment.hf_home
    print(os.environ["HF_HOME"])
    os.environ["JAVA_HOME"] = os.path.expanduser(cfg.environment.java_home)
    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ["PATH"]
    # os.environ["WANDB_DISABLED"] = "true"

    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")

    # login(HF_TOKEN)


def is_ampere_gpu() -> bool:
    """
    Returns True if the current CUDA device is from the Ampere architecture (compute capability 8.x).
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 8
