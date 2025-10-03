import os
import yaml

def setup_environment(config):
    env = config["environment"]
    os.environ["CUDA_DEVICE_ORDER"] = env["CUDA_DEVICE_ORDER"]
    os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    os.environ["HTTP_PROXY"] = env["HTTP_PROXY"]
    os.environ["HTTPS_PROXY"] = env["HTTPS_PROXY"]
    os.environ["HF_HOME"] = env["HF_HOME"]

    java_home = os.path.expanduser(env["JAVA_HOME"])
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:" + os.environ["PATH"]
