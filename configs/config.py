import argparse
import logging
import os

protocol_dict = {
    "iPhone11_iPhone11":1,
    "iPhone12_iPhone12":2,
    "iPhone11_iPhone12":3,
    "iPhone12_iPhone11":4
}

def get_logger(filename: str, protocol: int = -1) -> logging.Logger:
    logger = logging.getLogger(f"Eval_{protocol}_{filename}")
    logger.setLevel(logging.DEBUG)
    # if not logger.hasHandlers():

    if protocol == -1:
        log_dir = f"logs/{filename}"
    else:
        log_dir = f"logs/Protocol_{protocol}"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/{filename}.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tr",
        "--trainds",
        default="iPhone11",
        type=str,
        help="specify training dataset",
    )

    parser.add_argument(
        "-te",
        "--testds",
        default="iPhone12",
        type=str,
        help="specify testing dataset",
    )
    
    parser.add_argument(
        "-rdir",
        "--root_dir",
        default="/mnt/extravolume/data/",
        type=str,
        help="specify root directory address",
    )
    
    parser.add_argument(
        "-model",
        "--modeltype",
        default="proposed",
        type=str,
        help="specify models to train or test: sota or proposed",
    )
    
    parser.add_argument(
        "-run",
        "--runtype",
        default="test",
        type=str,
        help="specify runtype for proposed model: train or test (by defualt, sota will be both trained and tested)",
    )

    parser.add_argument(
        "-pc",
        "--pc_model",
        default="pointnet",
        type=str,
        help="specify sota pointcloud model",
    )
    
    return parser