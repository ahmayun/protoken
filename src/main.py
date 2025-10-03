import unsloth
import os
import multiprocessing
import time
import logging

from src.config.base_config import ConfigManager
from src.utils.utils import  CacheManager, save_json



flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.propagate = False

_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4



def main():
    start_time = time.time()
    cfg, experiment_key = ConfigManager.load_config_with_corresponding_key()
    print(f"=============== Training with Experiment Key: {experiment_key} ================")
    
    from src.fl.simulation import run_fl_experiment
    global_metrics_history = run_fl_experiment(cfg)
    print(f"Total Time Taken: {time.time() - start_time} seconds")
    
     
    CacheManager.consolidate_experiment(exp_key=experiment_key, experiment_config=cfg)
    save_json(global_metrics_history, f"results/fl_train_metrics_{experiment_key}.json")

if __name__ == "__main__":
    main()
