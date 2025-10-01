import unsloth
import os
import multiprocessing
import time
import logging

from src.config.base_config import ConfigManager
from src.utils.datasets import initialize_dataset_chunks
from src.utils.utils import  CacheManager, get_model_and_tokenizer
from src.utils.plotting import save_and_plot_metrics



flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.propagate = False

_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 4

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4



def main():
    start_time = time.time()
    cfg = ConfigManager.load_config()
    print(f"Configuration Loaded: {cfg}")
    initialize_dataset_chunks(get_model_and_tokenizer(cfg)[1])
    from src.fl.simulation import run_fl_experiment
    global_metrics_history = run_fl_experiment(cfg)
    print(f"Total Time Taken: {time.time() - start_time} seconds")
    CacheManager.consolidate_experiment(
        exp_key="Test-Refactor", experiment_config=cfg)
    save_and_plot_metrics(global_metrics_history, "results2_refactor")

if __name__ == "__main__":
    main()
