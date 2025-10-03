from pathlib import Path

from src.utils.utils import CacheManager
from src.provenance.evaluate_provenance import rounds_provenance
from src.utils.utils import save_json
from src.utils.plotting import plot_provenance_accuracy



if __name__ == "__main__":

    results_dir = Path("results")

    print(
        f"All completed experiment keys : {CacheManager.get_completed_experiments_keys()}")

    for exp_key in CacheManager.get_completed_experiments_keys():
        print(f"Running provenance analysis for experiment key: {exp_key}")
        json_path = results_dir / f"{exp_key}_provenance.json"

        if json_path.exists():
            print(f"Provenance info already exists: {json_path}")
            continue
        prov_dict = rounds_provenance(exp_key=exp_key)
        save_json(prov_dict, json_path)            
        plot_provenance_accuracy(json_path, results_dir=results_dir)
