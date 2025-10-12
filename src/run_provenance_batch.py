from pathlib import Path
from src.utils.utils import CacheManager
from src.provenance.evaluate_provenance import rounds_provenance
from src.utils.utils import save_json
from src.utils.plotting import plot_provenance_accuracy



if __name__ == "__main__":

    results_dir = Path("results")

    # print(
    #     f"All completed experiment keys : {CacheManager.get_completed_experiments_keys()}")
    print(f"{10*'-'} Completed Training Keys {10*'-'}")
    for exp_key in CacheManager.get_completed_experiments_keys():
        print(f"- {exp_key}") 
    

    print(f"{10*'-'} Running Provenance Analysis {10*'-'}")
    # for exp_key in CacheManager.get_completed_experiments_keys():
    for exp_key in ['[google_gemma-3-270m-it][rounds5][clients2][C0-medical-C1finance][LoRA-r8-alpha8][New2]']:
        print(f"Running provenance analysis for experiment key: {exp_key}")
        json_path = results_dir / f"{exp_key}_provenance.json"

        # if json_path.exists():
        #     print(f"Provenance info already exists: {json_path}")
        #     continue

        # if exp_key.lower().find('lora') != -1:
        #     print("Skipping LoRA-based experiment for now.")
        #     continue

        prov_dict = rounds_provenance(exp_key=exp_key)
        save_json(prov_dict, json_path)            
        plot_provenance_accuracy(json_path, results_dir=results_dir)
