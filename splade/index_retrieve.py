import hydra
from omegaconf import DictConfig
import time
import json
import os

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .index import index
from .retrieve import retrieve_evaluate
from .utils.hydra import hydra_chdir


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def index_retrieve(exp_dict: DictConfig):
    """Run indexing then retrieval/evaluation with a single command."""
    hydra_chdir(exp_dict)
    start_time = time.time()
    index(exp_dict)
    index_time = time.time() - start_time
    print(f"Index execution time: {index_time:.2f} seconds")

    start_time = time.time()
    retrieve_evaluate(exp_dict)
    retrieve_evaluate_time = time.time() - start_time
    print(f"Retrieve/Evaluate execution time: {retrieve_evaluate_time:.2f} seconds")

    out_dir = exp_dict.config.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "execution_time.json"), "w") as f:
        json.dump({
            "index_time": index_time,
            "retrieve_evaluate_time": retrieve_evaluate_time
        }, f, indent=4)


if __name__ == "__main__":
    index_retrieve()
