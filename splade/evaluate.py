import json
import os

import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.evaluation.eval import load_and_evaluate
from splade.utils.utils import get_dataset_name
from splade.utils.hydra import hydra_chdir

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME,version_base="1.2")
def evaluate(exp_dict: DictConfig):

    # for dataset EVAL_QREL_PATH
    # for metric of this qrel
    hydra_chdir(exp_dict)
    eval_qrel_path = exp_dict.data.EVAL_QREL_PATH
    eval_metric = exp_dict.config.eval_metric
    dataset_names = exp_dict.config.retrieval_name
    out_dir = exp_dict.config.out_dir

    # hybrid settings
    hybrid_alpha = exp_dict.config.get("hybrid_alpha", 1.0)
    if "config" in exp_dict and "hybrid_alpha" in exp_dict["config"]:
        hybrid_alpha = exp_dict["config"]["hybrid_alpha"]
    hybrid_alpha_list = exp_dict.config.get("hybrid_alpha_list", None)
    if hybrid_alpha_list is None:
        hybrid_alpha_list = [hybrid_alpha]
    enable_minmax = exp_dict.config.get("enable_minmax", True)

    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics, dataset_name) in enumerate(zip(eval_qrel_path, eval_metric, dataset_names)):
        if qrel_file_path is not None:
            res = {}
            print(eval_metrics)

            def eval_run(run_path, tag):
                if not os.path.exists(run_path):
                    return
                for metric in eval_metrics:
                    out_metric = load_and_evaluate(qrel_file_path=qrel_file_path,
                                                   run_file_path=run_path,
                                                   metric=metric)
                    if tag not in res:
                        res[tag] = {}
                    res[tag].update(out_metric)

            # raw runs per alpha
            for alpha in hybrid_alpha_list:
                run_path = os.path.join(out_dir, dataset_name, f'run_alpha_{alpha}.json')
                # fallback to run.json for first alpha
                if not os.path.exists(run_path) and alpha == hybrid_alpha_list[0]:
                    run_path = os.path.join(out_dir, dataset_name, 'run.json')
                eval_run(run_path, f"alpha_{alpha}")

                if enable_minmax:
                    run_path_mm = os.path.join(out_dir, dataset_name, f'run_minmax_alpha_{alpha}.json')
                    eval_run(run_path_mm, f"alpha_{alpha}_minmax")

            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res
            out_fp = os.path.join(out_dir, dataset_name, "perf.json")
            json.dump(res, open(out_fp,"a"))
    out_all_fp= os.path.join(out_dir, "perf_all_datasets.json")
    json.dump(res_all_datasets, open(out_all_fp, "a"))

    return res_all_datasets

if __name__ == '__main__':
    evaluate()
