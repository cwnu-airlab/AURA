import json
import logging
import os
import os.path

import hydra
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import BeirDataset
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing, SparseRetrieval
from .utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def retrieve(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    # allow adding keys when running in strict mode
    OmegaConf.set_struct(config, False)

    #if HF: need to udate config.
    if "hf_training" in config:
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None
       
    model = get_model(config, init_dict)

    batch_size_d = config["index_retrieve_batch_size"]
    batch_size_q = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)

    # Just some code to print debug information to stdout
    logging.basicConfig(format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # Download and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        exp_dict["beir"]["dataset"])
    out_dir = exp_dict["beir"]["dataset_path"]
    data_path = util.download_and_unzip(url, out_dir)

    config["index_dir"] = os.path.join(config["index_dir"], "beir", exp_dict["beir"]["dataset"])
    os.makedirs(config["index_dir"], exist_ok=True)

    config["out_dir"] = os.path.join(config["out_dir"], "beir", exp_dict["beir"]["dataset"])
    os.makedirs(config["out_dir"], exist_ok=True)

    # Optional hybrid sparse/dense fusion (CLS-based dense comes from model outputs)
    hybrid_alpha = config.get("hybrid_alpha", 1.0)
    if "config" in exp_dict and "hybrid_alpha" in exp_dict["config"]:
        hybrid_alpha = exp_dict["config"]["hybrid_alpha"]
    hybrid_alpha_list = config.get("hybrid_alpha_list", None)
    if hybrid_alpha_list is None:
        hybrid_alpha_list = [hybrid_alpha]
    config["hybrid_alpha"] = hybrid_alpha
    config["hybrid_alpha_list"] = hybrid_alpha_list
    # optional: toggle minmax fusion
    if "enable_minmax" not in config:
        config["enable_minmax"] = True
    # optional: cast dense index to fp16 to save VRAM/CPU RAM (may change scores slightly)
    if "dense_fp16" not in config:
        config["dense_fp16"] = False

    # Provide the data path where dataset has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) datapath/corpus.jsonl  (format: jsonlines)
    # (2) datapath/queries.jsonl (format: jsonlines)
    # (3) datapath/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    d_collection = BeirDataset(corpus, information_type="document")
    q_collection = BeirDataset(queries, information_type="query")

    # Index BEIR collection
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"], batch_size=batch_size_d,
                                    shuffle=False, num_workers=4)
    evaluator = SparseIndexing(model=model, config=config, compute_stats=True)
    evaluator.index(d_loader, id_dict=d_collection.idx_to_key)

    # Retrieve from BEIR collection
    q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"], batch_size=batch_size_q,
                                    shuffle=False, num_workers=1)
    evaluator = SparseRetrieval(config=config, model=model, compute_stats=True, dim_voc=model.output_dim, is_beir=True)
    evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"] + 1, id_dict=q_collection.idx_to_key,
                       alpha_list=hybrid_alpha_list)

    perf = {}

    def eval_run(run_path, tag):
        if not os.path.exists(run_path):
            return None
        with open(run_path) as reader:
            run = json.load(reader)
        new_run = dict()
        print(f"Removing query id from document list ({tag})")
        for query_id, doc_dict in tqdm(run.items()):
            query_dict = dict()
            for doc_id, doc_values in doc_dict.items():
                if query_id != doc_id:
                    query_dict[doc_id] = doc_values
            new_run[query_id] = query_dict
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, new_run, [1, 10, 100, 1000])
        results2 = EvaluateRetrieval.evaluate_custom(qrels, new_run, [1, 10, 100, 1000], metric="r_cap")
        res = {
            "NDCG@10": ndcg["NDCG@10"],
            "Recall@100": recall["Recall@100"],
            "Recall@1000": recall["Recall@1000"],
            "R_cap@100": results2["R_cap@100"]
        }
        perf[tag] = res
        print(f"{tag} -> {res}")

    for alpha in hybrid_alpha_list:
        # raw fusion
        fname = "run_alpha_{}.json".format(alpha)
        run_path = os.path.join(config.out_dir, fname)
        if not os.path.exists(run_path) and alpha == hybrid_alpha_list[0]:
            run_path = os.path.join(config.out_dir, "run.json")
        eval_run(run_path, f"alpha_{alpha}")

        # minmax fusion
        if config.get("enable_minmax", True):
            fname_mm = "run_minmax_alpha_{}.json".format(alpha)
            run_path_mm = os.path.join(config.out_dir, fname_mm)
            eval_run(run_path_mm, f"alpha_{alpha}_minmax")

    json.dump(perf, open(os.path.join(config.out_dir, "perf.json"), "w"))


if __name__ == "__main__":
    retrieve()
