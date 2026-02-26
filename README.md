# Aura
<!-- [![paper](https://img.shields.io/badge/arxiv-arXiv%3A2107.05720-brightgreen)](https://arxiv.org/abs/2107.05720)
[![huggingface weights](https://img.shields.io/badge/huggingface-splade-9cf)](https://huggingface.co/naver) 
[![github](https://img.shields.io/badge/github-splade-9cf)](https://github.com/naver/splade) -->


<!-- 구조 이미지 -->

<br>

This repository contains the code to perform **training**, **indexing** and **retrieval** for Aura models. It also includes everything needed to launch evaluation on the [BEIR](https://github.com/beir-cellar/beir) benchmark.

<br>

*** 

## Method
Aura is a neural retrieval model that builds on SPLADE-style sparse expansion while additionally incorporating a dense sementic signal. Specially, it learns query/document sparse representations through the BERT MLM head under sparsity regularization, and in parallel computes a dense representation to measure semantic similarity. To control sparsity more selectively, Aura introduces an IDF-weighted FLOPS regularizer that suppresses uninformative activations while emphasizing discriminative terms. Moreover, a training scheduler dynamically balances the dense retrieval objective and the IDF-FLOPS sparsity objective throughout optimization, stabilizing early training and encouraging an effective sparse–dense trade-off.

<br>

**Usage**
This repository lets you either train (`train.py`), index (`index.py`), retrieve (`retrieve.py`) (or perform every step with `all.py`) Aura models.

***

<br>

## Directory Structure

```
├── conf
│   ├── config_aura.yaml
│   ├── CONFIG_CHOICE.py
│   ├── config_default.yaml
│   ├── config_splade.yaml
│   ├── flops
│   ├── index
│   ├── README.md
│   ├── retrieve_evaluate
│   └── train
├── data
│   └── msmarco
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── splade
    ├── all.py
    ├── beir_eval.py
    ├── create_anserini.py
    ├── datasets
    ├── evaluate.py
    ├── evaluation
    ├── flops.py
    ├── hf
    ├── hf_train.py
    ├── hf_train_reranker.py
    ├── indexing
    ├── index.py
    ├── index_retrieve.py
    ├── __init__.py
    ├── losses
    ├── models
    ├── optim
    ├── rerank.py
    ├── retrieve.py
    ├── tasks
    ├── train.py
    └── utils
```
***

<br>
<br>


## Getting started

### Requirements

* We recommend starting from a fresh environment and installing the required packages via pip using the `requirements.txt` file.

```
virtualenv venv --python=python3.9
source venv/bin/activate
pip install -r requirements.txt
```

<br>

### Data

* We train our models using the [MS MARCO](https://github.com/microsoft/MSMARCO-Passage-Ranking) data.

* For easier setup, you can use the data package provided by SPLADE:
  * SPLADE data(queries, documents, hard negatives): [download](https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz)

* Additionally, to perform contrastive learning with BM25-hard negatives, you can use the following data:
  * BM25-hard negatives: [download](https://download.europe.naverlabs.com/splade/sigir22/triplets.tar.gz)


<br>

1.  After downloading the SPLADE data package, extract it in the project root directory.
```
tar -xzvf data.tar.gz
```

2. Then extract the BM25-hard negatives triplets in the `data/msmarco/triplets` directory.
```
tar -xzvf triplets.tar.gz
```

3. To run our code, an IDF `.npy` file is required. The path to this `.npy` file is not configured via the config files; instead, it is specified directly in `splade/loss/regularizer.py`. For convenience, the `.npy` file used in our method(IDF values computed from MS MARCO) is provided in `data/msmarco/msmarco_idf.npy`.

<br>

### Quick start (TRAIN)

* Experiments are designed to run with **4 GPUs**.
* You can adjust the alpha value by changing the `config.hybrid_alpha_list` parameter (`config.hybrid_alpha_list="[<your alpha value>]"`). You can also provide multiple values in the list to evaluate results for each alpha value.
* To run the full pipeline (shown here with MSMARCO data using `config_aura.yaml`), move to the project root directory and execute:
<!-- In order to perform all steps (here on toy data, i.e. `config_default.yaml`), go on the root directory and run: -->

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
SPLADE_CONFIG_NAME=config_aura 
python -m splade.all config.checkpoint_dir=experiments/test/checkpoint   config.index_dir=experiments/test/index   config.out_dir=experiments/test/out +config.hybrid_alpha_list="[0.75]"  config.index_retrieve_batch_size=256  +config.dense_to_gpu=true  +config.dense_fp16=true +config.empty_cuda_cache_after_index=true +config.empty_cuda_cache_after_retrieve=true +config.dense_chunk_size=50000 +config.enable_minmax=true
```

<br>

### Evaluating model

* After training, indexing and retrieval will run automatically. If it does not run for any reason(e.g., insufficient GPU memory), please follow the steps below:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
SPLADE_CONFIG_FULLPATH="$(pwd)/experiments/test/checkpoint/config.yaml"
python -m splade.index_retrieve  +config.hybrid_alpha_list="[0.75]"  config.index_retrieve_batch_size=256  config.dense_to_gpu=true  config.dense_fp16=true config.empty_cuda_cache_after_index=true config.empty_cuda_cache_after_retrieve=true config.dense_chunk_size=50000 config.enable_minmax=true
``` 

<br>

* If you encounter a config-related error, please follow the steps below:
1. Check if the config file exists.

2. If config file exists, use the following command to check which config is currently being exported:
```bash
echo "SPLADE_CONFIG_NAME=$SPLADE_CONFIG_NAME"
echo "SPLADE_CONFIG_FULLPATH=$SPLADE_CONFIG_FULLPATH"
```

3. If the exported config is not the one you want, please run the commands below to **unset** the current export and then export the correct config:
```bash
unset SPLADE_CONFIG_NAME
unset SPLADE_CONFIG_FULLPATH

export SPLADE_CONFIG_FULLPATH="$(pwd)/experiments/test/checkpoint/config.yaml"
```

<br>

### BEIR eval

You can also evaluate on the BEIR datasets using the following command (as in the in-domain evaluation, you can control the alpha value via `config.hybrid_alpha_list`)

```bash
export SPLADE_CONFIG_FULLPATH="/path/to/checkpoint/dir/config.yaml"
for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid climate-fever dbpedia-entity fever hotpotqa nq; do   python -m splade.beir_eval     +beir.dataset=$dataset     +beir.dataset_path=datasets/beir     ++config.hybrid_alpha_list="[0.75]"     config.index_retrieve_batch_size=256  ++config.empty_cuda_cache_after_index=true   ++config.dense_fp16=true ++config.dense_to_gpu=true ++config.empty_cuda_cache_after_retrieve=true ++config.dense_chunk_size=50000 ++config.enable_minmax=true; done
```

***

***

<br>

The following citation and reference below are based on information provided in the official SPLADE Github repository.

<br>

# Cite

Please cite our work as:

* (v1) SIGIR21 short paper

```
@inbook{10.1145/3404835.3463098,
author = {Formal, Thibault and Piwowarski, Benjamin and Clinchant, St\'{e}phane},
title = {SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3463098},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2288–2292},
numpages = {5}
}
```

* (v2) arxiv

```
@misc{https://doi.org/10.48550/arxiv.2109.10086,
  doi = {10.48550/ARXIV.2109.10086},
  url = {https://arxiv.org/abs/2109.10086},
  author = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, Stéphane},
  keywords = {Information Retrieval (cs.IR), Artificial Intelligence (cs.AI), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

* (v2bis) SPLADE++, SIGIR22 short paper

```
@inproceedings{10.1145/3477495.3531857,
author = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, St\'{e}phane},
title = {From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531857},
doi = {10.1145/3477495.3531857},
abstract = {Neural retrievers based on dense representations combined with Approximate Nearest Neighbors search have recently received a lot of attention, owing their success to distillation and/or better sampling of examples for training -- while still relying on the same backbone architecture. In the meantime, sparse representation learning fueled by traditional inverted indexing techniques has seen a growing interest, inheriting from desirable IR priors such as explicit lexical matching. While some architectural variants have been proposed, a lesser effort has been put in the training of such models. In this work, we build on SPLADE -- a sparse expansion-based retriever -- and show to which extent it is able to benefit from the same training improvements as dense models, by studying the effect of distillation, hard-negative mining as well as the Pre-trained Language Model initialization. We furthermore study the link between effectiveness and efficiency, on in-domain and zero-shot settings, leading to state-of-the-art results in both scenarios for sufficiently expressive models.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2353–2359},
numpages = {7},
keywords = {neural networks, indexing, sparse representations, regularization},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```

* efficient SPLADE, SIGIR22 short paper

```
@inproceedings{10.1145/3477495.3531833,
author = {Lassance, Carlos and Clinchant, St\'{e}phane},
title = {An Efficiency Study for SPLADE Models},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531833},
doi = {10.1145/3477495.3531833},
abstract = {Latency and efficiency issues are often overlooked when evaluating IR models based on Pretrained Language Models (PLMs) in reason of multiple hardware and software testing scenarios. Nevertheless, efficiency is an important part of such systems and should not be overlooked. In this paper, we focus on improving the efficiency of the SPLADE model since it has achieved state-of-the-art zero-shot performance and competitive results on TREC collections. SPLADE efficiency can be controlled via a regularization factor, but solely controlling this regularization has been shown to not be efficient enough. In order to reduce the latency gap between SPLADE and traditional retrieval systems, we propose several techniques including L1 regularization for queries, a separation of document/query encoders, a FLOPS-regularized middle-training, and the use of faster query encoders. Our benchmark demonstrates that we can drastically improve the efficiency of these models while increasing the performance metrics on in-domain data. To our knowledge, we propose the first neural models that, under the same computing constraints, achieve similar latency (less than 4ms difference) as traditional BM25, while having similar performance (less than 10% MRR@10 reduction) as the state-of-the-art single-stage neural rankers on in-domain data.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2220–2226},
numpages = {7},
keywords = {splade, sparse representations, latency, information retrieval},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```

***


# License

SPLADE Copyright (c) 2021-present NAVER Corp.

SPLADE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
(see [license](license.txt))

You should have received a copy of the license along with this work. If not,
see http://creativecommons.org/licenses/by-nc-sa/4.0/ .
