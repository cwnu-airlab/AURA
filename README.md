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

This branch contains an implementation that applies the above method the MSE Distillation version.

<br>

**Usage**
This branch enables users to train a Aura model(MSE Distillation version) and perform document retrieval when applying the retriever to RAG pipeline.

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

```bash
virtualenv venv --python=python3.9
source venv/bin/activate
pip install -r requirements.txt
```

<br>

### Data

* In the SPLADE MSE Distillation version, training is performed using the [MS MARCO](https://github.com/microsoft/MSMARCO-Passage-Ranking) data.

* If a user wants to train the model with their own dataset, please follow the steps below.

<br>

1.  Create a data directory in the `data` directory to store the `raw.tsv` file used for training.
```bash
mkdir data/<directory name>
```

2. Update the `TRAIN_DATA_DIR` path in the `config_aura_msedistil.yaml` file.


3. Prepare a `.tsv` training file in the following format:
```
[query] \t [positive document] \t [negative document] \t [positive document score] \t [negative document score]
```

<br>

### Quick start (TRAIN)

* You can use the code in `aura_msedistil.py` to intergrate the retriever into a RAG pipeline.

```bash
python aura_msedistil.py --mode train
```

<br>

### Evaluating model

* After training, you can use the code in `aura_msedistil.py` to retrieve the top-k documents for each query from your target dataset.

* To perform document retrieval and evaluation using your trained model, you need to prepare the following 3 datasets in specific formats:

1. Target Document Collection (Required for the `Index` step)
- This is the entire set of documents to be searched.
- It must be placed as a `raw.tsv` file under the directory specified by `COLLECTION_PATH`.
- The format should be a 2-column structure: `[Document ID] \t [Document Text]`.

2. Evaluation Query Collection (Required for the `Retrieve` step)
- This is the set of queries the model will actually search for.
- It must be placed as a `raw.tsv` file under the directory specified by `Q_COLLECTION_PATH`.
- The format should be the same as the document collection, a 2-column structure: `[Query ID] \t [Query Text]`.

3. Ground Truth File (Required for the `Evaluate` step)
- This is the QREL (Query Relevance) file used to score the retrieval performance.
- It must be a `.json` file located at the path specified by `EVAL_QREL_PATH`.
- The JSON structure must be in the format `{"Query ID": {"Target Document ID": 1}}` to properly calculate performance metrics such as MRR and NDCG after retrieval.

* You can load the trained model, run evaluation(indexing & retrieval), and check the results using the following command.

```bash
python aura_msedistil.py --mode eval
```

* After the evaluation is complete, if you only want to inspect the retrieval results, you can use the following command. The terminal will display the top documents for each query according to the number specified by the user as an argument.

```bash
python aura_msedistil.py --mode results --top_k 10
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
