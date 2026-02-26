config files are defined using [facebook hydra](https://hydra.cc).

Hydra defines a modular config, which can be altered using the command line.


To check the setting used in our method, please refer to [conf/config_aura.yaml](./config_aura.yaml).

The description below is quoted from the SPLADE Github README.

***
***


### Everything at once

```SPLADE_CONFIG_NAME=config_aura python -m src.all``` will train, index, retrieve and evaluate.

### More examples (train)


#### _Train another model_

__Option 1__: create/use another config file such as: ```conf/config_aura.yaml```:

```
# FILES
defaults: # (these specify which config FILES to use)
  ############## TRAIN ###################################
  - train/config: splade
  - train/data: msmarco
  - train/model: splade
  ############## INDEX ###################################
  - index: msmarco
  ############## RETRIEVE ################################
  - retrieve_evaluate: all
  ############### FLOPS ##################################
  - flops: msmarco

# Direct PARAMETER setting
config:
  loss: InBatchPairwiseNLLWithCls
  augment_pair: in_batch_negatives
  info_step_based_scheduler:
    lambda_cls: 1.0
    T_mid: 50000
  regularizer:
    FLOPS:
      idf_path: 'data/msmarco/idf/msmarco_idf.npy'
      e: 1e-06
      lambda_q: 0.0003
      lambda_d: 0.0001
      T: 50000
      targeted_rep: rep
      reg: FLOPS
  checkpoint_dir: models/aura/checkpoint
  index_dir: models/aura/index
  out_dir: models/aura/out

```

To tell the code to use this config file, either modify ```CONFIG_NAME``` to ```config_aura```
in [conf/CONFIG_CHOICE.py](./CONFIG_CHOICE.py) to use it instead of ```config_default.yaml```

**or**

equivalently, define and use an environment variable: ```SPLADE_CONFIG_NAME=config_aura python -m src.train```

It is also possible to re-use a config saved from a previous experiment; in this case provide its full
path: ```SPLADE_CONFIG_FULLPATH=/path/to/my/exp/config_aura.yaml python -m src.train```

__Option 2__: provide command line arguments for everything that changes from default or desired config, specifying
files or parameters to change:

```python -m src.train config.checkpoint_dir=<MY_RES_DIR> train/data=msmarco train/model=splade init_dict.model_type_or_dir='bert-base-uncased' ...```

### Note on naming in hydra

TLDR: for overriding a parameter, use the parameter _package naming_, for overriding a package, use the _package path_

#### Overriding a parameter

If you are overriding a `parameter` from the command line you should name is using its `package position`. For example,
overriding `model_type_or_dir`: this parameter is defined in [train/config/splade.yaml](train/config/splade.yaml), and
the content of this file is:

```
# @package _global_

init_dict:
  model_type_or_dir: distilbert-base-uncased
  model_type_or_dir_q: null
  freeze_d_model: 0
  agg: max
  fp16: true

config:
  tokenizer_type: distilbert-base-uncased
```

Therefore, the parameter is referred to with `init_dict.model_type_or_dir=...`

```
python -m src.train init_dict.model_type_or_dir=<CHOICE_FOR_MODEL>
```

if the header in [train/config/splade.yaml](train/config/splade.yaml) was `# @package awesome_package` the command would
thus be:

```
python -m src.train awesome_package.init_dict.model_type_or_dir=<CHOICE_FOR_MODEL>
```

#### Overriding a package

Override a package by referring to its path:

```
python -m src.train train/data=msmarco
```

### Index, retrieve

same thing with ```src.index```, ```src.retrieve```.
