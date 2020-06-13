# Feasible Counterfactual Explanations
Code accompanying the paper [Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers](https://arxiv.org/abs/1912.03277), selected for Oral Spotlight at the [NeurIPS 2019 Workshop](http://tripods.cis.cornell.edu/neurips19_causalml/) on Machine learning and Causal Inference for improved decision making

# DiCE

This work is also being integerated with [DiCE](https://www.microsoft.com/en-us/research/project/dice/), an open source library for explaining ML models. Please check the [feasible-cf branch](https://github.com/microsoft/DiCE/tree/feasible-cf) and this [tutorial](https://github.com/microsoft/DiCE/blob/feasible-cf/notebooks/DiCE_getting_started_feasible.ipynb) on DiCE for updates regarding the same. 

# Cite
```bibtex
@article{mahajan2019preserving,
  title={Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers},
  author={Mahajan, Divyat and Tan, Chenhao and Sharma, Amit},
  journal={arXiv preprint arXiv:1912.03277},
  year={2019}
}
```

# Code Structure 

### generativecf 

Contains the code for experiments on Simple-BN, Sangiovese, Adult dataset

### generativecf-mnist

Containts the code for experiments on MNIST 

## generativecf

* models/

  - Contains pre trained models for the different methods across datasets

* data/

  - Contains the processed data files for all the datasets; download the data files from this [link](https://drive.google.com/drive/folders/1wI_PdEC9bTc80Lj0Th8LZ63dQGmtaDij?usp=sharing)

- master_evalute.py 

  - Utilizes the pre trained models (models/) and datasets (data/) to reproduce the results mentioned in the paper. The results are stored in the directory /results

  - It also generates a file 'plot_dict.json' in the directory r_plots/; where you may convert it to plotdf.csv file and then execute 'plot_figures.R' script to get better graphs stored in the directory /results

- base-generative-cf.py

  - Implementation of BaseGenCF for all datasets

  - Usage: python3 base-generative-cf.py --htune 0 --batch_size 64 --epoch 50 --dataset_name bn1 --margin 0.1  --validity_reg 10 

- ae-base-generative-cf.py

  - Implementation of AEGenCF for all datasets

  - Usage: python3 ae-base-generative-cf.py --htune 0 --batch_size 64 --epoch 50 --dataset_name bn1 --ae_path bn1-64-50-target-class--1-auto-encoder.pth --margin 0.1  --validity_reg 10 --ae_reg 10

- oracle-generative-cf.py

  - Implementation of OracleGenCF for all datasets

  - Usage: python3 oracle-generative-cf.py --htune 0 --batch_size 64 --epoch 50 --dataset_name bn1 --cf_path bn1-margin-0.014-validity_reg-54.0-epoch-50-base-gen.pth --oracle_data bn1-fine-tune-size-100-upper-lim-10-good-cf-set.json --margin 0.1 --validity_reg 10 --oracle_reg 10

- model-approx-generative-cf.py

  - Implementation of ModelApproxGenCF for Simple-BN dataset

  - Usage: python3 model-approx-generative-cf.py --htune 0  --batch_size 64 --epoch 50 --dataset_name bn1 --ae_reg 0 --ae_path bn1-64-50-target-class--1-auto-encoder.pth  --margin 0.1 --validity_reg 10 --constraint_reg 10

- model-approx-generative-cf-bnlearn.py

  - Implementation of ModelApproxGenCF for Sangiovese dataset

  - Usage: python3 model-approx-generative-cf-bnlearn.py --htune 0 --batch_size 512 --epoch 50 --dataset_name sangiovese --ae_reg 0 --ae_path sangiovese-512-50-target-class--1-auto-encoder.pth  --margin 0.1 --validity_reg 10 --constraint_reg 10 --constrained_nodes 'BunchN'

- unary-const-generative-cf.py

  - Implementation of ModelApproxGenCF for Adult dataset C1 constraint ( Non Decreasing Age )

  - Usage: python3 unary-const-generative-cf.py --htune 0 --batch_size 2048 --epoch 50 --dataset_name adult --margin 0.1 --validity_reg 10 --constraint_reg 10

- unary-ed-const-generative-cf.py

  - Implementation of ModelApproxGenCF for Adult dataset C2 constraint ( Age-Ed Causal Constraint )

  - Usage: python3 unary-ed-const-generative-cf.py --htune 0 --batch_size 2048 --epoch 50 --dataset_name adult --margin 0.1 --validity_reg 10 --constraint_reg 10

- scm-generative-cf.py

  - Implementation of SCMGenCF for Simple-BN dataset 

  - Usage: python3 scm-generative-cf.py --htune 0 --batch_size 64 --epoch 50 --dataset_name bn1 --margin 0.1 --validity_reg 10 --scm_reg 10

- scm-generative-cf-bnlearn.py

  - Implementation of SCMGenCF for Sangiovese dataset 

  - Usage: python3 scm-generative-cf-bnlearn.py --htune 0 --batch_size 512 --epoch 50 --dataset_name sangiovese --validity_reg 10 --scm_reg 10 --constraint_node 'BunchN'

- contrastive_explanations.py

  - Implementation of CEM for all datasets

  - Usage: python3 contrastive_explanations.py --dataset_name bn1 --htune 0 --train_case_pred 0 --train_case_ae 0 --explain_case 1 --sample_size 3 --timeit 0 --c_init 10 --max_iterations 1000 --beta 0.1 --kappa 0.1 --gamma 1 --c_steps 2

- timeit-base-generative-cf.py

  - Computing the training and evaluaiton time of BaseGenCF

  - Usage: python3 timeit-base-generative-cf.py --htune 0 --batch_size 64 --epoch 50 --dataset_name bn1 --margin 0.1  --validity_reg 10 --cf_path bn1-margin-0.014-validity_reg-54.0-epoch-50-base-gen.pth

- timeit-oracle-generative-cf.py

  - Computing the training and evaluaiton time of Example-based CF

  - Usage: python3 timeit-oracle-generative-cf.py --htune 0 --batch_size 64 --epoch 50 --dataset_name bn1 --cf_path bn1-margin-0.014-validity_reg-54.0-epoch-50-base-gen.pth --oracle_data bn1-fine-tune-size-100-upper-lim-10-good-cf-set.json --margin 0.1 --validity_reg 10 --oracle_reg 10 

## generativecf/scripts/

- blackboxmodel.py

  - Contains the architecture of the ML model to be explained across datasets

- vae_model.py

  - Contains the architecutre of the BaseGenCF and AutoEncoder model

- blackbox-model-train.py

  - Trains the ML model to be explained across datasets

  - Usage: python3 blackbox-model-train.py bn1

- auto-encoder-train.py

  - Trains the Auto Encoder model used in AEGenCF and computing IM Metric

  - Usage: python3 auto-encoder-train.py --dataset_name bn1 --batch_size 64 --epoch 50 --target_class -1

- good-cf-set-gen.py

  - Contains the code for generating labelled queries for OracleGenCF for Simple-BN, Adult dataset

  - Usage: python3 good-cf-set-gen.py --dataset_name bn1 --fine_tune_size 100 --upper_limit 10 --cf_path bn1-margin-0.014-validity_reg-54.0-epoch-50-base-gen.pth

- good-cf-set-gen-bnlearn.py

  - Contains the code for generating labelled queries for OracleGenCF for Simple-BN, Adult dataset

  - Usage: python3 good-cf-set-gen-bnlearn.py --dataset_name sangiovese --fine_tune_size 100 --upper_limit 10 --cf_path sangiovese-margin-0.161-validity_reg-94.0-epoch-50-base-gen.pth --constraint_node BunchN

- datagen.py

  - Creates train, val, test splits with other important processed data for all datasets

  - Usage: python3 datagen.py bn1

- evaluation_functions.py

  - Contains evaluations metrics like Target-Class Validity, Constraint Feasibility Score, etc. for all datasets

- bnlearn_parser.py

  - Reads the sangiovese-scm.txt and creates the SCM 

- helpers.py

  - Contains code for generating the  Adult dataset

- sangiovese-data-gen.py

  - Contains code for processing the Sangiovese dataset

- simple-bn-gen.py

  - Contains the code for generating the Simple-BN dataset


## generativecf-mnist 

Similar description as stated above for generativecf files; with the only difference that evaluation happens for MNIST dataset.

