# transformer-text-classifier

Text classification using Transformers, TensorFlow, and Keras. Adapted for register classification, based on original version by [Sampo Pyysalo](https://github.com/spyysalo/transformer-text-classifier).

## Quickstart

Install requirements

    python -m pip install -r requirements.txt

Download sample data

    git clone https://github.com/spyysalo/ylilauta-corpus.git

Train

    python train.py --model_name TurkuNLP/bert-base-finnish-cased-v1 \
        --train ylilauta-corpus/data/ylilauta-train-1000.txt \
        --dev ylilauta-corpus/data/ylilauta-dev.txt

Launch multiple instances with slurm:

    sbatch slurm_train_arg.sh [MODEL_NAME] [MODEL_ALIAS] [SOURCE_LANGUAGE] [TARGET_LANGUAGE] [LRs] [EPOCHSs] [INSTANCEs]

    e.g., to test two learning rates and number of epochs each 3 times:
    sbatch slurm_train_arg.sh "jplu/tf-xlm-roberta-large" xlmrL en fi "8e-6 1e-5" "3 5" "1 2 3"
