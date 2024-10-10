cuda_device=$1
gamma=$2
export CUDA_VISIBLE_DEVICES=${cuda_device}

python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=10 exp.logging=False
# python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=101 exp.logging=False
# python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=1010 exp.logging=False
# python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=10101 exp.logging=False
# python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=101010 exp.logging=False