cuda_device=$1
export CUDA_VISIBLE_DEVICES=${cuda_device}

python runnables/train.py  +dataset=mimic3_real  +backbone/wt=mimic_real +backbone/hparams=mimic_real exp.logging=False exp.seed=10
# python runnables/train.py  +dataset=mimic3_real  +backbone/wt=mimic_real +backbone/hparams=mimic_real exp.logging=False exp.seed=101
# python runnables/train.py  +dataset=mimic3_real  +backbone/wt=mimic_real +backbone/hparams=mimic_real exp.logging=False exp.seed=1010
# python runnables/train.py  +dataset=mimic3_real  +backbone/wt=mimic_real +backbone/hparams=mimic_real exp.logging=False exp.seed=10101
# python runnables/train.py  +dataset=mimic3_real  +backbone/wt=mimic_real +backbone/hparams=mimic_real exp.logging=False exp.seed=101010