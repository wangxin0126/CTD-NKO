cuda_device=$1
export CUDA_VISIBLE_DEVICES=${cuda_device}

# generate results for Table 3
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=10 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=101 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=1010 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=10101 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=101010 exp.logging=False model.use_global=False model.name=CTD_NKO_local

python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=10 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=101 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=1010 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=10101 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=two exp.seed=101010 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now

python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=10 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=101 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=1010 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=10101 exp.logging=False model.use_global=False model.name=CTD_NKO_local
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=101010 exp.logging=False model.use_global=False model.name=CTD_NKO_local

python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=10 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=101 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=1010 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=10101 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now
python runnables/train.py  +dataset=cancer_sim  +backbone/wt=cancer +backbone/hparams/cancer=four exp.seed=101010 exp.logging=False exp.update_freq=-1 model.name=CTD_NKO_now

# generate results for ablation study of using balnaced representation
python runnables/train.py  +dataset=cancer_sim  +backbone/br=cancer +backbone/hparams/cancer=four exp.seed=10 exp.logging=False model.name=CTD_NKO_br
python runnables/train.py  +dataset=cancer_sim  +backbone/br=cancer +backbone/hparams/cancer=four exp.seed=101 exp.logging=False model.name=CTD_NKO_br
python runnables/train.py  +dataset=cancer_sim  +backbone/br=cancer +backbone/hparams/cancer=four exp.seed=1010 exp.logging=False model.name=CTD_NKO_br
python runnables/train.py  +dataset=cancer_sim  +backbone/br=cancer +backbone/hparams/cancer=four exp.seed=10101 exp.logging=False model.name=CTD_NKO_br
python runnables/train.py  +dataset=cancer_sim  +backbone/br=cancer +backbone/hparams/cancer=four exp.seed=101010 exp.logging=False model.name=CTD_NKO_br