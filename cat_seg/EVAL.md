Update the variables in config and run evaluate.py

``` bash
conda activate segtto
python3 evaluate.py 
```

or else 

``` bash
sh eval.sh --config-file configs/seg_tto.yaml --datasets 'cryonuseg_sem_seg_test,cwfid_sem_seg_test'
