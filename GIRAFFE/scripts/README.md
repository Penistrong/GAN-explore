# GIRAFFE

## Train Steps

### Evalute Ground-Truth FID-score

```shell
python -m Evalutation.calc_gt_fid <path/to/dataset> --img-size 64 --regex True --gpu 0 --out-file <path/to/out_file> 
```

### Train on the dataset

```shell
python -m GIRAFFE.scripts.train configs/giraffe_on_<dataset_name>.yaml
```
