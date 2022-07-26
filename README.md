# CGAN_Cmap

This is the code and data for [CGAN-Cmap: protein contact map prediction using deep generative adversarial neural networks](https://www.biorxiv.org/content/10.1101/2022.07.26.501607v1) paper submitted in Briefings in Bioinformatics. 

## Code Usage

### Dependencies

- Tensorflow == 2.3.0
- H5Py == 2.10.0
- Matplotlib == 3.5.1
- Numpy == 1.21.6

### Commands

- `--traintest`: 
  - Options: 
    - `traintest` which train the model and test that
    - `test` which test the saved model
- `--test_data`: The type of data set to use. default = initial, options: [initial, CAMEO, casp12,casp13,casp14]
- `--batch_size`: The batch size for the model training (default = 4)
- `--n_epoch`: The number of epochs for training (default = 500)
- `--save_step` : Save models every x epoch (default = 50)
- `--lr` : Learning rate (default = 2e-4)
- `--SE_concat` : Number of SE_Concat block used in the generator (default = 3)
- `--Premodel_name` : Pretrain model name (default = CGAN_Cmap.h5)

### Data and Models

The data can be downloaded from this [link](https://drive.google.com/file/d/1jsbmcryO1_e7bsrbAZ9B9Ljhgm5j6U4K/view?usp=sharing) (Folder includes training, validation and initial test sets ( ready to use for training) and Casp 11, 12, 13, 14, and CAMEO.). You have to extract that to the data folder ( it would be like `data/`) . To download the pretrained models, you can use this [link](). You have to extract the models under GANTL folder (it would be like `GANTL/model/`). 

### Using Example

To train a model you can use the following command:

```python
python main.py --traintest traintest
```

The code will train the model and save the models in `GANTL/model` folder. It also save the images obtained during the training in `GANTL/images`. The final predictions will be saved in `GANTL/prediction`.

To test the model, one can run the following command:

```python
python main.py --traintest test
```

The result will be saved in  `GANTL/prediction`.

## Citation 
If you use this works please cite 
CGAN-Cmap: protein contact map prediction using deep generative adversarial neural networks uploaded in biorxiv:
```
@article {Madani2022.07.26.501607,
	author = {Madani, Mohammad and Behzadi, Mohammad Mahdi and Song, Dongjin and Ilies, Horea and Tarakanova, Anna},
	title = {CGAN-Cmap: protein contact map prediction using deep generative adversarial neural networks},
	elocation-id = {2022.07.26.501607},
	year = {2022},
	doi = {10.1101/2022.07.26.501607},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/07/28/2022.07.26.501607},
	eprint = {https://www.biorxiv.org/content/early/2022/07/28/2022.07.26.501607.full.pdf},
	journal = {bioRxiv}
}
```
