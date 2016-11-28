# CS 273B Final Project

**Note:** Run `nvidia-smi` before running any scripts related to Neural Net, to make sure that no one else is using the compute!

## Running Models
All the models are run from `run.py`. All the hyper-parameters are initialized in the respective model functions. We have `argparse` functionality to run each of the individual models. There are three running modes `--train`, `--test`, and full pass which is defined by the lack of train/test flag. Furthermore, we can run four different models using the `--model` flag. The model names are: `cae`, `cnn`, `nn`, and `mmnn` for the Convolutional Autoencoder, Convolutional Neural Network, Neural Network (phenotype), and MultiModal Neural Network (CNN + NN), respectively.

### Convolutional Autoencoder command line run

Train the model on train dataset.
```
python run.py --train --model cae
```

Test the model on test dataset.
```
python run.py --test --model cae
```

Run a full pass on the model on using the whole dataset. This should only be used once we find a stable autoencoder, to dimensionality reduction on the whole dataset before feeding it to the other models. **For the time being, do not execute this command.**
```
python run.py --model cae
```

**Note:** Only run the "full pass" option with on Convolutional Autoencoder. There should be functionality preventing this, but just a heads up.

### Convolutional Neural Network

```
python run.py --train --model cnn
python run.py --test --model cnn
```

### Neural Network (phenotype)

```
python run.py --train --model nn
python run.py --test --model nn
```

### MultiModal Neural Network (CNN + NN)

```
python run.py --train --model mmnn
python run.py --test --model mmnn
```

**Note:** If no `--model` is supplied, it defaults to `mmnn`

## Convolutional Autoencoder

### Configurations

| \# of Layers  | Stride Sizes | Filter Sizes | Learning Rate alpha | beta 1 | beta 2 | Layer Activation Ratio rho | Activation Term Mixing Term lambda | Optimizer | Batch Norm |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 1,1 | 3,3 | 0.001 | 0.99 | N/A | 0.5 | 0.6 | Rmsprop | No |
| 2 | 1,3 | 3,3 | 0.001 | 0.99 | N/A | 0.7 | 0.6 | Rmsprop | No |
| 2 | 1,3 | 3,3 | 0.001 | 0.99 | N/A | 0.7 | 0.6 | Rmsprop | Yes|

### Results
#### Input Image
----------------
```
![inputImage](images/inputImage.png?raw=true "Input Image")
```
#### Decoded Image
------------------
```
![decodedImage](images/decodedImage.png?raw=true "Decoded Image")
```
#### Encoded Image
------------------
```
![encodedImage](images/encodedImage.png?raw=true "Encoded Image")
```