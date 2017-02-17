# CS 273B Final Project

**Note:** Run `nvidia-smi` before running any scripts related to Neural Net, to make sure that no one else is using the compute!

## Running Models
All the models are run from `run.py`. All the hyper-parameters are initialized in the respective model functions. We have `argparse` functionality to run each of the individual models. There are three running modes `--train`, `--test`, and full pass which is defined by the lack of train/test flag. Furthermore, we can run four different models using the `--model` flag. The model names are: `cae`, `cnn`, `nn`, and `mmnn` for the Convolutional Autoencoder, Convolutional Neural Network, Neural Network (phenotype), and MultiModal Neural Network (CNN + NN), respectively.

### Run Blackout
```
python run.py --blackout --model cnn --chkPointDir /data/fdsafd
```

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

### Autoencoder

```
python run.py --train --model ae
python run.py --test --model ae
```

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

### Additional Flags Added by Yuki

```--numIters``` - Number of batches to run for the training set. Defaults to 200.

```--chkPointDir``` - Directory where checkpoint files are stored. Defaults to '/data/ckpt'

```--overrideChkpt``` - Add this flag if you want to ignore the checkpoint files already in the checkpoint directory.

```--dataDir``` - Specify the data directory to load the samples from.

### Example Usages:

Training CNN with 1000 iterations, store checkpoints in '/data/ckpt'

	python run.py --train --model cnn --numIters 1000

Training MMNN with 1000 iterations from a checkpoint stored under file './chkpt'

	python run.py --train --model mmnn --numIters 1000 --chkPointDir ./chkpt

Training CNN with 1000 iterations, store checkpoints in './chkpt', but ignore any checkpoints already stored in that directory

	python run.py --train --overrideChkpt --model cnn --numIters 1000 --chkPointDir ./chkpt --dataDir /data/swap_partial_13_binaries_reduced_compressed/


## Network Architecture Training Notes and Tips:

- Batch Normalization as its introductory paper suggests, improves performance and acheives the optimal solution in fewer iterations many of
the times. This is true for Multi-Modal architectures, including the Vanilla NN and the Convolutional Neural Network. But for the
Convolutional Autoencoder this does not seem to be true completely. Batch Norm does help, but Autoencoders in general are *extremely*
susceptible to weight initialization. We hope to use the Xavier initialization sometime soon.

- L1 regularizations work suprisingly well for Deeper networks (Convolutional and Vanilla), especially when Residual Networks are not used.
Deep networks may have sparse features in many layers, and L1 regularization as popularly known is best for enforcing sparsity.
Residual Networks get around sparse features using the identity mapping in their skip connections.

- To train such deep networks, we need to train it for at least 30 Epochs in practice, after which we should be mindful of overfitting
and generalization error. [Note: Epoch is defined as a pass through the entire dataset. After each epoch, it is preferable to re-shuffle
the dataset to allow for better generalization and prevent overfitting]

## Convolutional Autoencoder

### Configurations

| \# of Layers  | Stride Sizes | Filter Sizes | Learning Rate alpha | beta 1 | beta 2 | Layer Activation Ratio rho | Activation Term Mixing Term lambda | Optimizer | Batch Norm |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 1,1 | 3,3 | 0.001 | 0.99 | N/A | 0.5 | 0.6 | Rmsprop | No |
| 2 | 1,3 | 3,3 | 0.001 | 0.99 | N/A | 0.7 | 0.6 | Rmsprop | No |
| 2 | 1,3 | 3,3 | 0.001 | 0.99 | N/A | 0.7 | 0.6 | Rmsprop | Yes|

### Results
#### Input Image
![inputImage](images/inputImage.png?raw=true "Input Image")
----------------
#### Decoded Image
![decodedImage](images/decodedImage.png?raw=true "Decoded Image")
------------------
#### Encoded Image
![encodedImage](images/encodedImage.png?raw=true "Encoded Image")