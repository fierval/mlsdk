# Welcome to the Machine Learning SDK

## Installation

1. Recommend starting with a [Python 3.5 Anaconda distribution](https://www.anaconda.com/download/) as it contains lots of packages by default (`numpy`, `scipy`, `accelerate` with `numba` that we are using to speed things up to name a few).

2. [Create a virtual environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)

3. Install `opencv`, `tensorflow`, `keras`, `pydensecrf`, `randomcolor`.

4. Optional: install `holoviews`

5. Clone and install from GitHub
    ```bash
    git clone git@github.com:iunullc/machine-learning-sdk.git
    cd machine-learning-sdk
    python setup.py install
    ```
6. If contributing to the SDK:
    ```bash
    python setup.py develop
    ```
The APIs are now available under ```iuml``` namespace.

OpenCV can be easily installed with Anaconda:

```bash
conda install -c menpo opencv
```

**Note**: Since the package does not contain a requirements file yet, expect to incrementally add required packages if needed (you will be notified when you try to import something)
{: .notice--warning}

## Quick Start: Segmenting plants/buds

For each established pipeline (e.g.: bud counting) we need to train or enhance a model specific to a given greenhouse. Details like camera positioning, lighting, plant positioning change and so our neural net needs to be made aware of these changes.

The approach is to create (and reuse) "labs" in Jupyter notebook form that allow us to experiment, finetune and release model weights stored in [HDF5](https://support.hdfgroup.org/HDF5/whatishdf5.html) format.

[A complete example](https://github.com/iunullc/machine-learning-sdk/blob/master/Notebooks/Smith/Smith%20Plants%20Counting%20Experiments.ipynb) can be copied and modified for each particular case.

### Choosing Data

Data needs to be picked in such a way that most conditions the camera sees in the greenhouse are represented. Our cameras take a lot of very similar pictures. The dataset needs to reflect variance, not similarity of our data.

### Dataset Annotation

[Creating Fiji-based annotations]()

### Creating training and validation datasets.

The raw dataset together with its annotation masks extracted from Fiji, should reside in the following directory structure:

```
root_source
|
|___ images
|    |
|    |___ snapshot1.jpg
|
|___ masks
|    |
|    |___ snapshot1.jpg
```

Here "images" and "masks" should contain the same number of files, where each image file has a corresponding mask file. Currently we support .jpg and .png formats, but other formats can be easily added.

Each mask is a matrix the same size (width, height) as its corresponding image which contains '1's for each pixel belonging to a bud (plant, flower), and '0' otherwise. If you are saving results of Fiji Weka classification as JPEGs, there will be an extra step to convert these JPEGs to the correct format:

```python
from iuml.tools.image_utils import convert_mask_fiji_to_class
convert_mask_fiji_to_class(fiji_masks, masks_path)
```

This method will take two path: source ```fiji_masks``` and destination ```masks_path``` and convert the masks to the right representation. Training may now start.

### Training

#### Creating trainer object

We use an [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) type network called [U-Net](https://arxiv.org/abs/1505.04597) with some adjustments.

Create the trainer object:

```python
params = dict(batch_size = 2, epochs=20, images='images', masks='masks', image_shape=(816, 608))
trainer = create_trainer('Unet', training_root_path, **params)
```

Here `images` and `masks` parameters are the names of subfolders off of `root_source` shown on the above diagram.

At this point model hyper-parameters may be fine-tuned by setting ```trainer.optimizer```. The default is  `Adam(lr=1e-5)`

The key parameter `training_root_path` may not exist before the call. Calling `create_trainer` will have created it. We will populate the training dataset in the next step.
{: notice-info}

#### Create training and validation datasets

Calling `split_train_val_data` method of our `trainer` object will create the required dataset structure:

```python
trainer.split_train_val_data(training_root_path, val_fraction = 0.2)
```

```
        training_root_path
        |
        |___ training
        |    |
        |    |___images
        |    |
        |    |___masks
        |
        |___ validation
        |    |
        |    |___images
        |    |
        |    |___masks

```

`val_fraction` parameter (default: 0.1) will place a randomly selected fraction of the input under `validation` subfolder.

#### Run training

```python
trainer.train()
```
Get familiar output from Keras:

```
inputs shape: (?, 608, 816, 3)
pool1 shape: (?, 304, 408, 64)
pool2 shape: (?, 152, 204, 128)
pool3 shape: (?, 76, 102, 256)
pool4 shape: (?, 76, 102, 512)
conv7 shape: (?, 152, 204, 256)
conv8 shape: (?, 304, 408, 128)
conv9 shape: (?, 608, 816, 2)
Starting training for C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cnn\training, 70 samples
Validation dataset: C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cnn\validation, 18 samples
Epoch 1/30
34/35 [============================>.] - ETA: 0s - loss: 0.4947 - acc: 0.8126Epoch 00001: val_loss improved from inf to 0.47567, saving model to C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cnn\model_unet.hd5
```
Runs training from scratch.
The resulting model is saved under `training_root_path/model_unet.hd5`

#### Run more training

At this point if we want to run more training we need to set the `weights_file` property of the `trainer` object to point to the currently created model weigths file:

```python
trainer.weights_file = os.path.join(training_root_path, trainer.model_file_name)
tainer.train()
```

### Running the model on a test set

```python
trainer.predict(dir=test_dir)
```

Assuming we are still using the same model we just trained, the command above will create masks for images stored in a ```test_dir```. Alternatively, if images are already pre-loaded into a list, the same API can be called with ```data=list_of_images```

If all we are interested in is creating masks from an already trained model (i.e. we have never trained a model in this notebook), it may be done as follows:

```python
traner = create_trainer('Unet', '', weights_file=model_weights_file, image_shape=(816, 608), n_classes = 2)
trainer.predict(dir=test_dir)
```

I.e., call `create_trainer` with an empty `training_root_path`
