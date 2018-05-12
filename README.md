# Flower Detection and Counting Pipeline

End-to-end system for training, testing, and deploying ML based flower detection.

## Terminology
*Positive* (sample, class, etc) - anything containing the actual flower of the kind we are training to detect  
*Negative* (sample, class, etc) - anything **not** containing  the actual flower.  
*Tile* - an image of size *l*x*l* extracted from an original camera shot, suitable for training (by default *l* = 32)

## Prerequisits
1. Python 3.5 (2.7.13 cautiously supported), Anaconda distro preferred since it has all the necessary components: numpy, scikit-learn, etc.
2. OpenCV 3.3+
2. CUDA v9.0 with compatible cuDNN 
3. Tensorflow v1.6+
4. Keras v2.0+
5. iuml package developed in this repo

Install iuml package from the root directory:

```sh
$ python setup.py install
$ python setup.py develop
``` 
from the root of this repository.

## Workflow
We use the file system to organize our dataset.

### Data Flow

![Data Flow](docs/images/data_flow.png)

1. From available images, pick two sets:
    
    - "Positive" - images containing plants with buds/flowers that will be extracted through manual annotation
    - "Negative" - images with no buds/flowers or no plants at all. Negative samples will be extracted automatically. Alternatively, "negative" samples may be manually produced by annotating this image set.
    
    Drop these images into their corresponding folders based on the diagram above. 
    
2. *Data Extraction*. Use annotation tool (described below) to select positive examples. For negative examples, the tool will either produce them automatically or manually from the "negaive" image set. This step will create *Tiled Images*
3. *Training and Validation*. Prep extracted tiles for training and train. We split the dataset into "training" and "validation" datasets. The training tool does it automatically.

### Data Extraction

Run the data extraction tool. From the root of this git repository:

```sh
$ cd data-gathering
$ python extract_data.py --root-source <root_source> --root-dest <tiled_images_root> --scale-down 2
```

Here:
```sh
--root-source - root folder of the raw images picked for training set creation
```
Optional parameters:

```sh
--positive - name of the "positive" class subfolder (default: positive)
--negative - name of the "negative" class subfolder (default: negative)
--root-dest - destination root folder for tiled images. If specified - tiling is performed.
--scale-down - display factor. Scale the image down so it fits on screen
--tile l - length of the tile for l x l tiles (32 by default)
--manual-negative - if specified, use the manual extraction tool for negative sample extraction
--sample-tiles f - 0 < f < 1. If specified AND --manual-negative is NOT specified, will only produce a fraction f of all tiles extracted from negative images
```

Of all these folders, only the root source tree needs to exist.

The tool will launch the annotation experience for the positive images (and then optionally - for the negative ones).

All annotations are persisted to a _.json_ file, so if additional annotation is necessary it can be done by running the tool again.

Actual annotations are created by marking centers of flowers/buds with the left mouse button. For negative samples areas at a reasonable distance from buds/flowers should be picked.

Useful keystrokes:
```sh
n - go to next image
p - go to previous image
x - remove last annottion
d - toggle image data display in the top left corner
o - toggle automatic annotation while simply moving the mouse over the image
ESQ, q - finish the manual annotation process and produce tiles
```

Here is an example of an annotated image:

![Annotated Positive](docs/images/annotated_positive.png)

The tool extracts tiles used for training the classifier:

![Positive Example](docs/images/positive_example.jpg)

### Training

Training by transfer learning (fine-tuning) is implemented with Keras (TensorFlow backend). We support VGG-16, InceptionV3, and Xception networks.

Run the following from "training" subdirectory of the repository.

```sh
$ python train_classifier.py --root-tiles <dataset_dir> --root-train <training_dir>
```

Here:
- dataset_dir - directory that contains tiled images, created in the previous step
- training_dir - directory where "training" and "validation" datasets are stored.

```--root-tiles``` can be used when running training for the first time to create training dataset structure.

For the rest of the options run:

```sh
$ python train_classifier.py -h
```
