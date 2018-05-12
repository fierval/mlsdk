# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .train import TrainClassifierBase
from .train_inception_base import TrainClassifierInceptionBase
from .classifiers import TrainClassifierVgg16, TrainClassifierInceptionV3, TrainClassifierXception, TrainClassifierResnet50
from .crf import get_dcrf_mask