# Initializing src module and its submodules

import os
import logging

# Importing config to make it accessible from src directly
from .config import config

# Initializing dataset submodule
from .dataset import data_loader, data_preprocessor, data_splitter, data_augmentation

# Initializing models submodule
from .models import cnn, transformer, svm, bayesian, vision_transformer

# Initializing training submodule
from .training import cnn_training, transformer_training, svm_training, bayesian_training, vision_transformer_training

# Initializing evaluation submodule
from .evaluation import cnn_evaluation, transformer_evaluation, svm_evaluation, bayesian_evaluation, vision_transformer_evaluation

# Initializing utils submodule
from .utils import logger, metrics, visualization, helpers, file_utils, data_utils

# Initializing processing submodule
from .processing import audio_processing, video_processing, image_processing, text_processing

# Importing other modules
from . import blockchain, nlp, dsp, train, evaluate

# Configuring logging
log_file = os.path.join(config.LOG_DIR, 'system.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Initialization of the src module and its submodules is complete.")
