import os
from dotenv import load_dotenv

# Loading environment variables from .env file if present
load_dotenv()

class Config:
    # General Settings
    PROJECT_NAME = "Multidisciplinary Deepfake Detection"
    VERSION = "0.1.0"
    
    # Directory Settings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', 'saved_models')
    LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
    REPORT_DIR = os.path.join(BASE_DIR, '..', 'reports')

    # Logging Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    LOG_FILE = os.path.join(LOG_DIR, 'system.log')
    
    # Data Settings
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'sample_data.csv')
    PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')

    # Model Hyperparameters
    CNN_PARAMS = {
        'input_shape': (64, 64, 3),
        'num_classes': 2,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    TRANSFORMER_PARAMS = {
        'input_dim': 512,
        'model_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'output_dim': 10,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    SVM_PARAMS = {
        'kernel': 'linear',
        'C': 1.0
    }

    BAYESIAN_PARAMS = {
        'prior_mean': 0,
        'prior_std': 1
    }

    VISION_TRANSFORMER_PARAMS = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 10,
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    # Blockchain Settings
    BLOCKCHAIN_DIFFICULTY = 4

    # Other Settings
    RANDOM_SEED = 42

    @staticmethod
    def ensure_directories():
        """
        Ensure that all necessary directories exist.
        """
        directories = [
            Config.DATA_DIR,
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.MODEL_DIR,
            Config.LOG_DIR,
            Config.REPORT_DIR
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    @staticmethod
    def print_config():
        """
        Print the current configuration settings.
        """
        config_dict = {attr: value for attr, value in Config.__dict__.items() if not callable(getattr(Config, attr)) and not attr.startswith("__")}
        for key, value in config_dict.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Ensure all directories exist
    Config.ensure_directories()
    
    # Print configuration
    Config.print_config()
