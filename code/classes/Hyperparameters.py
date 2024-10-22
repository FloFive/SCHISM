import configparser  # Import configparser for .ini file handling


# Hyperparameters class
class Hyperparameters:
    def __init__(self, **kwargs):
        """
        Initializes the Hyperparameters instance with optional parameters.
        """
        self.img_side_length = kwargs.get('img_side_length', 512)
        self.num_sample = kwargs.get('num_sample', 1000)
        self.imgtype = kwargs.get('imgtype', "scanner")
        self.pretrained = kwargs.get('pretrained', True)
        self.backbone = kwargs.get('backbone', 'resnet50')
        self.image_preprocessing_functions = kwargs.get('image_preprocessing_functions', [])
        self.metrics = kwargs.get('metrics', ["CategoricalCrossentropy, OneHotMeanIoU"])
        self.featuremaps = kwargs.get('featuremaps', 16)
        self.epochs = kwargs.get('epochs', 20)
        self.val_split = kwargs.get('val_split', 0.8)
        self.displaySummary = kwargs.get('displaySummary', True)
        self.maxNorm = kwargs.get('maxNorm', 3)
        self.learningRate = kwargs.get('learningRate', 1e-4)
        self.batchNorm = kwargs.get('batchNorm', True)
        self.batch_size = kwargs.get('batch_size', 3)
        self.save_model = kwargs.get('save_model', True)
        self.dropOut = kwargs.get('dropOut', True)
        self.dropOutRate = kwargs.get('dropOutRate', 0.4)
        self.L2 = kwargs.get('L2', 1e-4)
        self.early_stopping = kwargs.get('early_stopping', [])
        self.loss_early_stopping = kwargs.get('loss_early_stopping', False)
        self.patience = kwargs.get('patience', 25)
        self.run_name = kwargs.get('run_name', "default_run")

    def __repr__(self):
        return f"Hyperparameters({self.__dict__})"

    def save_to_ini(self, file_path):
        """Saves hyperparameters to an .ini file."""
        config = configparser.ConfigParser()
        config['Hyperparameters'] = {k: str(v) for k, v in self.__dict__.items()}
        with open(file_path, 'w') as configfile:
            config.write(configfile)

    @classmethod
    def load_from_ini(cls, file_path):
        """Loads hyperparameters from an .ini file."""
        config = configparser.ConfigParser()
        config.read(file_path)
        params = {k: v for k, v in config['Hyperparameters'].items()}
        return cls(**params)
