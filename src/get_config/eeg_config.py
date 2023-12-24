from loguru import logger
import sys

from get_config.eeg_config_validate import EEGConfigConfig, EEGConfigException

class EEGConfig:

    CONFIG_FILE_PATH = "src/config/ecg.config"

    def __init__(self, config_block, config_file_path=None):
        if config_file_path is not None:
            self.CONFIG_FILE_PATH = config_file_path
        logger.info("Read config file: {}", self.CONFIG_FILE_PATH)
        logger.debug("Config: {}", config_block)

        config = {}
        try:
            config = EEGConfigConfig(self.CONFIG_FILE_PATH, config_block)
        except EEGConfigException as e:
            logger.error("Invalid config file: {}", e)
            sys.exit(1)
    
        self.file_name = config[config_block]["file_name"].strip()
        self.multiplier = float(config[config_block]["multiplier"].strip())
        self.data_type = config[config_block]["data_type"].strip()
        self.sigma = [float(x) for x in config[config_block]["sigma"].strip().split(',')]
        self.config_block = config_block
        self.data_path = config["DEFAULT"]["data_path"].strip()
        self.img_path = config["DEFAULT"]["img_path"].strip()

    def getDataType(self):
        return self.data_type
    
    def getFileName(self):
        return self.file_name
    
    def getMultiplier(self):
        return self.multiplier
    
    def getConfigBlock(self):
        return self.config_block
    
    def getDataPath(self):
        return self.data_path
    
    def getImgPath(self):
        return self.img_path
    
    def getSigma(self):
        return self.sigma

    def __str__(self):
        logger.debug("toString {}", self.config_block)
        return ((
                "\nConfig block: {0}\n" +
                "File mame: {1}\n" +
                "Multiplier: {2}\n" +
                "Default data path {3}\n" +
                "Default images path {4}\n"
                ).format(self.config_block, self.file_name, self.multiplier, self.data_path, self.img_path))