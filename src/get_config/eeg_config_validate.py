from configparser import ConfigParser
from pathlib import Path


class EEGConfigException(Exception):
    pass

class EEGConfigConfig(ConfigParser):
    def __init__(self, config_file, config_block):
        super(EEGConfigConfig, self).__init__()

        if not Path(config_file).is_file():
            raise EEGConfigException(
                    'The config file %s does not exist' % config_file)
                    
        self.read(config_file)
        self.validate_config(config_block)

    def validate_config(self, config_block):
        required_values = {
            'DEFAULT': {
                'data_path' : None,
                'img_path' : None
            },
            '%s' % (config_block): {
                'file_name': None,
                'multiplier': None,
                'data_type': ('openbci'),
                'sigma': None
            }
        }

        for section, keys in required_values.items():
            if section not in self:
                raise EEGConfigException(
                    'Missing section "%s" in the config file' % section)

            for key, values in keys.items():
                if key not in self[section] or self[section][key] == '':
                    raise EEGConfigException((
                        'Missing value for "%s" under section "%s" in ' +
                        'the config file') % (key, section))

                if values:
                    if self[section][key] not in values:
                        raise EEGConfigException((
                            'Invalid value for "%s" under section "%s" in ' +
                            'the config file') % (key, section))