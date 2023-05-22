import json


class Configs:
    def __init__(self, configs, key):
        self.configs = configs
        self.root_key = key

    def __getattr__(self, item):
        try:
            value = self.configs[item]
        except KeyError:
            raise Exception(f'Key not found: {self.root_key}.{item}')

        return Configs(self.configs[item], f'{self.root_key}.{item}') if type(value) is dict else value

    def __delattr__(self, item):
        try:
            del self.configs[item]
        except KeyError:
            raise Exception(f'Key not found: {self.root_key}.{item}')

    def __str__(self):
        return f'{self.root_key}: {self.configs}'

    def hasattr(self, item):
        try:
            value = self.configs[item]
        except KeyError:
            return False
        return True

    def add(self, key, value, overwrite=True):
        if key in self.configs.keys():
            if overwrite:
                self.configs[key] = value
            else:
                raise Exception(f'Item {self.root_key}.{key} already exists in config')
        else:
            self.configs[key] = value
        
    def edit(self, key, value):
        try:
            self.configs[key] = value
        except KeyError:
            raise Exception(f'Key not found: {self.root_key}.{str(key)}')

def read_json_configs(filepath: str):
    configs_obj = Configs(json.load(open(filepath)), filepath.split('/')[-1].split('.')[0])
    return configs_obj


def read_dict_configs(configs_dict):
    configs_obj = Configs(configs_dict,'configs_obj')
    return configs_obj
