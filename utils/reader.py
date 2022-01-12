import yaml


def read_config():
    stream = open('config/settings.yaml', 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config
