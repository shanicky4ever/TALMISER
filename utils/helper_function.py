import yaml
import pathlib
import os
import shutil


def makedir(path, del_before=True):
    if del_before and os.path.exists(path):
        shutil.rmtree(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def save_obj(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f, allow_unicode=True)


def load_obj(path):
    with open(path, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj
