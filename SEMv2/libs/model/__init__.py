from .model import Model


def build_model(cfg):
    model = Model(cfg)
    return model
