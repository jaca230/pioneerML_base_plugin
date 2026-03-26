from .pipeline import full_chain_pipeline
from pioneerml_base_plugin.utils.config_loader import load_full_chain_config


def load_config() -> dict:
    return load_full_chain_config()


__all__ = ["full_chain_pipeline", "load_config"]
