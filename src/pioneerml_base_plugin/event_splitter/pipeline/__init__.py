from pioneerml.pipeline.pipelines.inference import inference_pipeline
from pioneerml.pipeline.pipelines.training import training_pipeline
from pioneerml_base_plugin.utils.config_loader import load_model_pipeline_config

MODEL_KEY = "event_splitter"


def load_config() -> dict:
    return load_model_pipeline_config(MODEL_KEY)


__all__ = ["MODEL_KEY", "training_pipeline", "inference_pipeline", "load_config"]
