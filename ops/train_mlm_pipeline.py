import tempfile

from kfp.v2.dsl import pipeline
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

from .components.prepare_data import prepare_data
from .components.train_mlm import train_mlm


@pipeline(name="train-mlm-pipeline", pipeline_root="gs://mtglearn/pipelines/train-mlm")
def train_mlm_pipeline():

    prepare_data_op = prepare_data(seed=624352, n_duplicates=5)

    train_file = prepare_data_op.outputs["dataset"]
    train_mlm(
        model_name_or_path="roberta-base",
        train_file=train_file,
        batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=4,
        save_steps=2500,
    ).add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", "NVIDIA_TESLA_T4"
    ).set_gpu_limit(
        1
    )


if __name__ == "__main__":

    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        Compiler().compile(train_mlm_pipeline, f.name, pipeline_parameters={})
        job = aiplatform.PipelineJob(
            display_name="train-mlm-pipeline", template_path=f.name, project="mtglearn"
        )
        job.submit()
