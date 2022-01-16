import tempfile

from kfp.v2.dsl import pipeline
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

from .components.prepare_fake_data import prepare_fake_data
from .components.train_mlm import train_mlm


@pipeline(name="test-pipeline", pipeline_root="gs://mtglearn/pipelines/test")
def test_pipeline():

    prepare_data_op = prepare_fake_data()

    train_file = prepare_data_op.outputs["dataset"]
    train_mlm(
        model_name_or_path="roberta-base",
        train_file=train_file,
        batch_size=1,
        learning_rate=5e-5,
        num_train_epochs=0.5,
        save_steps=500,
    )


if __name__ == "__main__":

    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        Compiler().compile(test_pipeline, f.name, pipeline_parameters={})
        job = aiplatform.PipelineJob(
            display_name="test-pipeline", template_path=f.name, project="mtglearn"
        )
        job.submit()
