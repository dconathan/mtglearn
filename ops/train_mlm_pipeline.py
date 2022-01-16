from kfp.v2.dsl import pipeline


@pipeline(name="train-mlm-pipeline", pipeline_root="gs://mtglearn/pipelines/train-mlm")
def train_mlm_pipeline():
    pass


if __name__ == "__main__":

    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        Compiler().compile(train_mlm_pipeline, f.name, pipeline_parameters={})
        job = aiplatform.PipelineJob(
            display_name="train-mlm-pipeline", template_path=f.name, project="mtglearn"
        )
        job.submit()
