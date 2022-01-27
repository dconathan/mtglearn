from kfp.v2.dsl import component, Dataset, Output


@component(base_image="python:3.7-slim", packages_to_install=["faker"])
def prepare_dummy_data(dataset: Output[Dataset]):

    # the trainer script expects the suffix of the train data file to be .txt
    dataset.path = dataset.path + ".txt"

    n = 1000

    import faker

    fake = faker.Faker()

    with open(dataset.path, "w") as f:
        for _ in range(n):
            f.write(fake.sentence(50) + "\n")

    print("done!")
