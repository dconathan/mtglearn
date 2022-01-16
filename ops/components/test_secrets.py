from kfp.v2.dsl import component


@component(
    base_image="python:3.7-slim", packages_to_install=["google-cloud-secret-manager"]
)
def test_secrets():

    from google.cloud import secretmanager

    name = "projects/mtglearn/secrets/TEST_SECRET/versions/latest"

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": name})
    secret = response.payload.data.decode("UTF-8")

    print(f"TEST_SECRET is {secret}")
