import os
import uuid

from google.cloud import storage

bucket_name = os.environ.get("GCS_BUCKET_NAME")
if not bucket_name:
    raise RuntimeError("GCS_BUCKET_NAME is not configured. Set it to a sandbox bucket before running this test.")

object_name = f"integration-tests/gcs-check-{uuid.uuid4().hex}.txt"
expected = b"GCS integration test: upload-download verified"

client = storage.Client()
blob = client.bucket(bucket_name).blob(object_name)

try:
    blob.upload_from_string(expected, content_type="text/plain")
    actual = blob.download_as_bytes()

    assert actual == expected, "Downloaded content differs from uploaded content"
    print("PASS: GCS upload/download")
    print(f"Object: gs://{bucket_name}/{object_name}")
finally:
    if blob.exists():
        blob.delete()
