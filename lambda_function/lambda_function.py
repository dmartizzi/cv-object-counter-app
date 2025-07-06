import os
import io
import base64
import logging
import json
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print("Loading Lambda function")


def get_image(img_name, s3bucket, s3prefix):
    """
    Retrieve an image for a given id

    Input :
        img_name : str, image name

        s3bucket : str, name of the bucket

        s3prefix : str, prefix

    Output :
        img : ImageFile, image object
    """
    client = boto3.client("s3")
    key = os.path.join(s3prefix, img_name)
    response = client.get_object(Bucket=s3bucket, Key=key)
    img = response["Body"].read()

    return img


def lambda_handler(event, context):

    print("Context:::", context)
    print("EventType::", type(event))

    runtime = boto3.Session().client("sagemaker-runtime")

    s3imgbucket = event["s3_bucket"]
    s3imgprefix = event["s3_prefix"]
    endpoint_name = event["endpoint_name"]

    image_name = event["image_name"]
    img_bytes = get_image(image_name, s3imgbucket, s3imgprefix)

    encoded_img = base64.b64encode(img_bytes).decode("ASCII")
    json_request = json.dumps({"image": encoded_img})

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json_request,
    )

    result = response["Body"].read().decode("utf-8")

    body = {"image": encoded_img, "prediction": json.loads(result)}

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/plain", "Access-Control-Allow-Origin": "*"},
        "type-result": str(type(result)),
        "Content-Type-In": str(context),
        "body": body,
    }
