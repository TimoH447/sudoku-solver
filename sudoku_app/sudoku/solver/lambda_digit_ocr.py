import json
import boto3

def lambda_digit_recognition(bucket: str,key: str) -> int:
    """
    predict digit images with tensorflow model in lambda.
    arguments:
    bucket: name of the bucket
    key: key of the image
    returns a digit
    """
    # payload needs to specify the bucket and the key of an image of a digit we want to read
    payload = {
        "bucket": bucket,
        "image_key": key
        }

    # Erstellen Sie einen Lambda-Client mit der IAM-Rolle
    lambda_client = boto3.client('lambda')

    # Rufen Sie die Lambda-Funktion auf
    response = lambda_client.invoke(
        FunctionName='digit-ocr-v1',
        Payload=json.dumps(payload)
    )

    # Lesen Sie den Rückgabewert aus der Antwort des Lambda-Clients
    payload_str = response['Payload'].read().decode('utf-8')
    payload_dict = json.loads(payload_str)
    predicted_digit = payload_dict['predicted_digit']
    
    return predicted_digit