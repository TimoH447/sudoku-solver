import io
import json
from PIL import Image
import boto3

class AwsInterface:
    def __init__(self,in_development):
        if in_development:
            with open("/Users/timoh/OneDrive/Dokumente/Bildung/Programmieren/sudoku-solver/env_var.json","r") as f:
                env = json.load(f)
            self.s3 = boto3.client('s3',region_name = env['AWS_REGION'],aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],aws_access_key_id=env['AWS_ACCESS_KEY'])
            self.lambda_client = boto3.client('lambda',region_name = env['AWS_REGION'],aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],aws_access_key_id=env['AWS_ACCESS_KEY'])
        else:
            self.s3 = boto3.client('s3', region_name="eu-central-1")
            self.lambda_client = boto3.client('lambda',region_name="eu-central-1")

    def upload_photo(self,image,image_name):
        buffer = io.BytesIO()
        image.save(buffer,format='png')
        buffer.seek(0)

        # info: put_object will throw an error when its unsuccessful
        response = self.s3.put_object(Bucket='sudoku-solver-bucket',Key=image_name,Body=buffer)
        return response

    def delete_photo(self,bucket_name,image_key):

        response= self.s3.delete_object(Bucket=bucket_name,Key=image_key)
        return response

    def lambda_digit_recognition(self,bucket: str,key: str) -> int:
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

        # Rufen Sie die Lambda-Funktion auf
        response = self.lambda_client.invoke(
            FunctionName='digit-ocr-v2',
            Payload=json.dumps(payload)
        )
        print("lambda response")

        # Lesen Sie den Rückgabewert aus der Antwort des Lambda-Clients
        payload_str = response['Payload'].read().decode('utf-8')
        payload_dict = json.loads(payload_str)
        payload_dict = json.loads(payload_dict['body'])
        print(payload_dict)
        try:
            predicted_digit = payload_dict.get('predicted_digit',0)
        except:
            print(payload_dict)
            predicted_digit = 0
        
        return predicted_digit 