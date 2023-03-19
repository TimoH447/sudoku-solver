import io
from PIL import Image
import boto3

def upload_photo(image,image_name):
    buffer = io.BytesIO()
    image.save(buffer,format='png')
    buffer.seek(0)

    s3 = boto3.client('s3')
    s3.put_object(Bucket='sudoku-solver-bucket',Key=image_name,Body=buffer)