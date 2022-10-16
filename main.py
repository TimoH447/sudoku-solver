from io import TextIOWrapper
import base64
from flask import Flask
from flask import request
from flask import render_template

from src.frontend import convert_to_frontend
from src.image_reading import main

app = Flask(__name__)

@app.route("/")
def index():
    img = main()
    ergebnis = convert_to_frontend(img)
    return "<h1> Hello World! </h1> <br> <hr>" 

@app.route("/1")
def page_1(name= "Timo"):
    return render_template('example.html',name = name)

@app.route("/2", methods=['GET','POST'])
def page_2():
    if request.method == 'POST':
        img = request.files['img']
        img.save('assets/upload.jpg')
        main()
        return "Sucess" 
    
    return render_template('capture_img.html')


if __name__=="__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
