import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'images')

app = Flask(__name__, template_folder="static")

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        img_file = request.files.get('pic')
        file_name = img_file.filename
        # 文件名的安全转换
        filename = secure_filename(file_name)
        print(filename)
        # 保存文件
        img_file.save(os.path.join(UPLOAD_PATH, filename))
        return '上传文件成功！'


if __name__ == '__main__':
    app.run(debug=True)

