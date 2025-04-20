
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from services import process_image_file

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    try:
       result = process_image_file(input_path, OUTPUT_FOLDER)
       return jsonify({
            'no_bg_image': f"/download/{os.path.basename(result['no_bg_path'])}",
            'color_bg_image': f"/download/{os.path.basename(result['color_bg_path'])}"
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Authenticate user here
        return redirect(url_for("home"))
    return render_template("/login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # Register user here
        return redirect(url_for("home"))
    return render_template("/signup.html")


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    result = process_image_file(input_path, OUTPUT_FOLDER)

    return render_template("index.html",
                           nobg_url='/' + result['no_bg_path'],
                           colorbg_url='/' + result['color_bg_path'])

if __name__ == '__main__':
    app.run(debug=True)