from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename
from src.predict import predict_currency

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Only JPG, JPEG, PNG are allowed.")
    
    # Get currency selection
    currency_code = request.form.get('currency', 'PKR')
    country_name = request.form.get('country', 'Pakistan')
    
    # Currency mapping for display
    currency_mapping = {
        'PKR': {'name': 'Pakistani Rupee', 'flag': '🇵🇰'},
        'INR': {'name': 'Indian Rupee', 'flag': '🇮🇳'},
        'USD': {'name': 'US Dollar', 'flag': '🇺🇸'},
        'GBP': {'name': 'British Pound', 'flag': '🇬🇧'},
        'SAR': {'name': 'Saudi Riyal', 'flag': '🇸🇦'},
        'AED': {'name': 'UAE Dirham', 'flag': '🇦🇪'},
        'CNY': {'name': 'Chinese Yuan', 'flag': '🇨🇳'},
        'EUR': {'name': 'Euro', 'flag': '🇪🇺'}
    }
    
    currency_info = currency_mapping.get(currency_code, currency_mapping['PKR'])
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + '_' + filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save file
    file.save(file_path)
    
    try:
        # Get prediction
        result = predict_currency(file_path)
        
        # Parse result
        parts = result.split()
        label = parts[0]
        confidence = float(parts[2].rstrip('%'))
        
        return render_template('result.html', 
                             prediction=label, 
                             confidence=confidence, 
                             image_path=unique_filename,
                             currency_code=currency_code,
                             currency_name=currency_info['name'],
                             currency_flag=currency_info['flag'])
    
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.unlink(file_path)
        return render_template('index.html', error=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)