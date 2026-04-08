from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
import pandas as pd
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
from src_csv import inference, config
from src_images import config as img_config
from src_images.model import DiabeticRetinopathy
from chat.rag_pipeline import get_answer

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Thay đổi thành key bảo mật

# Các model có sẵn
MODELS = ['logistic', 'random_forest', 'svm', 'xgboost']

# Thư mục upload
UPLOAD_FOLDER = os.path.join(config.base_dir, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_MODEL_CHECKPOINT = os.path.join(img_config.model_dir, 'best_cnn.pt')
if not os.path.exists(IMAGE_MODEL_CHECKPOINT):
    fallback_checkpoint = os.path.join(img_config.model_dir, 'last_cnn.pt')
    IMAGE_MODEL_CHECKPOINT = fallback_checkpoint if os.path.exists(fallback_checkpoint) else IMAGE_MODEL_CHECKPOINT


def load_image_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiabeticRetinopathy().to(device)
    if not os.path.exists(IMAGE_MODEL_CHECKPOINT):
        raise FileNotFoundError('Không tìm thấy file checkpoint cho model ảnh.')
    checkpoint = torch.load(IMAGE_MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, device


def predict_image(image_path):
    model, device = load_image_model()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Không thể đọc ảnh.')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_config.image_size, img_config.image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std
    image = torch.from_numpy(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    max_idx = int(np.argmax(probabilities))
    predicted_class = img_config.categorys[max_idx]
    confidence = float(probabilities[max_idx] * 100)
    proba_dict = {img_config.categorys[i]: float(probabilities[i] * 100) for i in range(len(probabilities))}
    return predicted_class, confidence, proba_dict


@app.route('/')
def home():
    return redirect(url_for('prediction'))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    current_model = session.get('current_model', 'logistic')
    prediction_result = None
    proba_dict = None
    input_data = None
    file_result_path = None
    show_download_button = False
    used_model = None
    model_change_message = None
    file_message = None

    if request.method == 'POST':
        # Change model
        if 'change_model' in request.form:
            selected_model = request.form.get('model')
            if selected_model in MODELS:
                session['current_model'] = selected_model
                current_model = selected_model
                model_change_message = f'Model đã được thay đổi thành: {current_model}'
            else:
                model_change_message = 'Model không hợp lệ'

        # Manual input prediction
        elif 'predict_manual' in request.form:
            try:
                input_data = {
                    'Pregnancies': float(request.form['Pregnancies']),
                    'Glucose': float(request.form['Glucose']),
                    'BloodPressure': float(request.form['BloodPressure']),
                    'SkinThickness': float(request.form['SkinThickness']),
                    'Insulin': float(request.form['Insulin']),
                    'BMI': float(request.form['BMI']),
                    'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
                    'Age': float(request.form['Age'])
                }
                prediction_result, proba_dict = inference.model_from_dic(input_data, current_model)
                used_model = current_model
            except Exception as e:
                flash(f'Lỗi: {str(e)}', 'error')

        # File upload prediction
        elif 'predict_file' in request.form:
            if 'file' not in request.files:
                file_message = 'Không có file nào được chọn'
            else:
                file = request.files['file']
                if file.filename == '':
                    file_message = 'Không có file nào được chọn'
                elif file and (file.filename.endswith('.csv') or file.filename.endswith(('.xlsx', '.xls'))):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    try:
                        result_df = inference.model_from_file(file_path, current_model)
                        result_filename = 'predictions_' + filename
                        file_result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                        result_df.to_csv(file_result_path, index=False)
                        show_download_button = True
                        used_model = current_model
                        file_message = 'Dự đoán hoàn thành! Nhấn nút Download để tải file kết quả.'
                    except Exception as e:
                        file_message = f'Lỗi xử lý file: {str(e)}'
                else:
                    file_message = 'Chỉ chấp nhận file CSV hoặc Excel'

    return render_template('prediction.html', 
                         models=MODELS, 
                         current_model=current_model,
                         prediction=prediction_result,
                         proba_dict=proba_dict,
                         input_data=input_data,
                         file_result_path=file_result_path,
                         show_download_button=show_download_button,
                         used_model=used_model,
                         model_change_message=model_change_message,
                         file_message=file_message,
                         form_data=request.form)

@app.route('/image-prediction', methods=['GET', 'POST'])
def image_prediction():
    image_result = None
    image_confidence = None
    image_proba_dict = None
    image_preview_path = None
    image_file_message = None

    if request.method == 'POST':
        if 'image_file' not in request.files:
            image_file_message = 'Không có ảnh được chọn'
        else:
            image_file = request.files['image_file']
            if image_file.filename == '':
                image_file_message = 'Không có ảnh được chọn'
            elif image_file and image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                image_filename = secure_filename(image_file.filename)
                image_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image_file.save(image_upload_path)
                try:
                    image_result, image_confidence, image_proba_dict = predict_image(image_upload_path)
                    image_preview_path = image_filename
                    image_file_message = 'Dự đoán ảnh hoàn tất!'
                except Exception as e:
                    image_file_message = f'Lỗi xử lý ảnh: {str(e)}'
            else:
                image_file_message = 'Chỉ chấp nhận ảnh PNG/JPG/JPEG và các định dạng ảnh hợp lệ'

    return render_template('image_prediction.html',
                           image_result=image_result,
                           image_confidence=image_confidence,
                           image_proba_dict=image_proba_dict,
                           image_preview_path=image_preview_path,
                           image_file_message=image_file_message)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        flash('File không tồn tại', 'error')
        return redirect(url_for('prediction'))


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    flash('Ảnh không tồn tại', 'error')
    return redirect(url_for('prediction'))

@app.route('/statistics')
def statistics():
    # Đọc các report từ thư mục result
    reports = {}
    for model in MODELS:
        report_path = os.path.join(config.dir_result, f'train_report_{model}.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                reports[model] = f.read()
        else:
            reports[model] = 'Report không có sẵn'
    return render_template('statistics.html', reports=reports)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    answer = None
    question = None
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            try:
                answer = get_answer(question)
            except Exception as e:
                answer = f'Lỗi: {str(e)}'
    return render_template('chatbot.html', question=question, answer=answer)

@app.route('/chatbot/api', methods=['POST'])
def chatbot_api():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return {'error': 'No question provided'}, 400
    try:
        answer = get_answer(question)
        return {'answer': answer}
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
