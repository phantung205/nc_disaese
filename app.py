from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
import pandas as pd
from werkzeug.utils import secure_filename
from src import inference, config

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Thay đổi thành key bảo mật

# Các model có sẵn
MODELS = ['logistic', 'random_forest', 'svm', 'xgboost']

# Thư mục upload
UPLOAD_FOLDER = os.path.join(config.base_dir, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        flash('File không tồn tại', 'error')
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

if __name__ == '__main__':
    app.run(debug=True)
