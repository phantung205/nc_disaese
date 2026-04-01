import os

# root path project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# path data
# ----------------------------
data_dir  = os.path.join(base_dir,"data")
# data raw
raw_data_path  = os.path.join(data_dir,"raw","diabetes.csv")
# path  data  preprocessing
processed_data_dir = os.path.join(data_dir,"processed")


# ---------------------------
# path reports
# ---------------------------
report_dir = os.path.join(base_dir,"reports")
# dir report data
eda_report_dir = os.path.join(report_dir,"eda")
# name file report
file_name_report  = "report_diabetes.html"
# dir result
dir_result = os.path.join(report_dir,"result")

# --------------------------
# required columns
# --------------------------
target_col = [
    "Outcome"]
numerical_col = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

# --------------------------
# test size và random-state
# -------------------------
test_size = 0.2
random_state = 42

# ------------------------
# dir model
# ------------------------
dir_model = os.path.join(base_dir,"model")
