import os.path

from src import config
from ydata_profiling import  ProfileReport
import pandas as pd

def generate_classifier_report():
    report_dir = config.eda_report_dir
    file_name = config.file_name_report

    # kiểm tra thư mục
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    df = pd.read_csv(config.raw_data_path)

    # tạo report
    profile = ProfileReport(df,title=file_name, explorative=True)

    # đường dẫn report đây đủ
    report_path = os.path.join(report_dir,file_name)

    # ghi đè file nếu đã tồn tại
    profile.to_file(report_path)

    print(f" Report created at: {report_path}")


if __name__ == '__main__':
    generate_classifier_report()