from sklearn.model_selection import train_test_split
import pandas as pd
from src import config
import os


def load_data():
    return pd.read_csv(config.raw_data_path)

def  clean_raw_data(df, is_train=True):
    df = df.copy()

    #  xóa các cột có dữ liệu trung lặp
    df = df.drop_duplicates()

    # xóa các cột ko cẩn thiết
    if "..." in df.columns:
        df = df.drop("...", axis=1)

    # 3. Lọc age không hợp lệ
    if "age" in df.columns:
        df = df[df["age"] > 0]

    # dữ lại các cột cần
    if is_train:
        required_cols = (
            config.numerical_col +
            config.target_col
        )
        df = df[required_cols]
    else :
        df = df[config.numerical_col]

    return df

def preprocess_and_split(test_size=None,random_state=None):
    processed_dir = config.processed_data_dir

    if test_size is None:
        test_size = config.test_size
    if random_state is None:
        random_state = config.random_state

    # load data
    df = load_data()

    # clear data
    df = clean_raw_data(df,True)

    # split targit , sample
    x = df.drop(config.target_col,axis=1)
    y = df[config.target_col]

    # train , test split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=random_state,stratify=y)

    os.makedirs(processed_dir, exist_ok=True)
    x_train.to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_and_split()
    print(x_train.head(2))
    print(x_test.head(2))
    print(y_train.head(2))
    print(y_test.head(2))