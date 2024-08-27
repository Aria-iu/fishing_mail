# main.py
import sys

def main():
    # 获取传入的文件路径
    file_path = sys.argv[1]

    # 打开并读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()
        print(f"Received file content:\n{content}")

if __name__ == "__main__":
    main()
