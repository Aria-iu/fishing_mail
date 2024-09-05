import sys
import os

# 添加 code 目录到模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

# 导入 TrainModel 模块
import TrainModel

def main():
    # 获取传入的文件路径
    if len(sys.argv) < 2:
        print("请提供要检测的文件路径")
        sys.exit(1)
    file_path = sys.argv[1]
    print("邮件路径：", file_path)

    # 将内容作为列表传递给 check 函数进行检测
    predictions = TrainModel.recognize([file_path])
    print("检测完成")

    # 输出结果
    print("检测结果:", predictions)

if __name__ == "__main__":
    main()
