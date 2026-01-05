import os
import matplotlib.pyplot as plt
import re
import cv2
import numpy as np
from models import *

def built_model(args):
    model = Model_HAR(args)
    return model

def get_save_path(args):
    save_path = os.path.join(args.save_path, args.data, args.finetune_type)
    if args.shot is not None:
        save_path = os.path.join(save_path, args.shot)
    return save_path

def split_data(args):
    print("开始切割数据...")
    print(f"切割设置：{args.shot}")
    split_type = args.shot
    samples = int(''.join(re.findall(r'\d+', split_type)))
    train_data = np.load(f'data(har)/{args.data}/x_train.npy') # 你的训练集数据
    labels = np.load(f'data(har)/{args.data}/y_train.npy')  # 对应的类别标签

    x_test = np.load(f'data(har)/{args.data}/x_test.npy')
    y_test = np.load(f'data(har)/{args.data}/y_test.npy')

    num_classes = len(np.unique(labels))  # 获取类别数量

    new_data = []
    new_labels = []
    unselected_data = []
    unselected_labels = []


    seed = args.seed
    np.random.seed(seed)
    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]  # 找到当前类别的样本索引
        selected_indices = np.random.choice(class_indices, samples, replace=False)  # 从当前类别中随机选择十个样本
        for idx in class_indices:
            if idx in selected_indices:
                new_data.append(train_data[idx])  # 将选中的样本数据添加到新数据集中
                new_labels.append(labels[idx])  # 将对应的标签添加到新标签列表中
            else:
                unselected_data.append(train_data[idx])  # 将未被选中的样本数据添加到未被选中数据集中
                unselected_labels.append(labels[idx])  # 将对应的标签添加到未被选中标签列表中

    # 将新数据集和新标签转换为NumPy数组
    new_data = np.array(new_data)
    new_labels =np.array(new_labels)
    unselected_data = np.array(unselected_data)
    unselected_labels = np.array(unselected_labels)

    save_path = f'data(har)/{args.data}/{split_type}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'x_train.npy'), new_data)
    np.save(os.path.join(save_path, 'y_train.npy'), new_labels)

    np.save(os.path.join(save_path, 'x_valid.npy'), unselected_data)
    np.save(os.path.join(save_path, 'y_valid.npy'), unselected_labels)

    np.save(os.path.join(save_path, 'x_test.npy'), x_test)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)
    # 获取当前随机数生成器的状态
    # state = np.random.get_state()

    # # 打印随机种子
    # seed = state[1][0]
    with open(os.path.join(save_path, 'log.txt'), 'w') as file:
        file.write(f'当前数据集随机种子数为{seed}')



def get_sensor_figure(data, index):
    plt.figure(figsize=(5.12, 0.12), dpi=100)  # 设置图形大小
    for i in range(data.shape[1]):
        plt.plot(data[:, i], linewidth=2)# 绘制数据
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 加粗边框线宽
    plt.savefig(f'plot_figure/Sensor_figure/sensor{index}.png', dpi=100)


def image2numpy(image_path):
    mg = cv2.imread(image_path, 1)
    img = np.float32(mg) / 255
    return img

