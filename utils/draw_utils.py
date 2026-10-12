import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# 绘制混淆矩阵
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    # plt.imshow(cm, cmap='autumn')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        print("Successfully Saved!")


# 绘制t-SNE可分性图
def get_color(labels):
    """获取标签对应的颜色，使用更丰富的颜色集"""
    # 使用更美观且区分度高的颜色
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
        '#98D8C8', '#F7DC6F', '#D7BDE2', '#5DADE2',
        '#F1948A', '#AED6F1', '#ABEBC6', '#F8C471'
    ]
    
    # 如果标签数量超过预定义颜色，则循环使用
    unique_labels = np.unique(labels)
    if len(unique_labels) > len(colors):
        # 生成更多区分度高的颜色
        additional_colors = [f'#{np.random.randint(0, 0xFFFFFF):06x}' for _ in range(len(unique_labels) - len(colors))]
        colors.extend(additional_colors)
    
    # 创建标签到颜色的映射
    label_color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    return [label_color_map[label] for label in labels]

def draw_SNE(data_feature, label, save_path, title="t-SNE Visualization", 
             figsize=(10, 8), point_size=30, alpha=0.7, show_legend=True):
    """
    绘制美化后的t-SNE图像
    
    参数:
        data_feature: 输入的特征数据
        label: 数据标签
        save_path: 图像保存路径
        title: 图像标题
        figsize: 图像大小
        point_size: 散点大小
        alpha: 点的透明度
        show_legend: 是否显示图例
    """
    # 执行t-SNE降维
    data_embedded = TSNE(
        n_components=2, 
        init="pca", 
        random_state=42,  # 添加随机种子确保结果可复现
        perplexity=30,    # 调整困惑度参数，影响聚类效果
        learning_rate=200 # 学习率参数
    ).fit_transform(data_feature)
    
    # 获取颜色
    colors = get_color(label)
    
    # 提取坐标
    x = data_embedded[:, 0]
    y = data_embedded[:, 1]
    
    # 设置中文字体，确保中文正常显示
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    # plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 获取唯一标签用于绘制图例
    unique_labels = np.unique(label)
    
    # 为每个类别绘制散点，便于添加图例
    for lab in unique_labels:
        mask = (label == lab)
        plt.scatter(
            x[mask], y[mask], 
            color=colors[np.where(mask)[0][0]],  # 获取该类别的颜色
            s=point_size, 
            alpha=alpha,
            # label=f'类别 {lab}' if show_legend else ""
        )
    
    # 添加标题和坐标轴标签
    # plt.title(title, fontsize=16, pad=20)
    # plt.xlabel('t-SNE 维度 1', fontsize=12, labelpad=10)
    # plt.ylabel('t-SNE 维度 2', fontsize=12, labelpad=10)
    
    # 添加网格线，使图像更易读
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    # if show_legend:
    #     plt.legend(title="类别", bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()  # 自动调整布局，防止元素重叠
    
    # 保存图像（先保存再显示，避免保存空白图像）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    plt.show()
    
    # 关闭图像，释放资源
    plt.close()
    
    print(f'图像已成功保存至: {save_path}')


