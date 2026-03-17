import os
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def extract_label_from_path(file_path):
    """
    根据文件路径提取类别标签。
    假设路径格式为: /.../motion/category/file_name.npy
    """
    parts = file_path.split(os.sep)
    return parts[-2] 

def evaluate_codebook_quality(npz_path):
    # 1. 定义原始文件夹名称和映射关系
    # target_categories 必须对应你硬盘上真实的文件夹名
    target_categories = {'dance', 'up_down', 'walk', 'jump', 'run'}
    
    # 定义重命名映射
    RENAME_MAP = {
        'dance': 'Acrobatic',
        'up_down': 'Recovery',
        "walk": "Walking",
        "jump": "Jumping",
        "run": "Running"
    }
    
    # 2. 加载数据
    print(f"正在加载数据: {npz_path}")
    data = np.load(npz_path)
    
    if 'codebook' not in data:
        raise KeyError("npz 文件中未找到 'codebook' 键")
    codebook = data['codebook']
    K, C = codebook.shape
    print(f"码本规模: {K} 个向量, 每个维度为 {C}")

    embeddings = []
    labels = []
    
    file_keys = [k for k in data.keys() if k != 'codebook']
    
    print(f"开始筛选并重命名类别...")
    for file_path in file_keys:
        raw_label = extract_label_from_path(file_path)
        
        # 如果不在目标类别中，跳过
        if raw_label not in target_categories:
            continue
            
        indices = data[file_path] 
        vectors = codebook[indices]
        
        # 计算时间平均值
        e_sem = np.mean(vectors, axis=(0, 1)) 
        
        # 执行重命名逻辑：如果在映射表里就换名，否则保持原样
        display_label = RENAME_MAP.get(raw_label, raw_label)
        
        embeddings.append(e_sem)
        labels.append(display_label)

    if len(embeddings) == 0:
        print("错误: 筛选后没有剩余样本。")
        return

    X = np.array(embeddings)
    label_names = np.array(labels)
    
    # 标签编码 (此时编码的是重命名后的名称)
    le = LabelEncoder()
    y = le.fit_transform(label_names)

    print(f"\n成功提取 {len(X)} 个样本，涵盖类别: {list(le.classes_)}")

    # 3. 计算轮廓系数
    s_score = silhouette_score(X, y)
    print(f"\n[指标 1] 轮廓系数 (Silhouette Score): {s_score:.4f}")

    # 4. 线性可分性测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print("\n[指标 2] 分类性能报告 (KNN Classifier):")
    # 这里会自动显示 Acrobatic 和 recovery
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 5. 可视化 (t-SNE)
    print("\n[指标 3] 正在生成学术论文风格的可视化图 (PNG 格式)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 调整画布比例，使其在双栏论文中占满半行时更美观
    plt.figure(figsize=(7, 6)) 
    
    # 增加散点大小 (s=40) 并增加白边 (edgecolors='white')，提高区分度
    for i, class_name in enumerate(le.classes_):
        mask = (y == i)
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                    label=class_name, alpha=0.8, s=40, edgecolors='white', linewidth=0.5)
    
    # --- 标题与坐标轴优化 (字体加大) ---
    # 标题：更具学术感的表达
    plt.title("Latent Space Distribution of Motion Semantics", fontsize=18, pad=15)
    
    # 坐标轴：使用标准的数学表示 z1, z2
    plt.xlabel(r"$z_1$", fontsize=16)
    plt.ylabel(r"$z_2$", fontsize=16)
    
    # --- 细节美化 ---
    # 右上角图例：加大字号 (fontsize=14)，增加半透明背景 (framealpha=0.9)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
    
    # 刻度字号
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 只保留轻微的网格线
    plt.grid(True, linestyle=':', alpha=0.4)
    
    # 移除顶部和右侧的边框，使图表更开阔
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # --- 保存设置 ---
    # 1. 替换文件名，表明是最终论文图
    output_img = npz_path.replace(".npz", "_final_paper_fig.png")
    
    # 2. 关键：dpi=300 保证清晰度，bbox_inches='tight' 去除多余白边
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    
    print(f"适配论文的高分辨率 PNG 图已保存至: {output_img}")
    plt.show()
if __name__ == "__main__":
    LOG_DIR = "/home/ubuntu/projects/hjj-robot_lab/source/vq-vae/logs/2026-02-18_14-06-53"
    NPZ_FILENAME = "codebook_interp.npz"
    PATH = os.path.join(LOG_DIR, NPZ_FILENAME)
    
    if os.path.exists(PATH):
        evaluate_codebook_quality(PATH)
    else:
        print(f"错误: 找不到文件 {PATH}")