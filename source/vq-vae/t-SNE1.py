import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def extract_label_from_path(file_path):
    parts = file_path.split(os.sep)
    return parts[-2] 

def evaluate_codebook_quality(npz_path):
    # 1. 配置
    target_categories = {'dance', 'up_down', 'walk', 'jump', 'run'}
    RENAME_MAP = {
        'dance': 'Acrobatic',
        'up_down': 'Recovery',
        "walk": "Walking",
        "jump": "Jumping",
        "run": "Running"
    }
    
    # 2. 数据处理
    print(f"正在加载数据: {npz_path}")
    data = np.load(npz_path)
    codebook = data['codebook']
    
    embeddings = []
    labels = []
    file_keys = [k for k in data.keys() if k != 'codebook']
    
    for file_path in file_keys:
        raw_label = extract_label_from_path(file_path)
        if raw_label not in target_categories:
            continue
            
        indices = data[file_path] 
        vectors = codebook[indices] 
        
        # 特征提取：均值 + 最大值 拼接
        v_flat = vectors.reshape(-1, vectors.shape[-1])
        e_combined = np.concatenate([np.mean(v_flat, axis=0), np.max(v_flat, axis=0)])
        
        labels.append(RENAME_MAP.get(raw_label, raw_label))
        embeddings.append(e_combined)

    X = np.array(embeddings)
    X = StandardScaler().fit_transform(X) # 标准化提高分离度
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 3. t-SNE 计算
    print("正在执行 t-SNE 降维...")
    tsne = TSNE(
        n_components=2, 
        perplexity=40, 
        early_exaggeration=18, # 稍微加大夸张系数，推开簇群
        init='pca', 
        learning_rate='auto',
        random_state=42
    )
    X_embedded = tsne.fit_transform(X)

    # 4. 绘图
    plt.figure(figsize=(9, 8), dpi=300)
    # 使用更有质感的调色板
    palette = sns.color_palette("Set2", len(le.classes_))
    
    for i, class_name in enumerate(le.classes_):
        mask = (y == i)
        plt.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1], 
            label=class_name, 
            alpha=0.8, 
            s=70,           # 放大点的大小
            edgecolors='white', 
            linewidth=0.8,
            color=palette[i],
            zorder=3
        )
    
    # --- 标题与轴处理 ---
    plt.title("Latent Space Distribution of Motion Semantics", fontsize=20, pad=20, fontweight='bold')
    
    # 移除 z1, z2 标签
    plt.xlabel("")
    plt.ylabel("")
    
    # --- 图例优化：移至右上角内部 ---
    plt.legend(
        loc='upper left', 
        fontsize=14, 
        frameon=True, 
        framealpha=0.9, 
        edgecolor='gray',
        shadow=True,
        borderpad=0.3
    )
    
    # --- 细节微调 ---
    # 放大坐标轴刻度数字
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # 网格线淡化
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 移除多余边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 保存
    output_img = npz_path.replace(".npz", "_final_distribution.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    
    print(f"可视化完成！图片已保存至: {output_img}")
    plt.show()

if __name__ == "__main__":
    LOG_DIR = "/home/ubuntu/projects/hjj-robot_lab/source/vq-vae/logs/2026-02-18_14-06-53"
    NPZ_FILENAME = "codebook_interp.npz"
    PATH = os.path.join(LOG_DIR, NPZ_FILENAME)
    
    if os.path.exists(PATH):
        evaluate_codebook_quality(PATH)
    else:
        print(f"找不到文件: {PATH}")