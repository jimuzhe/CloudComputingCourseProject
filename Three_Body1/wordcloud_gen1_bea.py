# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud(input_file, output_image, mask_image=None):
    """
    使用自定义形状生成词云图
    """
    # 读取词频统计结果
    df = pd.read_csv(input_file, sep='\t', names=['word', 'frequency'])
    word_freq = dict(zip(df['word'], df['frequency']))

    # 加载形状模板
    mask = None
    if mask_image:
        mask = np.array(Image.open(mask_image).convert("L"))  # 转为灰度模式，确保模板形状正常
        mask[mask > 200] = 255  # 将背景调整为白色
        mask[mask <= 200] = 0   # 将前景调整为黑色

    # 创建词云对象
    wc = WordCloud(
        font_path="/usr/share/fonts/wryh/MSYHBD.TTC",  # 替换为系统中支持中文的字体路径
        background_color="white",
        max_words=300,
        scale=2,
        mask=mask,  # 图案形状
        contour_width=0,  # 轮廓宽度
        contour_color="black",  # 轮廓颜色
        max_font_size=70,
        random_state=42
    )

    # 生成词云
    wc.generate_from_frequencies(word_freq)

    # 保存为图片
    wc.to_file(output_image)
    print(f"词云图已生成: {output_image}")

    # 显示词云
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    create_wordcloud(
        input_file="wordcount_output/part-r-00000",  # MapReduce输出的词频文件
        output_image="threebody_wordcloud1.png",     # 输出的词云图片文件
        mask_image="wx1.png"  # 自定义形状模板
    )