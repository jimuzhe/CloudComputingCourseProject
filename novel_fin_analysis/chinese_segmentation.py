# -*- coding: utf-8 -*-
import jieba
import re

def load_stopwords(filepath):
    """加载停用词表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}

def segment_text(input_file, output_file, stopwords_file):
    """
    对文本进行分词并保存结果
    参数:
        input_file: 输入的小说文本路径
        output_file: 分词结果输出路径
        stopwords_file: 停用词表路径
    """
    # 加载停用词
    stopwords = load_stopwords(stopwords_file)
    
    # 加载自定义词典（如果有）
    # jieba.load_userdict("user_dict.txt")
    
    # 读取小说文本
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 预处理：删除非中文字符
    content = re.sub(r'[^\u4e00-\u9fa5]', ' ', content)
    
    # 分词
    words = jieba.cut(content)
    
    # 过滤停用词和空白字符
    filtered_words = []
    for word in words:
        word = word.strip()
        if word and word not in stopwords and len(word) > 1:
            filtered_words.append(word)
    
    # 将分词结果写入文件，每个词占一行
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_words))
    
    print(f"分词完成！结果已保存到: {output_file}")

if __name__ == "__main__":
    segment_text(
        input_file="threebody.txt",          # 输入的小说文本
        output_file="segmented_text.txt",    # 分词结果输出
        stopwords_file="stopwords.txt"       # 你的自定义停用词表
    )