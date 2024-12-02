# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from gensim import corpora, models
from collections import defaultdict
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

class TopicAnalyzer:
    def __init__(self, segmented_file, num_topics=5):
        self.num_topics = num_topics
        self.segmented_file = segmented_file
        
    def load_data(self):
        """加载分词后的文本数据"""
        with open(self.segmented_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        # 将文本分成段落（每50个词一段）
        self.documents = []
        current_doc = []
        for word in texts:
            word = word.strip()
            if word:
                current_doc.append(word)
                if len(current_doc) >= 50:
                    self.documents.append(current_doc)
                    current_doc = []
        if current_doc:
            self.documents.append(current_doc)
            
    def build_model(self):
        """构建并训练LDA模型"""
        # 构建词典
        self.dictionary = corpora.Dictionary(self.documents)
        
        # 计算词频
        corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        
        # 训练LDA模型
        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # 保存可视化结果
        vis_data = pyLDAvis.gensim_models.prepare(
            self.lda_model, corpus, self.dictionary)
        pyLDAvis.save_html(vis_data, 'lda_visualization.html')
        
    def print_topics(self):
        """打印主题分析结果"""
        print("\n=== 主题分析结果 ===")
        for idx, topic in self.lda_model.print_topics(-1):
            print(f'\n主题 {idx + 1}:')
            # 解析主题词和权重
            terms = [(term.split('*')[1].replace('"', '').strip(), 
                     float(term.split('*')[0].strip()))
                    for term in topic.split(' + ')]
            
            # 按权重排序并格式化输出
            sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
            for term, weight in sorted_terms:
                print(f'  {term}: {weight:.3f}')
                
    def save_results(self, output_file):
        """保存分析结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("《三体》主题分析结果\n")
            f.write("=" * 50 + "\n\n")
            
            for idx, topic in self.lda_model.print_topics(-1):
                f.write(f'主题 {idx + 1}:\n')
                terms = [(term.split('*')[1].replace('"', '').strip(), 
                         float(term.split('*')[0].strip()))
                        for term in topic.split(' + ')]
                
                sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
                for term, weight in sorted_terms:
                    f.write(f'  {term}: {weight:.3f}\n')
                f.write("\n")

def main():
    # 创建主题分析器
    analyzer = TopicAnalyzer('segmented_text.txt', num_topics=4)
    
    # 加载数据
    print("加载分词数据...")
    analyzer.load_data()
    
    # 构建模型
    print("构建LDA模型...")
    analyzer.build_model()
    
    # 打印主题
    analyzer.print_topics()
    
    # 保存结果
    analyzer.save_results('topic_analysis_results.txt')
    print("\n分析结果已保存到 topic_analysis_results.txt")
    print("交互式可视化结果已保存到 lda_visualization.html")

if __name__ == "__main__":
    main()