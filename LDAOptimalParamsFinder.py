import pandas as pd
import numpy as np
from gensim import corpora, models
from collections import defaultdict
from matplotlib import font_manager
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import seaborn as sns

class TopicAnalyzer:
    def __init__(self, segmented_file):
        self.segmented_file = segmented_file
        self.documents = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        
    def load_data(self):
        """加载分词后的文本数据"""
        print("加载数据...")
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
            
        print(f"共加载 {len(self.documents)} 个文档")
            
    def prepare_corpus(self):
        """准备词典和语料库"""
        print("准备语料库...")
        self.dictionary = corpora.Dictionary(self.documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        
    def grid_search(self, topic_range=(2,15), passes_range=(5,30), topic_step=1, passes_step=5):
        """网格搜索最优参数组合"""
        print("\n开始网格搜索最优参数...")
        results = []
        total_iterations = ((topic_range[1] - topic_range[0]) // topic_step) * \
                          ((passes_range[1] - passes_range[0]) // passes_step)
        current_iteration = 0
        
        for num_topics in range(topic_range[0], topic_range[1], topic_step):
            for passes in range(passes_range[0], passes_range[1], passes_step):
                current_iteration += 1
                progress = (current_iteration / total_iterations) * 100
                print(f"\r进度: {progress:.1f}% - 测试参数: topics={num_topics}, passes={passes}", end="")
                
                model = models.LdaModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    num_topics=num_topics,
                    passes=passes,
                    random_state=42
                )
                
                # 计算coherence score
                coherence_model = CoherenceModel(
                    model=model,
                    texts=self.documents,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                
                # 计算perplexity
                perplexity = model.log_perplexity(self.corpus)
                
                results.append({
                    'num_topics': num_topics,
                    'passes': passes,
                    'coherence': coherence_score,
                    'perplexity': perplexity
                })
        
        print("\n网格搜索完成")
        return results
    
    def plot_grid_search_results(self, results_df, save_path='grid_search_results.png'):
        """绘制网格搜索结果热力图"""
        print("绘制网格搜索结果...")
        
        # 创建热力图数据
        pivot_coherence = results_df.pivot(
            index='passes', 
            columns='num_topics', 
            values='score'
        )
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_coherence, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Combined Score'})
        font_path = '/usr/share/fonts/wryh/MSYHBD.TTC'  # 确保路径和文件名是正确的
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.title('参数网格搜索结果',fontproperties=font_prop)
        plt.xlabel('主题数量',fontproperties=font_prop)
        plt.ylabel('迭代次数',fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def build_model(self, num_topics, passes):
        """构建并训练LDA模型"""
        print(f"\n使用最优参数构建模型 (topics={num_topics}, passes={passes})...")
        self.lda_model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=passes,
            alpha='auto',
            per_word_topics=True
        )
        
        # 保存可视化结果
        print("生成交互式可视化...")
        vis_data = pyLDAvis.gensim_models.prepare(
            self.lda_model, self.corpus, self.dictionary)
        pyLDAvis.save_html(vis_data, 'lda_visualization.html')
        
    def print_topics(self):
        """打印主题分析结果"""
        print("\n=== 主题分析结果 ===")
        for idx, topic in self.lda_model.print_topics(-1):
            print(f'\n主题 {idx + 1}:')
            terms = [(term.split('*')[1].replace('"', '').strip(), 
                     float(term.split('*')[0].strip()))
                    for term in topic.split(' + ')]
            
            sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
            for term, weight in sorted_terms:
                print(f'  {term}: {weight:.3f}')
                
    def save_results(self, output_file):
        """保存分析结果到文件"""
        print(f"\n保存分析结果到 {output_file}...")
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
    # 创建主题分析器实例
    analyzer = TopicAnalyzer('segmented_text2.txt')
    
    # 加载数据和准备语料库
    analyzer.load_data()
    analyzer.prepare_corpus()
    
    # 进行网格搜索
    results = analyzer.grid_search(
        topic_range=(2, 15),  # 主题数范围
        passes_range=(5, 31), # 迭代次数范围
        topic_step=1,         # 主题数步长
        passes_step=5         # 迭代次数步长
    )
    
    # 转换结果为DataFrame并计算综合得分
    df_results = pd.DataFrame(results)
    df_results['coherence_norm'] = (df_results['coherence'] - df_results['coherence'].min()) / \
                                 (df_results['coherence'].max() - df_results['coherence'].min())
    df_results['perplexity_norm'] = (df_results['perplexity'].max() - df_results['perplexity']) / \
                                   (df_results['perplexity'].max() - df_results['perplexity'].min())
    df_results['score'] = df_results['coherence_norm'] + df_results['perplexity_norm']
    
    # 绘制网格搜索结果
    analyzer.plot_grid_search_results(df_results)
    
    # 找出最优参数组合
    best_params = df_results.loc[df_results['score'].idxmax()]
    optimal_topics = int(best_params['num_topics'])
    optimal_passes = int(best_params['passes'])
    print(f"\n最优参数组合:")
    print(f"主题数: {optimal_topics}")
    print(f"迭代次数: {optimal_passes}")
    print(f"Coherence Score: {best_params['coherence']:.4f}")
    print(f"Perplexity: {best_params['perplexity']:.4f}")
    
    # 使用最优参数构建最终模型
    analyzer.build_model(optimal_topics, optimal_passes)
    
    # 打印主题
    analyzer.print_topics()
    
    # 保存结果
    analyzer.save_results('topic_analysis_results.txt')
    
    print("\n分析完成! 输出文件:")
    print("- topic_analysis_results.txt (主题分析结果)")
    print("- lda_visualization.html (交互式可视化)")
    print("- grid_search_results.png (参数搜索结果热力图)")

if __name__ == "__main__":
    main()