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
        """���طִʺ���ı�����"""
        print("��������...")
        with open(self.segmented_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        # ���ı��ֳɶ��䣨ÿ50����һ�Σ�
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
            
        print(f"������ {len(self.documents)} ���ĵ�")
            
    def prepare_corpus(self):
        """׼���ʵ�����Ͽ�"""
        print("׼�����Ͽ�...")
        self.dictionary = corpora.Dictionary(self.documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        
    def grid_search(self, topic_range=(2,15), passes_range=(5,30), topic_step=1, passes_step=5):
        """�����������Ų������"""
        print("\n��ʼ�����������Ų���...")
        results = []
        total_iterations = ((topic_range[1] - topic_range[0]) // topic_step) * \
                          ((passes_range[1] - passes_range[0]) // passes_step)
        current_iteration = 0
        
        for num_topics in range(topic_range[0], topic_range[1], topic_step):
            for passes in range(passes_range[0], passes_range[1], passes_step):
                current_iteration += 1
                progress = (current_iteration / total_iterations) * 100
                print(f"\r����: {progress:.1f}% - ���Բ���: topics={num_topics}, passes={passes}", end="")
                
                model = models.LdaModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    num_topics=num_topics,
                    passes=passes,
                    random_state=42
                )
                
                # ����coherence score
                coherence_model = CoherenceModel(
                    model=model,
                    texts=self.documents,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                
                # ����perplexity
                perplexity = model.log_perplexity(self.corpus)
                
                results.append({
                    'num_topics': num_topics,
                    'passes': passes,
                    'coherence': coherence_score,
                    'perplexity': perplexity
                })
        
        print("\n�����������")
        return results
    
    def plot_grid_search_results(self, results_df, save_path='grid_search_results.png'):
        """�������������������ͼ"""
        print("���������������...")
        
        # ��������ͼ����
        pivot_coherence = results_df.pivot(
            index='passes', 
            columns='num_topics', 
            values='score'
        )
        
        # ����ͼ��
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_coherence, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Combined Score'})
        font_path = '/usr/share/fonts/wryh/MSYHBD.TTC'  # ȷ��·�����ļ�������ȷ��
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.title('���������������',fontproperties=font_prop)
        plt.xlabel('��������',fontproperties=font_prop)
        plt.ylabel('��������',fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def build_model(self, num_topics, passes):
        """������ѵ��LDAģ��"""
        print(f"\nʹ�����Ų�������ģ�� (topics={num_topics}, passes={passes})...")
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
        
        # ������ӻ����
        print("���ɽ���ʽ���ӻ�...")
        vis_data = pyLDAvis.gensim_models.prepare(
            self.lda_model, self.corpus, self.dictionary)
        pyLDAvis.save_html(vis_data, 'lda_visualization.html')
        
    def print_topics(self):
        """��ӡ����������"""
        print("\n=== ���������� ===")
        for idx, topic in self.lda_model.print_topics(-1):
            print(f'\n���� {idx + 1}:')
            terms = [(term.split('*')[1].replace('"', '').strip(), 
                     float(term.split('*')[0].strip()))
                    for term in topic.split(' + ')]
            
            sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
            for term, weight in sorted_terms:
                print(f'  {term}: {weight:.3f}')
                
    def save_results(self, output_file):
        """�������������ļ�"""
        print(f"\n������������ {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("�����塷����������\n")
            f.write("=" * 50 + "\n\n")
            
            for idx, topic in self.lda_model.print_topics(-1):
                f.write(f'���� {idx + 1}:\n')
                terms = [(term.split('*')[1].replace('"', '').strip(), 
                         float(term.split('*')[0].strip()))
                        for term in topic.split(' + ')]
                
                sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
                for term, weight in sorted_terms:
                    f.write(f'  {term}: {weight:.3f}\n')
                f.write("\n")

def main():
    # �������������ʵ��
    analyzer = TopicAnalyzer('segmented_text2.txt')
    
    # �������ݺ�׼�����Ͽ�
    analyzer.load_data()
    analyzer.prepare_corpus()
    
    # ������������
    results = analyzer.grid_search(
        topic_range=(2, 15),  # ��������Χ
        passes_range=(5, 31), # ����������Χ
        topic_step=1,         # ����������
        passes_step=5         # ������������
    )
    
    # ת�����ΪDataFrame�������ۺϵ÷�
    df_results = pd.DataFrame(results)
    df_results['coherence_norm'] = (df_results['coherence'] - df_results['coherence'].min()) / \
                                 (df_results['coherence'].max() - df_results['coherence'].min())
    df_results['perplexity_norm'] = (df_results['perplexity'].max() - df_results['perplexity']) / \
                                   (df_results['perplexity'].max() - df_results['perplexity'].min())
    df_results['score'] = df_results['coherence_norm'] + df_results['perplexity_norm']
    
    # ���������������
    analyzer.plot_grid_search_results(df_results)
    
    # �ҳ����Ų������
    best_params = df_results.loc[df_results['score'].idxmax()]
    optimal_topics = int(best_params['num_topics'])
    optimal_passes = int(best_params['passes'])
    print(f"\n���Ų������:")
    print(f"������: {optimal_topics}")
    print(f"��������: {optimal_passes}")
    print(f"Coherence Score: {best_params['coherence']:.4f}")
    print(f"Perplexity: {best_params['perplexity']:.4f}")
    
    # ʹ�����Ų�����������ģ��
    analyzer.build_model(optimal_topics, optimal_passes)
    
    # ��ӡ����
    analyzer.print_topics()
    
    # ������
    analyzer.save_results('topic_analysis_results.txt')
    
    print("\n�������! ����ļ�:")
    print("- topic_analysis_results.txt (����������)")
    print("- lda_visualization.html (����ʽ���ӻ�)")
    print("- grid_search_results.png (���������������ͼ)")

if __name__ == "__main__":
    main()