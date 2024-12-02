# 小说词云分析流程

## 操作流程

[小说词云分析流程](https://s0wpguaqyby.feishu.cn/wiki/BoMUwrIPWiOTKtkr2T4cG1YUn0f?from=from_copylink)

## 文件夹结构
```
novel_fin_analysis/               # 三体全集
│
├── Three_Body1/               # 三体第一部
│   ├── -stopwords.txt         # 停用词表
│   ├── -threebody.txt         # 小说文本
│   ├── -chinese_segmentation.py  # 分词脚本
│   ├── -segmented_text1.txt   # 分词结果
│   ├── -SegmentedWordCount.java  # 词频统计MapReduce程序
│   ├── -wordcount_output      # 词频统计结果
│   ├── -wx1.png               # 自定义词云图模板
│   ├── -wx2.png               # 自定义词云图模板
│   ├── -wordcloud_gen1_bea.py  # 词云生成脚本
│   ├── -threebody_wordcloud1.png  # 词云图
│   ├── -topic_modeling.py     # LDA主题分析脚本
│   ├── -topic_analysis_results.txt  # 主题分析结果
│   └── -lda_visualization.html  # 主题分析可视化结果
│
├── Three_Body2/               # 三体第二部
├── Three_Body3/               # 三体第三部
├── LDAOptimalParamsFinder.py   # LDA参数优化脚本
│
└── 各文件夹下有参数优化结果图：
    ├── coherence_plot.png
    ├── perplexity_plot.png
    └── grid_search_results.png
```
