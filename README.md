# WordDiscoveryNLP

这里是一个基于统计方法的新词发现项目。它主要通过对大规模语料进行处理，计算词的词频、左右邻居的熵值以及点互信息（PMI），从而识别出新的中文词汇。该项目使用Python编写，并依赖jieba进行中文分词。

## 数学原理

1. **点互信息 (PMI)**：
   点互信息用于衡量两个词同时出现的概率与它们各自独立出现的概率之间的关系，公式如下：

   \[
   PMI(word1, word2) = \log\frac{P(word1, word2)}{P(word1) \times P(word2)}
   \]

   其中，\( P(word1, word2) \) 是两个词同时出现的概率，\( P(word1) \) 和 \( P(word2) \) 分别是两个词各自出现的概率。

2. **熵 (Entropy)**：
   熵用于衡量某个词左右邻居的分布情况，公式如下：

   \[
   H(X) = -\sum_{i} P(x_i) \log P(x_i)
   \]

   其中，\( P(x_i) \) 是某个词左右邻居 \( x_i \) 出现的概率。

3. **综合得分**：
   通过结合PMI和左右熵，计算候选词对的综合得分，公式如下：

   \[
   score(word1, word2) = PMI(word1, word2) + \min(H(left_neighbors(word2)), H(right_neighbors(word1)))
   \]

## 示例代码

以下是完整的代码实现和示例用法：

```python
import os
from WordDiscovery import WordDiscoveryNLP, load_vocabulary, load_txt, load_stopwords, DATA_DIR

if __name__ == '__main__':
    corpus = load_txt(os.path.join(DATA_DIR, "demo.txt"), transform=lambda t: t.strip())

    model = WordDiscoveryNLP()
    model.add_vocabulary_dict()
    model.load_stopwords()

    for text in corpus:
        model.add_text(text)

    print(model.score())
```

## 结果分析

通过运行上述代码，可以得到文本中二元组（pair）的得分，并按照得分从高到低进行排序。得分越高，表示该pair的结合越紧密，越有可能构成一个新的词汇。

### 示例输出

运行上述代码后的输出结果如下：

```
OrderedDict([
(('陈时', '中'), 28.53579180782953), 
(('世界卫生', '大会'), 11.216921802538403), 
(('九二', '共识'), 10.722180793102735), 
(('世卫', '大会'), 8.31635906036165), 
(('民进党', '当局'), 7.793129155791486), 
... ])
```

### 导出新词

通过调用 `export_new_words_to_file` 方法，可以将得分结果导出到指定的文件中，方便后续分析和处理。

```
model.export_new_words_to_file('new_words.txt')
```

通过这些得分结果，用户可以进一步分析并确定哪些pair可以聚合成新的词汇。