# SCEPI terminology evidence

核验日期：2026-10-20。范围为 `manuscript.tex` 正文、图表和补充材料中的生物学、模型与统计术语，共 60 项。每项保留五篇独立论文的实际用例，共 300 条，涉及 120 篇不同论文。完整题名、DOI、来源链接、实际词形和出现位置保存在 [term_evidence.csv](term_evidence.csv)。同一论文的不同版本在同一术语下只计一次。

用例取自论文正文、图注、方法、补充材料或摘要。摘要用例明确标记为 `Abstract`。参考文献列表、致谢和含义不同的文字匹配不计入。单复数、连字符和通行同义表达的实际词形分别保留。用例证明术语的既有使用；本研究的定义、模型和实验结论由正文公式与实验记录给出。

`PDF page` 使用所读论文的文件页码。`Extracted-text paragraph` 使用 `pdftotext` 导出文本按空行划分的段落编号；可用记录中的实际词形在原始 PDF 内检索。

SCEPI 是作者确定的方法名称，展开为 **Single-mutant-Conditioned Epistasis Prediction**。SCEPI adaptation 表示正文定义的目标组合适应模型。S-all、Z、S128 和 C128 等符号表示信息条件或样本预算。S/T 和 I/R 表示表注定义的初始化或干预条件。表格中的消融名称对应明确操作。上述名称和变量与既有学术术语分别记录。上游模型名称沿用原始论文，并在正文给出来源。

| 编号 | 核验概念 | 独立论文数 |
|---|---|---:|
| T01 | Epistasis / epistatic interaction | 5 |
| T02 | Compensatory mutation / intragenic compensation | 5 |
| T03 | Single mutant | 5 |
| T04 | Double mutant | 5 |
| T05 | Wild type | 5 |
| T06 | Protein stability | 5 |
| T07 | Folding / unfolding free energy | 5 |
| T08 | Deep mutational scanning | 5 |
| T09 | Double-mutant cycle | 5 |
| T10 | Global epistasis | 5 |
| T11 | Pairwise interaction | 5 |
| T12 | Additive model / additive effect | 5 |
| T13 | Main effect | 5 |
| T14 | Genotype-phenotype map | 5 |
| T15 | Protein language model | 5 |
| T16 | Sequence representation / embedding | 5 |
| T17 | Transfer learning | 5 |
| T18 | Gaussian process | 5 |
| T19 | Ridge regression | 5 |
| T20 | Multilayer perceptron | 5 |
| T21 | One-hot encoding | 5 |
| T22 | Monotone / monotonic function | 5 |
| T23 | Huber loss | 5 |
| T24 | Negative log-likelihood | 5 |
| T25 | Interval-censored observation | 5 |
| T26 | Early stopping | 5 |
| T27 | Cross-validation | 5 |
| T28 | Sequence identity | 5 |
| T29 | Bootstrap | 5 |
| T30 | Confidence interval | 5 |
| T31 | Root mean squared error | 5 |
| T32 | Spearman correlation | 5 |
| T33 | Enrichment factor | 5 |
| T34 | Hypergeometric distribution | 5 |
| T35 | L-BFGS | 5 |
| T36 | AdamW | 5 |
| T37 | GELU | 5 |
| T38 | Softplus | 5 |
| T39 | Attention head | 5 |
| T40 | Rotary position encoding / embedding | 5 |
| T41 | Average product correction | 5 |
| T42 | Gaussian likelihood / Gaussian noise | 5 |
| T43 | Median absolute deviation | 5 |
| T44 | Kernel regression | 5 |
| T45 | Sigmoid function | 5 |
| T46 | Data augmentation | 5 |
| T47 | Siamese network | 5 |
| T48 | Mean / average pooling | 5 |
| T49 | Hyperparameter selection / tuning | 5 |
| T50 | Homology | 5 |
| T51 | Pretraining | 5 |
| T52 | Conditioning model predictions | 5 |
| T53 | Residual | 5 |
| T54 | Neural network | 5 |
| T55 | Hidden layer | 5 |
| T56 | cDNA display | 5 |
| T57 | Proteolysis | 5 |
| T58 | C-alpha distance | 5 |
| T59 | Protein contact prediction / contact map | 5 |
| T60 | Indicator variable | 5 |

`Double-mutant cycle` 对应野生型、两个单突变和相应双突变的四项比较。本文分别定义测量所得残差与网络四次求值的代数约束。`Enrichment factor` 对应固定预算下命中频率相对于随机选择期望的比值；本文采用等蛋白平均命中率之比。`Median absolute deviation` 的用例采用离散程度或稳健统计中的含义。`Indicator variable` 的用例采用条件或事件状态的数值编码。

写作参照采用 RECOMB 2023 HOGVAX。获奖身份、篇幅、段落结构、引用位置和模板依据保存在 [EXPERIMENTS.md](../EXPERIMENTS.md)。当前稿件使用用户指定的 LLNCS 排版，遵守 RECOMB 的十页正文、Letter 页面、10 磅普通字号和一英寸页边距要求。
