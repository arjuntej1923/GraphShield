# GraphShield

## Graph Neural Networks for Illicit Bitcoin Transaction Detection

GraphShield is a research-oriented framework for studying illicit transaction detection in cryptocurrency transaction networks.

The project investigates whether relational information represented by transaction graphs provides useful information beyond conventional transaction-level machine learning features.

Rather than assuming that graph neural networks are automatically superior, GraphShield performs a systematic comparison between traditional machine learning and graph neural networks using temporal evaluation, graph-specific ablations, error analysis, computational analysis, and multi-seed robustness experiments.

---

# 1. Research Question

The central research question is:

> Does incorporating graph structure improve illicit Bitcoin transaction detection compared with treating transactions primarily as independent feature vectors?

GraphShield compares:

### Traditional Machine Learning

- Logistic Regression
- Random Forest
- XGBoost

### Graph Neural Networks

- Graph Convolutional Network (GCN)
- GraphSAGE
- Graph Attention Network (GAT)

---

# 2. Dataset

GraphShield uses the Elliptic Bitcoin transaction dataset.

Dataset characteristics used in this study:

| Property | Value |
|---|---:|
| Total transaction nodes | 203,769 |
| Total graph edges | 234,355 |
| Time steps | 49 |
| Labeled transactions | 46,564 |
| Licit transactions | 42,019 |
| Illicit transactions | 4,545 |
| Illicit rate among labeled transactions | 9.76% |
| Transaction features | 165 |

The original labels are represented as:

```text
1 = Illicit
2 = Licit
Unknown = Unlabeled