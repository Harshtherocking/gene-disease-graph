# Link Prediction Models Performance Report

## Performance Metrics

| Model                   |    AUC |     AP |     F1 |   Precision |   Recall |
|:------------------------|-------:|-------:|-------:|------------:|---------:|
| GraphSAGE               | 0.9942 | 0.9919 | 0.9753 |      0.9676 |   0.9831 |
| GAT                     | 0.9936 | 0.9914 | 0.9739 |      0.9682 |   0.9797 |
| SEALLinkPrediction      | 0.9716 | 0.8389 | 0.8989 |      0.9315 |   0.9032 |
| GraphSAGELinkPrediction | 0.9575 | 0.8444 | 0.8835 |      0.9094 |   0.8159 |
| GCNLinkPrediction       | 0.9524 | 0.9084 | 0.9429 |      0.9604 |   0.9477 |
| GCN                     | 0.9419 | 0.9066 | 0.9155 |      0.8719 |   0.9636 |
| GATLinkPrediction       | 0.9101 | 0.8174 | 0.8838 |      0.8788 |   0.9104 |
| DeepWalk                | 0.8771 | 0.9525 | 0.8896 |      0.8429 |   0.815  |
| Node2Vec                | 0.8617 | 0.8693 | 0.9201 |      0.8922 |   0.8266 |
| HeuristicLinkPrediction | 0.8315 | 0.7901 | 0.7635 |      0.7236 |   0.7606 |
| MatrixFactorization     | 0.7583 | 0.7806 | 0.8145 |      0.8001 |   0.7487 |

## Interpretation

The best performing model is **GraphSAGE** with an AUC of 0.9942 and AP of 0.9919.

On average, GNN-based models achieve an AUC of 0.9602, which is 0.1280 higher than traditional methods.

Among embedding-based models, **DeepWalk** performs best with an AUC of 0.8771.

## Recommendations

Based on the performance metrics:

1. **GraphSAGE** should be the primary model for gene-disease link prediction tasks.
2. Consider ensemble approaches combining the top 2-3 models for potentially improved performance.
3. For large-scale applications where computational efficiency is important, evaluate the trade-off between model performance and training/inference time.
