# Sentiment Analysis for Reddit Comments using NLP

![main_screen](https://i.ibb.co/p2c68xL/Screenshot-1.png)


## Comparing the performance of all the models

| Model           | Accuracy  | Precision | Recall    | F1        |
|-----------------|-----------|-----------|-----------|-----------|
| Baseline        | 77.256985 | 0.803389  | 0.772570  | 0.686895  |
| Simple Dense    | 88.501501 | 0.881913  | 0.885015  | 0.879412  |
| LSTM            | 83.329485 | 0.846905  | 0.833295  | 0.838088  |
| GRU             | 84.737936 | 0.850582  | 0.847379  | 0.848798  |
| Bidirectional   | 85.800046 | 0.855731  | 0.858000  | 0.856707  |
| Conv1D          | 86.261833 | 0.856871  | 0.862618  | 0.856728  |
| Ensemble        | 87.116139 | 0.869451  | 0.871161  | 0.870194  |

- **Baseline**: Naive Bayes (Model 0)
- **Simple Dense**: Feed-forward neural network (Model 1)
- **LSTM**: Long Short-Term Memory model (Model 2)
- **GRU**: Gated Recurrent Unit model (Model 3)
- **Bidirectional**: Bidirectional-LSTM model (Model 4)
- **Conv1D**: 1D Convolutional Neural Network (Model 5)
- **Ensemble**: Combination of multiple models for improved accuracy (Model 0 + Model 1 + Model 5)
