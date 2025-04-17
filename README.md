# Customer Feedback Analysis

![main_screen](https://i.ibb.co/wFCwXgfv/customer.png)


## Comparing the performance of all the models

| Model         |   Accuracy | Precision |  Recall  |    F1    |
|---------------|-----------:|----------:|---------:|---------:|
| Baseline      | 79.776763  | 0.761436  | 0.797768 | 0.723605 |
| Simple Dense  | 85.289073  | 0.833047  | 0.852891 | 0.834432 |
| LSTM          | 88.847842  | 0.878399  | 0.888478 | 0.881060 |
| GRU           | 88.495132  | 0.878578  | 0.884951 | 0.881209 |
| Bidirectional | 88.564618  | 0.878972  | 0.885646 | 0.881670 |
| Conv1D        | 88.087008  | 0.871726  | 0.880870 | 0.874662 |
| BERT          | 84.926687  | 0.879090  | 0.849267 | 0.860673 |

- **Baseline**: Naive Bayes (Model 0)
- **Simple Dense**: Feed-forward Neural Network (Model 1)
- **LSTM**: Long Short-Term Memory Model (Model 2)
- **GRU**: Gated Recurrent Unit Model (Model 3)
- **Bidirectional**: Bidirectional-LSTM Model (Model 4)
- **Conv1D**: 1D Convolutional Neural Network (Model 5)
- **BERT**: Pre-trained BERT (Model 6)
