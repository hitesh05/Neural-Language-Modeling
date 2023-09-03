# Advanced NLP: Assignment 1
**Author:** *Hitesh Goel (20201115003)*

## Part 1: NNLM

### Declaring and Loading the model

To run the file:
```
python nnlm_main.py
```

To declare the model:
```
lm = NNLM(len(train_dataset.vocab), h1=500, h2=500).to(device)
```

To load the model:
```
lm = torch.load('nnlm.pth')
```

The model is available on OneDrive:
[link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/hitesh_goel_research_iiit_ac_in/ERJ8NB7UIyVFv6Y6Zfz5FL0B4cvFnXvdTaXGvMlLI2AkaQ?e=gRCqne)

### Perplexities
The perplexity files are uploaded on OneDrive:

[Test_file](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/hitesh_goel_research_iiit_ac_in/Ec2tO5VrxQpIlyIZKMFjPBsB1YpA3rpisaJiSb310jjv3A?e=uBaLJ7)

[Train_file](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/hitesh_goel_research_iiit_ac_in/EVqqL3tXFitJq5DtRoV67PQB6A0J1AyY0Sbiqo5qMF8bSQ?e=eT9mmC)

### Hyperparameter finetuning (bonus)
The model was tuned over learning rates and hidden sizes.

Outputs are contained in the file: *ffn.txt*

*Graphs and visualisations:*
![prp with lr-0.05](graphs/ffn_analysis/perp0.05.png)

![prp with lr-0.1](graphs/ffn_analysis/perp0.1.png)

![prp with lr-0.5](graphs/ffn_analysis/perp0.5.png)

*most optimal parameters found:* lr=0.1	hs=300

## Part 2: LSTM

### Declaring and Loading the model

To run the file:
```
python lstm_main.py
```

To declare the model:
```
    lm = My_LSTM(len(train_dataset.vocab), embedding_matrix=train_dataset.embeddings,h1=300).to(device)

```

To load the model:
```
lm = torch.load('lstm.pth')
```

The model is available on OneDrive:
[link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/hitesh_goel_research_iiit_ac_in/EYW9FpRvpkFCuRR1wsV8PM0BgwTSL9Dxp0WZxwe7JP0SDw?e=aSMK1I)

### Perplexities
The perplexity files are uploaded on OneDrive:

[Test_file](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/hitesh_goel_research_iiit_ac_in/EVqqL3tXFitJq5DtRoV67PQB6A0J1AyY0Sbiqo5qMF8bSQ?e=hPwOuT)

[Train_file](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/hitesh_goel_research_iiit_ac_in/Ed1-yMQBgklDgmFOw_j4Ln4BgCyzUDhUz0rrT3ScvMmYmw?e=DpHM8Y)


## Part 3: TRANSFORMER

### Declaring and Loading the model

To run the file:
```
python transformer_main.py
```

To declare the model:
```
lm = TransformerDecoder(len(train_dataset.vocab),embedding_dim=300,num_layers=6,num_heads=6,max_seq_length=train_dataset.max_len, embedding_matrix=train_dataset.embeddings).to(device)

```

To load the model:
```
lm = torch.load('transformer.pth')
```

The model is available on OneDrive:
[link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/hitesh_goel_research_iiit_ac_in/EZfSx0sItq5Gjw-Mm27_10sB022kLN9yQSQAzqWYMw4R4Q?e=MIxZ7H)

### Perplexities
The perplexity files are uploaded on OneDrive:

[Test_file](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/hitesh_goel_research_iiit_ac_in/EeKFUuaVAQpKofgHVBlEanEBR8LMgECRV8GMFkPH_RHmzg?e=HC6mLi)

[Train_file](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/hitesh_goel_research_iiit_ac_in/ESnab_3FrddFl-qpiVnao8sB4wjQUQc7FSFMYjjeS1zXFQ?e=Qfygyj)

### Analysis and Visualistions
As expected, LSTM outperforms a simple FFN model, but a transformer is superior to an LSTM model which is also indicated by the lower perplexity scores of the model. 

![comparison](graphs/comparison.png)