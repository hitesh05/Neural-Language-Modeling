import matplotlib.pyplot as plt
import numpy as np

# epochs = np.arange(1, 21)
# loss = [5.1034,4.3069,3.9024,3.6026,3.3480,3.1165,2.8975,2.6895,2.4925,2.3002,2.1194,1.9516,1.7905,1.6450,1.5159,1.4006,1.2951,1.2018,1.1225,1.0505]
# # Plotting the loss values
# plt.plot(epochs, loss, marker='o', linestyle='-')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('lr=1e-4, bs=8, dropout=0.20')
# plt.grid(True)
# plt.savefig('loss_curve_lr=1e-4_bs=8_dropout=0.20.png')
# plt.show()

models = [1,2]
loss = [0.0877, 0.0776]

# Plotting the loss values
plt.plot(models, loss, marker='o', linestyle='-')
plt.xlabel('Model')
plt.ylabel('Bleu Score (test)')
plt.title('Bleu scores')
plt.grid(True)
plt.savefig('bleu_curve_layers.png')
plt.show()