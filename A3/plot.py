import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 21)
loss = [5.0031, 4.1747, 3.7498, 3.4149, 3.1083, 2.8159, 2.5354, 2.2650, 2.0104, 1.7747, 1.5639, 1.3755, 1.2172, 1.0826, 
        0.9711, 0.8803, 0.8051, 0.7463, 0.6935, 0.6528]

# Plotting the loss values
plt.plot(epochs, loss, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('lr=1e-4, bs=8, dropout=0.1')
plt.grid(True)
plt.savefig('loss_curve_lr=1e-4_bs=8_dropout=0.1.png')
plt.show()