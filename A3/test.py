f1 = 'transformer_data/train.en'
f2 = 'transformer_data/train.fr'

with open(f1, 'r') as f:
    print(len(f.readlines()))
    
with open(f2, 'r') as f:
    print(len(f.readlines()))