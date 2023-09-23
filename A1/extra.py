# filename = '/home/hitesh/Documents/IIIT-H/Anlp/Assignments/A1/Neural-Language-Modeling/prp_files/2020115003-LM3-train-perplexity.txt'

# t = 0
# l = 0
# with open(filename, 'r') as f:
#     for line in f:
#         try:
#             x = float(line.strip().split()[-1])
#             t+=x
#             l+=1
#         except:
#             pass
        
# print(t/l)

# lines = []
# with open(filename, 'r') as f:
#     for line in f:
#         try:
#             line = line.strip().split()
#             x = float(line[-1])
#             x /= 10
#             line[-1] = x
#             line = ' '.join(map(str, line))
#             line+='\n'
#             lines.append(line)
#         except:
#             lines.append(line)

# with open(filename, 'w') as f:
#     f.writelines(lines)
            
import matplotlib.pyplot as plt

# Data
prp = [88.7909159356471078, 104.1018438496683145, 136.450727745468637]
names = ['transformer', 'lstm', 'nnlm']

# Create a bar graph with narrower bars
plt.bar(names, prp, color=['blue', 'green', 'red'], width=0.5)

# Add labels and title
plt.xlabel('Models')
plt.ylabel('PRP Values')
plt.title('PRP Values for Different Models')

# Show the graph
plt.savefig('graphs/comparison.png')
plt.show()          
            