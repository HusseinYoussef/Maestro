from tqdm import tqdm
import numpy as np
x = 15
y = 0
for i in tqdm(np.arange(x), total= x):
    y += 1