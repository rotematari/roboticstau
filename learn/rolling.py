import pandas as pd
import numpy as np

def flatten(self):

    df = np.array(self)

    print(df.flatten())


x = np.random.rand(5, 5)

df = pd.DataFrame(x)

y = df.rolling(window=2).apply(func=flatten(), raw=True)