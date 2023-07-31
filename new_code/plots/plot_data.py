import matplotlib.pyplot as plt
import pandas as pd 


# Load the data from the CSV file
data = pd.read_csv(r'new_code/data/data/30_Jul_2023_16_02.csv')

data = data.dropna()

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,5))
y1 = data.drop(['sesion_time_stamp'],axis=1)[['M1x', 'M1y', 'M1z', 'M2x', 'M2y', 'M2z', 'M3x', 'M3y', 'M3z',
       'M4x', 'M4y', 'M4z']]
y2 = data.drop(['sesion_time_stamp'],axis=1)[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11',
       'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21',
       'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
       'S32']]

ax1.plot(y1)

ax2.plot(y2)
ax1.legend()

plt.show() 