import pandas as pd
import matplotlib.pyplot as plt

Title = 'state_9_t1'
fileName = Title + '.csv'
df = pd.read_csv(fileName)
#df = pd.read_excel(fileName)
print(df)
# df = pd.read_csv("Palm Mapping_.CSV")

df.plot()
df.set_index('time')
plt.xlabel("time [s]", fontsize = 18)
plt.ylabel("sensor reading", fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(Title, fontsize = 18)
plt.legend()
# plt.legend(loc='center', bbox_to_anchor=(0.5, -0.11),
#            fancybox=True, ncol=5, fontsize = 12)
plt.show()

# x = df[T]
# y1 = df[P1]
# y2 = df[P2]
# y3 = df[P3]
# y4 = df[P4]
# y5 = df[P5]

# df.head()

# fig = px.line(df, x = 'T', y = 'P1', title='Palm Sensors')
# fig = px.line(df, x = 'T', y = 'P2', title='Palm Sensors')
# fig = px.line(df, x = 'T', y = 'P3', title='Palm Sensors')
# fig = px.line(df, x = 'T', y = 'P4', title='Palm Sensors')
# fig = px.line(df, x = 'T', y = 'P5', title='Palm Sensors')
# fig.show()