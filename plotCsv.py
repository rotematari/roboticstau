import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Title = 'state_9_t1'
fileName = Title + '.csv'


def animate(i):
    df = pd.read_csv(fileName)
    # print(df)
    x = df['time']
    y1 = df['S1']
    y2 = df['S2']
    #y3 = df["B3"]
    # y4 = df[20]
    plt.cla()
    plt.plot(x, y1, label='S1')
    plt.plot(x, y2, label='S2')
    #plt.plot(x, y3, label='B3')
    #  plt.plot(x, y4, label='B2')
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = animation.FuncAnimation(plt.gcf(), animate, interval=1)
plt.tight_layout()
plt.show()
