import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def electrical_disp(sim_dir):
    ''''''
    
    I_df = pd.read_csv(sim_dir+'I_df.csv')
    V_df = pd.read_csv(sim_dir+'V_df.csv')
    M_df = pd.read_csv(sim_dir+'M_df.csv')
    
    plt.rcParams['axes.grid'] = True
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,8))
    
    I_df.drop('Unnamed: 0', axis=1).plot(ax=axes[0,0], title='Current Time Series', legend=False)
    V_df.drop('Unnamed: 0', axis=1).plot(ax=axes[0,1], title='Voltage Time Series', legend=False)
    M_df.drop('Unnamed: 0', axis=1).plot(ax=axes[1,0], title='Memristance Time Series', legend=False)
    
    for n, edge in enumerate(I_df.columns[1:]):
        edge_tup = literal_eval(edge)
        V_a = V_df[str(edge_tup[0])]
        V_b = V_df[str(edge_tup[1])]
        V = V_a - V_b 
        axes[1,1].plot(V, I_df[edge])
    axes[1,1].set_title('Current-Voltage Curve')
    axes[1,1].set_xlabel('Voltage')
    axes[1,1].set_ylabel('Current')
        
    plt.show()
    