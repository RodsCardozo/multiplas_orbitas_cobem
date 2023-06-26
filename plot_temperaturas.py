
# Plot dos resultados
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
nome_pasta = input(f'Insira o nome da pasta: ')
nome_arquivo = input(f'Insira o nome do arquivo: ')
df_final = pd.read_csv(f'{nome_pasta}/{nome_arquivo}')
# Defina o tamanho do gráfico

# Plotando os dados
# using the variable ax for single a Axes

s = np.array(df_final['substrato 1'])

t = np.linspace(0,len(s), len(s))
# using the variable axs for multiple Axes
'''ax1 = (df_final['substrato 1'].to_list)
ax2 = (df_final['substrato 3'].to_list)
ax3 = (df_final['substrato 5'].to_list)
ax4 = (df_final['substrato 7'].to_list)
ax5 = (df_final['Tampa 1'])
ax6 = (df_final['Tampa 2'])'''
ax3 = (df_final['PCB 1'])
ax4 = (df_final['PCB 2'])
ax5 = (df_final['PCB 3'])
ax6 = (df_final['PCB 4'])
fig, axs = plt.subplots(2,2)

fig.suptitle('Temperaturas')
axs[0,0].plot(t, ax3, 'tab:red')
axs[0,0].set_title('PCB 1')
axs[1,0].plot(t, ax4, 'tab:green')
axs[1,0].set_title('PCB 2')
axs[0,1].plot(t, ax5, 'tab:blue')
axs[0,1].set_title('PCB 3')
axs[1,1].plot(t, ax6, 'tab:orange')
axs[1,1].set_title('PCB 4')
plt.show()
'''plt.plot(t, df_final['substrato 1'], label='substrato 1 [i]')
plt.plot(t, df_final['substrato 3'], label='substrato 3 [j]')
plt.plot(t, df_final['substrato 5'], label='substrato 5 [-i]')
plt.plot(t, df_final['substrato 7'], label='substrato 7 [-j]')
plt.plot(t, df_final['Tampa 1'], label='Topo 1 [-k]')
plt.plot(t, df_final['Tampa 2'], label='Topo 2 [k]')'''

'''plt.plot(t, df_final['PCB 1'], label='PCB 1')
plt.plot(t, df_final['PCB 2'], label='PCB 2')
plt.plot(t, df_final['PCB 3'], label='PCB 3')
plt.plot(t, df_final['PCB 4'], label='PCB 4')'''


'''plt.plot(t, df_final['Face5'], label='Face5')
plt.plot(t, df_final['Face6'], label='Face6')'''




