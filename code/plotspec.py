import os
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_spec(t, f, waveform, amp_zxx, event):
    p = os.getcwd()
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(4,1, hspace=0)
    ax1 = plt.subplot(gs[0])
    ax1.plot(waveform, linewidth=0.3, color='k')
    ax1.set_xticks([]); #ax1.set_yticks([])
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlim(0, len(waveform))

    ax2 = plt.subplot(gs[1:])
    sc = ax2.pcolormesh(t, f, amp_zxx, cmap='OrRd', vmin=0, vmax=1)
    #ax2.set_yscale('log')
    #cb = fig.colorbar(sc, ax=ax2, fraction=.05)
    #cb.ax.tick_params(labelsize = 20)
    ax2.set_xlim(0,4)
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Frequency (Hz)')

    plt.suptitle(event)
    plt.savefig(p+'/../events/spectrograms/visualize/%s.png'%event, bbox_inches = 'tight', dpi=300)
    plt.close()

