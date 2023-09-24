import os
import numpy as np
from obspy import read
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.signal import find_peaks

import torch
torch.manual_seed(42)
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torch import nn, optim
import torch.nn.functional as F
from torchvision.utils import save_image

import config
import re
import pickle

from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

import pandas as pd
from time import time
import argparse
from glob import glob
import seaborn as sns

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,256,40)
                        nn.Conv2d(in_channels=1,
                               out_channels=8,
                               kernel_size=5,
                               stride=3,
                               padding=1),  # output shape (8,25,39)
                        nn.BatchNorm2d(8),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(  # input shape (8,24,20)
                        nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=5,
                               stride=3,
                               padding=1),  # output shape (16,11,9)
                        nn.BatchNorm2d(16),
                        nn.ReLU()
                        )

        self.fc1 = nn.Sequential(
                        nn.Linear(16*28*4, 128),
                        #nn.BatchNorm2d(1),
                        nn.ReLU()
                        )

        self.fc2 = nn.Sequential(
                        nn.Linear(128, 16),
                        #nn.BatchNorm2d(1),
                        nn.ReLU()
                        )


        self.fc3 = nn.Sequential(
                        nn.Linear(16, 128),
                        #nn.BatchNorm2d(128),
                        nn.ReLU()
                        )

        self.fc4 = nn.Sequential(
                        nn.Linear(128, 16*28*4),
                        nn.ReLU()
                        )

        self.conv3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=16,
                                out_channels=8,
                                kernel_size=5,
                                stride=3,
                                padding=1,
                                output_padding=(1,1)),
                        nn.BatchNorm2d(8),
                        nn.ReLU()
                        )
        self.conv4 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=8,
                                out_channels=1,
                                kernel_size=5,
                                stride=3,
                                padding=1,
                                output_padding=(1,1)),
                        nn.BatchNorm2d(1),
                        nn.ReLU()
                        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # encode
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = x.view(x.size()[0], 1, 1, -1)
        x = self.dropout(x)
        #print(x.size())
        x = self.fc1(x)
        x = self.dropout(x)
        encoded = self.fc2(x)
        #print(encoded.size())
        
        # decode
        x = self.dropout(encoded)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        #print(x.size())
        x = x.view(-1,16,28,4)
        x = self.conv3(x)
        #print(x.size())
        decoded = self.conv4(x)
        #print(decoded.size())

        return encoded, decoded

def build_datafile_name(dic):
    #input: {'sta': xxx, 'starttime': xxx, 'duration': xxx}
    #output: 
    sta = dic['sta']
    start = dic['starttime']
    duration = dic['duration']

    wav = '../events/waveforms/0000'+sta+'_'+start+'_'+duration+'s.sac'
    spec = '../events/spectrograms/data/0000'+sta+'_'+start+'_'+duration+'s.txt'
    return wav, spec

def single_sample_process(spec):
    """
    input a spectrogram with shape of (256,40) and use the trained Autoencoder to output the reconstructed spectrogram
    """
    spec = spec.reshape(-1,1,spec.shape[-2],spec.shape[-1])
    spec = torch.tensor(spec).float()
    #data_loader = DataLoader(dataset = TensorDataset(spec),
     #               batch_size = 1,
     #               )
    # Check if GPU is active
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Load the model
    #model = AE()
    model = torch.load('../out/AE.pt').to(device)

    model.eval()
    spec = spec.to(device)
    encoded, decoded = model(spec)
    encoded = encoded.cpu().detach().numpy()
    decoded = decoded.cpu().detach().numpy()
    decoded = decoded.reshape(decoded.shape[-2], decoded.shape[-1])
    return decoded

def Plot_examples(cata, fignum=5):
    """
    input: cata(list with length--cluster_num)
                cata[i]: (event_num, 10)
                columns: eventname_dic, feature1, feature2, feature3, feature4, cluster, distance to center, waveformfile,specfile
           fignum: how many cases needed
    """
    cluster = cata.iloc[0]['cluster']
    feature_dim = len(cata.columns)-7
    
    idx = np.linspace(0, len(cata)-1, fignum)
    idx = [int(i) for i in idx]
    #idx = np.arange(0,10,1)
    #print('idx', idx)
    nrows = len(idx); ncols = 2
    fig = plt.figure(figsize=(10,4*nrows), dpi=100)
    grid = plt.GridSpec(nrows, 7, hspace=0.2, wspace=0.1)
    #fig.subplots_adjust(right=0.94)
    for i in range(nrows):
        waveformfile = cata.iloc[idx[i]]['waveformfile']
        waveform = read(waveformfile)[0].data
        feature = cata.iloc[idx[i]][['feature'+str(k) for k in range(1,feature_dim+1)]].values

        specfile = cata.iloc[idx[i]]['specfile']
        zxx = np.loadtxt(specfile) # (50f,78t)
        freq_zxx = np.sum(zxx, axis=-1)
        #sta = event_name[i][0]['sta']; starttime = event_name[i][0]['starttime']
        peaks, _ = find_peaks(freq_zxx, prominence=2)
        peaks = [[config.f[peak], freq_zxx[peak]] for peak in peaks]
        peaks = sorted(peaks, key=lambda x: x[-1], reverse=True) # freq, amp

        ax2 = fig.add_subplot(grid[i,1:4])
        ax2.pcolormesh(np.array(config.t)-2, config.f, zxx, vmin=0, vmax=1, shading='auto', cmap='Reds')
#        ax2.pcolormesh(np.array(config.t)-2, config.f[:20], zxx[:20])
        ax2.tick_params(labelleft=False)
        ax2.set_yticks([])
        #ax2.set_yscale('log')

        ax1 = fig.add_subplot(grid[i,0], 
                              #sharey=ax2
                              )
        ax1.plot(freq_zxx, config.f)
        for j in range(len(peaks)):
            ax1.text(np.max(freq_zxx), peaks[j][0], str(round(peaks[j][0],2))+'Hz', color='k', fontsize=10)
        ax1.invert_xaxis()
        ax1.set_xticks([])
        ax1.set_yticks([1,50,100,150,200,250])
        #ax1.tick_params(labelleft=True)
        #ax1.set_yscale('log')
        ax1.set_ylabel('Frequency [Hz]', fontsize=10)
        
        zxx_cons = single_sample_process(zxx)
        ax3 = fig.add_subplot(grid[i,4:], 
                              sharey=ax2
                              )
        sc = ax3.pcolormesh(np.array(config.t)-2, config.f, zxx_cons,
                            vmin=0, vmax=1, cmap='Reds', shading='auto'
                            )
        ax3.tick_params(labelleft=False)
        #ax3.set_yscale('log')

        ax1.set_ylim(1, 250)
        ax2.set_ylim(1, 250)
        ax3.set_ylim(1, 250)

#        sc = ax3.pcolormesh(np.array(config.t)-2, config.f[:20], zxx_cons[:20])
        #cb = fig.colorbar(ax0, ax=ax[i][1])
        #cb.ax.tick_params(labelsize = 10)
        fig.colorbar(sc, ax=(ax2,ax3), fraction=0.05)
        #ax3.set_ylabel('Frequency [Hz]', fontsize=8)
        #ax3.text(1.75, 200, 'd=%.2f'%(float(cata.iloc[idx[i]]['distance'])), color='white', fontsize=13, ha='right', va='top')

        if i==nrows-1:
            ax2.set_xlabel('Time [sec]', fontsize=12)
            ax3.set_xlabel('Time [sec]', fontsize=12)
        elif i==0:
            ax2.set_title('STFT (original)', fontsize=10)
            ax3.set_title('STFT (reconstructed)', fontsize=10)
    plt.suptitle('Cases of cluster '+str(cluster+1), y=0.92, fontsize=15)
    #plt.subplots_adjust(wspace=0)
    plt.savefig('../out/examples/c%d.pdf'%(cluster+1), bbox_inches='tight')
    #plt.show()
    plt.close()



def Plot_one(cata):
    """
    plot the nearest case for every cluster
    """
    feature_dim = len(cata[0].columns)-7

    fig = plt.figure(figsize = (5*5, 4*10),
                 #constrained_layout=True,
                 dpi=100,
                )
    gs0 = gridspec.GridSpec(10, 5, figure=fig, hspace=0.3, wspace=0.3)

    vmin = np.min([np.min(cata[i].values[:,2:feature_dim+2]) for i in range(len(cata))])
    vmax = np.max([np.max(cata[i].values[:,2:feature_dim+2]) for i in range(len(cata))])
    
    for i in range(len(cata)):
        A = cata[i].iloc[0]
        waveformfile = A['waveformfile']
        waveform = read(waveformfile)[0].data
        
        feature = []
        for k in range(1,feature_dim+1):
            feature.append(A['feature'+str(k)])
        feature = np.array(feature).reshape(1,-1)
        specfile = A['specfile']
        zxx = np.loadtxt(specfile) # (50,40)
        #zxx_cons = single_sample_process(zxx)

        gs00 = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs0[i], hspace=0)
        ax1 = fig.add_subplot(gs00[0, 0])
        sns.heatmap(feature, cmap='inferno_r', annot=feature, annot_kws={'fontsize': 4}, cbar=False,
                    vmin=0, vmax=5)
        #ax1.set_xlim(0,20)
        ax1.set_title('Cluster '+str(i+1)+': '+str(len(cata[i])))
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(gs00[1, 0])        
        ax2.plot(waveform, linewidth=0.5)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim(0,len(waveform))

        ax3 = fig.add_subplot(gs00[2:, 0])
        ax3.pcolormesh(np.array(config.t)-2, config.f, zxx, vmin=0, vmax=1, cmap='Reds')
#        ax3.pcolormesh(np.array(config.t)-2, config.f[:20], zxx[:20],
#                       #vmin=0, vmax=1
#                       )
        #ax3.set_yscale('log')
        #ax3.set_ylim(min(config.f), 20)
        #ax3.set_yticks([1,5,10,15,20])
        ax3.set_ylim(min(config.f), 250)
        ax3.set_yticks([1,50,100,150,200,250])

        #cb = fig.colorbar(ax0, ax=ax[i][1])
        #cb.ax.tick_params(labelsize = 10)
        #fig.colorbar(sc, ax=(ax[i][0],ax[i][1]), fraction=0.05)
        ax3.set_ylabel('Frequency [Hz]', fontsize=10)
        #ax3.text(10, 230, 'd=%.2f'%(float(A['distance'])), color='white', fontsize=8, ha='right', va='top')
        ax3.set_xlabel('Time [sec]', fontsize=10)
    plt.suptitle('Nearest-center event of each cluster', fontsize=30)
    fig.subplots_adjust(top=0.95)
    plt.savefig('../out/Nearest_cases.pdf'%A['cluster'], bbox_inches='tight')
    #plt.show()
    plt.close()

def dist(v1, v2):
    return np.linalg.norm(v1-v2)

if __name__ == '__main__':
    n_clusters = 50
    
    f = open('../out/features_loss.dat', 'rb')
    catalog = pickle.load(f) # catalog: shape=(*,2+feature_dim)
                             # rows=(event_name_dic 1, feature *, reconstruction loss 1)
    feature_dim = catalog.shape[1]-2
    print('feature shape:', feature_dim)

    
    feature = catalog[:,1:-1] # (*,feature_dim)
    print('features:', feature.shape)
    event_name = np.array([build_datafile_name(dic) for dic in catalog[:,0]]).reshape(len(catalog),-1)
    print('got the features!')
    
    
    # GMM clustering
    print('GMM begins...')
    gmm = GMM(n_clusters, covariance_type='full',
              random_state=1,
              ).fit(feature)
    c_predict = gmm.predict(feature)
    #print('c_predict:', type(c_predict), c_predict.shape)

    r1 = pd.Series(c_predict).value_counts() # 各类别的数目
    r2 = pd.DataFrame(gmm.means_) # 聚类中心
    r3 = pd.DataFrame(gmm.weights_)
    #r4 = pd.DataFrame(gmm.convariances_)
    centers = [list(r2.iloc[i].values) for i in range(len(r2))]
    r = pd.concat([r2, r1, r3], axis = 1) # 横向连接，得到聚类中心对应的数目
    r.to_excel('../out/Cluster centers.xlsx')
    r = pd.concat([pd.DataFrame(catalog), pd.Series(c_predict, index = pd.DataFrame(catalog).index)], axis = 1)
    temp = ['eventname_dic']
    temp.extend(['feature'+str(k) for k in range(1,feature_dim+1)]) #!!!!!!!!!!!!
    temp.extend(['loss','cluster'])
    r.columns = temp

    # Silhouette_score
    #silsc = silhouette_score(feature, c_predict, sample_size=len(feature), metric='euclidean')

    # Calculate the distance to centroid
    distance = [dist(feature[i], centers[c_predict[i]]) for i in range(len(feature))]
    distance = np.array(distance).reshape(len(distance),1)
    #total_catalog = np.hstack((event_name, ori, feature, recon, c_predict.reshape(c_predict.shape[0],1), distance))
    total_catalog = np.hstack((distance, event_name))

    A = pd.DataFrame(total_catalog)
    A.columns = ['distance','waveformfile','specfile']
    A = pd.concat([r,A], axis = 1)
    A.to_excel('../out/Total_Clusters.xlsx')
    print('Clustering over!')
    
    A = pd.read_excel('../out/Total_Clusters.xlsx')
    all_feature = A.iloc[:,2:2+feature_dim].values
    
    cata = []
    for i in range(n_clusters):
        temp_df = A.loc[A['cluster'] == i]
        temp = ['feature'+str(k) for k in range(1,feature_dim+1)] + ['loss','distance']
        temp_df = temp_df.copy()
        temp_df[temp] = temp_df[temp].astype(float)

        temp_df = temp_df.sort_values(by='distance', ascending=True) # sort by distance increasingly
        cata.append(temp_df)
        print('Cluster',i+1,':',len(temp_df),'samples')
    
    # Plot examples
    print('Plot examples...')
    fignum = 5
    for i in range(len(cata)):
        Plot_examples(cata[i], fignum)
    
    # Plot airscape
    print('Plot airscape...')
    Plot_one(cata) # plot the nearest case for each cluster

