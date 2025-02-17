
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from mpl_toolkits.mplot3d import Axes3D

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rcParams.update({'font.size': 16})
#BINp, TimBinnedp, popRateG1p,popRateG2p,popRatePedp,LfrG1p,LfrG2p,LfrPedp = np.load('Send2/Sim_0_100Seiz_rates.npy',allow_pickle=True)

#BINnp, TimBinnednp, popRateG1np,popRateG2np,popRatePednp,LfrG1np,LfrG2np,LfrPednp = np.load('Send2/Sim_0_101Seiz_rates.npy',allow_pickle=True)



#np.save('Sim_0_'+str(NbSim)+'_sptr.npy',[sp_t1,sp_t2,sp_t3])


Sp_FS_Prop,Sp_RS_Prop,Sp_Pois_Prop=np.load('ResultsHH2/HHSpikeTrains_seed1_Amp_140.0_Tau_65.0.npy',allow_pickle=True)
#v_RS_Prop=np.load('Sim_0_100_v_RS.npy',allow_pickle=True)
#v_FS_Prop=np.load('Seed0/Sim_0_100_v_FS.npy',allow_pickle=True)
connexions_sing_Prop=np.load('ResultsHH2/HHconnect_seed1_Amp_140.0_Tau_65.0.npy',allow_pickle=True).item()

                
                
#Sp_FS_NoProp,Sp_RS_NoProp,Sp_Pois_NoProp=np.load('Sim_0_101_sptr.npy',allow_pickle=True)
#v_RS_NoProp=np.load('Sim_0_101_v_RS.npy',allow_pickle=True)
#v_FS_NoProp=np.load('Seed0/Sim_0_101_v_FS.npy',allow_pickle=True)
#connexions_sing_NoProp=np.load('Sim_0_101_connect.npy',allow_pickle=True).item()

def PrepScatSort(d):
    d_sorted={}
    i=0
    CorSpk={}
    for k in sorted(d, key=lambda k: len(d[k]), reverse=False):
        d_sorted[i]=d[k]
        CorSpk[k]= i
        i+=1

    x=[] 
    y=[]
    for k in range(len(d_sorted)):
        for j in d_sorted[k]:
            y.append(k)    
            x.append(j)
    return [x, y], CorSpk

def connectStat(S):
        #St=zip(S.i, S.j)
        result1 = {}
        result2 = {}
        for PreS, PostS in S:
            result1.setdefault(PreS, []).append(PostS)
            result2.setdefault(PostS, []).append(PreS)
        return result1, result2


def RasterConnectSort(Con, d):
    d_sorted={}
    i=0
    CorCon={}
    for k in sorted(Con, key=lambda k: len(Con[k]), reverse=False):
        d_sorted[i]=d[k]
        CorCon[k]= i
        i+=1

    x=[] 
    y=[]
    for k in range(len(d_sorted)):
        for j in d_sorted[k]:
            y.append(k)    
            x.append(j)
    return [x, y], CorCon


def simplePlot(d):
    x=[] 
    y=[]
    for k in range(len(d)):
        for j in d[k]:
            y.append(k)    
            x.append(j)
    return [x,y]

S_12_Prop=connexions_sing_Prop['S_12']
#S_12_NoProp=connexions_sing_NoProp['S_12']
Sent_Prop_12,Received_Prop_12=connectStat(S_12_Prop)
#Sent_NoProp_12,Received_NoProp_12=connectStat(S_12_NoProp)

S_11_Prop=connexions_sing_Prop['S_11']
#S_11_NoProp=connexions_sing_NoProp['S_11']
Sent_Prop_11,Received_Prop_11=connectStat(S_11_Prop)
#Sent_NoProp_11,Received_NoProp_11=connectStat(S_11_NoProp)


#print(Sent_Prop)

#RasRSnoPropC, CorConNoPropRS=RasterConnectSort(Received_NoProp_12,Sp_RS_NoProp)
#RasFSnoPropC, CorConNoPropFS=RasterConnectSort(Received_NoProp_11,Sp_FS_NoProp)
#RasFSnoPropC[1]=[l+8000 for l in RasFSnoPropC[1]]
#fig3=plt.figure(figsize=(8,4))
#fig3.suptitle('No Prop ConnectSorted (received Inh)', fontsize=12)
#ax3=fig3.add_subplot(111)
#ax3.plot(RasRSnoPropC[0],RasRSnoPropC[1], ',g')
#ax3.plot(RasFSnoPropC[0],RasFSnoPropC[1], ',r') 
#ax3.set_xlim(1.5,3.5)
#ax3.set_xlabel('Time (s)')
#ax3.set_ylabel('Neuron Index (sorted)')
RasRSPropC, CorConPropRS=RasterConnectSort(Received_Prop_12,Sp_RS_Prop)
RasFSPropC, CorConPropFS=RasterConnectSort(Received_Prop_11,Sp_FS_Prop)
RasFSPropC[1]=[l+8000 for l in RasFSPropC[1]]
fig4=plt.figure(figsize=(8,5))
#fig4.suptitle('Prop ConnectSorted (received Inh)', fontsize=12)
ax4=fig4.add_subplot(111)
ax4.plot(RasRSPropC[0],RasRSPropC[1], ',g')
ax4.plot(RasFSPropC[0],RasFSPropC[1], ',r') 
ax4.set_xlim(1.75,2.25)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Neuron Index (sorted)')
#print(Sp_RS_NoProp)

#RasRSnoProp, CorSpkNoPropRS=PrepScatSort(Sp_RS_NoProp)
#RasFSnoProp, CorSpkNoPropFS=PrepScatSort(Sp_FS_NoProp)
#RasFSnoProp[1]=[l+8000 for l in RasFSnoProp[1]]
#fig=plt.figure(figsize=(8,4))
#fig.suptitle('No Prop', fontsize=12)
#ax=fig.add_subplot(111)
#ax.plot(RasRSnoProp[0],RasRSnoProp[1], ',g')
#ax.plot(RasFSnoProp[0],RasFSnoProp[1], ',r') 
#ax.set_xlim(1.5,3.5)


RasRSProp, CorSpkPropRS=PrepScatSort(Sp_RS_Prop)
RasFSProp, CorSpkPropFS=PrepScatSort(Sp_FS_Prop)
RasFSProp[1]=[l+8000 for l in RasFSProp[1]]
fig2=plt.figure(figsize=(8,5))
#fig2.suptitle('Prop', fontsize=12)
ax2=fig2.add_subplot(111)
ax2.plot(RasRSProp[0],RasRSProp[1], ',g')
ax2.plot(RasFSProp[0],RasFSProp[1], ',r') 
ax2.set_xlim(1.75,2.25)
#np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Neuron Index (sorted)')
###################################################################


ConSpkPropFS=[[CorSpkPropFS[i]+8000, CorConPropFS[i]+8000] for i, value in CorSpkPropFS.items()]
ConSpkPropFS=np.array(ConSpkPropFS).T

ConSpkPropRS=[[CorSpkPropRS[i], CorConPropRS[i]] for i, value in CorSpkPropRS.items()]
ConSpkPropRS=np.array(ConSpkPropRS).T

fig5=plt.figure(figsize=(8,5))
fig5.suptitle('Prop Sp vs Con', fontsize=12)
ax5=fig5.add_subplot(111)

XFSp,YFSp = ConSpkPropFS
XRSp,YRSp = ConSpkPropRS
ax5.plot(XRSp,YRSp, '.g')
ax5.plot(XFSp,YFSp, '.r') 
ax5.set_xlabel('Con received from Inh')
ax5.set_ylabel('Nb spikes sorted')
#lims = [
#    np.min([ax5.get_xlim(), ax5.get_ylim()]),  # min of both axes
#    np.max([ax5.get_xlim(), ax5.get_ylim()]),  # max of both axes
#]

# now plot both limits against eachother
#ax5.plot(lims, lims, 'k-', alpha=0.5, zorder=0)


#ConSpkNoPropFS=[[CorSpkNoPropFS[i]+8000, CorConNoPropFS[i]+8000] for i, value in CorSpkNoPropFS.items()]
#ConSpkNoPropFS=np.array(ConSpkNoPropFS).T

#ConSpkNoPropRS=[[CorSpkNoPropRS[i], CorConNoPropRS[i]] for i, value in CorSpkNoPropRS.items()]
#ConSpkNoPropRS=np.array(ConSpkNoPropRS).T


#fig6=plt.figure(figsize=(8,4))
#fig6.suptitle('No Prop Sp vs Con', fontsize=12)
#ax6=fig6.add_subplot(111)
#XFSnp,YFSnp = ConSpkNoPropFS
#XRSnp,YRSnp = ConSpkNoPropRS
#ax6.plot(XRSnp,YRSnp, '.g')
#ax6.plot(XFSnp,YFSnp, '.r')
#ax6.set_xlabel('Con received from Inh')
#ax6.set_ylabel('Nb spikes sorted')


RasFSProp=simplePlot(Sp_FS_Prop)
RasFSProp[1]=[l+8000 for l in RasFSProp[1]]
RasRSProp=simplePlot(Sp_RS_Prop)
fig7=plt.figure(figsize=(8,5))
#fig7.suptitle(, fontsize=12)
ax7=fig7.add_subplot(111)
ax7.plot(RasRSProp[0],RasRSProp[1], ',g')
ax7.plot(RasFSProp[0],RasFSProp[1], ',r') 
ax7.set_xlim(1.75,2.25)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Neuron Index')

#RasFSNoProp=simplePlot(Sp_FS_NoProp)
#RasFSNoProp[1]=[l+8000 for l in RasFSNoProp[1]]
#RasRSNoProp=simplePlot(Sp_RS_NoProp)
#fig8=plt.figure(figsize=(8,4))
#fig7.suptitle(, fontsize=12)
#ax8=fig8.add_subplot(111)
#ax8.plot(RasRSNoProp[0],RasRSNoProp[1], ',g')
#ax8.plot(RasFSNoProp[0],RasFSNoProp[1], ',r') 
#ax8.set_xlim(1.5,3.5)
#ax8.set_xlabel('Time (s)')
#ax8.set_ylabel('Neuron Index')

#fig9=plt.figure(figsize=(8,4))
#fig7.suptitle(, fontsize=12)
#ax9=fig9.add_subplot(111)
#ax9.plot(TimBinnedp/1000,popRateG1p, 'r')
#ax9.plot(TimBinnedp/1000,popRateG2p, 'g')
#ax9.set_xlabel('Time (s)')
#ax9.set_ylabel('Firing Rate (Hz)')
#ax9.set_xlim(1.5,3.5)
#ax9.set_ylim(0,210)

#fig10=plt.figure(figsize=(8,4))
#fig7.suptitle(, fontsize=12)
#ax10=fig10.add_subplot(111)
#ax10.plot(TimBinnednp/1000,popRateG1np, 'r')
#ax10.plot(TimBinnednp/1000,popRateG2np, 'g')
#ax10.set_xlabel('Time (s)')
#ax10.set_ylabel('Firing Rate (Hz)')
#ax10.set_xlim(1.5,3.5)
#ax10.set_ylim(0,210)
#lims = [
#    np.min([ax6.get_xlim(), ax6.get_ylim()]),  # min of both axes
#    np.max([ax6.get_xlim(), ax6.get_ylim()]),  # max of both axes
#]

# now plot both limits against eachother
#ax6.plot(lims, lims, 'k-', alpha=0.5, zorder=0)

#fig3D1 = plt.figure()
#plt.tight_layout()
#ax3D1 = fig3D1.add_subplot(111, projection='3d')
#ax3D1.plot(RasRSProp[1],RasRSPropC[1],RasRSProp[0], ',g')
#ax3D1.plot(RasFSProp[1],RasFSPropC[1],RasFSProp[0], ',r') 

#fig2.tight_layout()
#fig3.tight_layout()
#fig4.tight_layout()
#fig5.tight_layout()
#fig6.tight_layout()
#fig7.tight_layout()
#fig8.tight_layout()
#fig9.tight_layout()
#fig10.tight_layout()
plt.show()
