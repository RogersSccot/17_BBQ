
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import scipy.fftpack

def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)


start_scope()
Cnt=0
for Nseed in [i+15 for i in range(5)]:
    for NAmp in [10,12,14,16,18]: #0,2,4,6,8]: #,10,12,14,16,18]: #range(10): #2-12 ??
        for NbS in [0,2,4,6,8,10,12,14]: #range(15):
            NbSim=NbS
            Nsim=NbS
            Cnt=Cnt+1
            seed(Nseed)
            DT=0.01
            defaultclock.dt = DT*ms
            N1 = 2000#2000
            N2 = 8000#8000
            
            #v0_max = 3.
            TotTime=2500
            duration = TotTime*ms
            
            
            seed(Nseed)
            eqs = """
            dV/dt = (-GsynE*(V-Ee)-GsynI*(V-Ei)-Gl*(V - El) - GK*(n*n*n*n)*(V - EK) - GNa*(m*m*m)*h*(V-ENa) + I)/(Cm) : volt 
            dn/dt = 0.032*(mV**-1)*(15.*mV-V+VT)/(exp((15.*mV-V+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-V+VT)/(40.*mV))/ms*n : 1 
            dh/dt = 0.128*exp((17.*mV-V+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-V+VT)/(5.*mV)))/ms*h : 1 
            dm/dt = 0.32*(mV**-1)*(13.*mV-V+VT)/(exp((13.*mV-V+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(V-VT-40.*mV)/(exp((V-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
            dGsynI/dt = -GsynI/Tsyn : siemens
            dGsynE/dt = -GsynE/Tsyn : siemens
            Cm:farad
            Gl:siemens
            El:volt
            GNa:siemens
            ENa:volt
            GK:siemens
            EK:volt
            I:ampere
            VT:volt
            Ee:volt
            Ei:volt
            Tsyn:second
            """
            #% neuron_params
                           
            

            # Population 1 - FS
            
            G1 = NeuronGroup(N1, eqs, threshold='V > -10*mV', refractory='V > -50*mV', method='heun')
            # init:
            G1.V = -65 *mV
            G1.n = 0.0 
            G1.m = 0.0
            G1.h = 0.0
            G1.GsynI =0.0 *nS
            G1.GsynE =0.0 *nS
            # parameters
            G1.Cm = 200.*pF
            G1.Gl = 10.*nS
            G1.El = -65.*mV
            G1.VT = -50.*mV
            G1.GNa = 20000*nS #CHANGE ME IF YOU NEED!CHANGED!!!!CHANGED!!!!CHANGED!!!!
            G1.GK = 6000*nS  #CHANGE ME IF YOU NEED!
            G1.ENa = 55.*mV
            G1.EK = -90.*mV
            G1.I = 0.0  
            
            G1.Ee =0.*mV
            G1.Ei =-80.*mV
            G1.Tsyn =5.*ms
            
            
            
            # Population 2 - RS
            
            G2 = NeuronGroup(N2, eqs, threshold='V > -10*mV', refractory='V > -50*mV', method='heun')#'V > -40*mV', method='heun')
            # init:
            G2.V = -65.*mV
            G2.n = 0.0 
            G2.m = 0.0
            G2.h = 0.0
            G2.GsynI =0.0 *nS
            G2.GsynE =0.0 *nS
            #parameter
            G2.Cm = 200.*pF
            G2.Gl = 15.*nS
            G2.El = -65.*mV
            G2.VT = -48.*mV
            G2.GNa = 20000*nS  #CHANGE ME IF YOU NEED!CHANGED!!!!CHANGED!!!!CHANGED!!!!CHANGED!!!!CHANGED!!!!
            G2.GK = 6000*nS   #CHANGE ME IF YOU NEED!
            G2.ENa = 55.*mV
            G2.EK = -90.*mV
            G2.I = 0.0  
            
            G2.Ee =0.*mV
            G2.Ei =-80.*mV
            G2.Tsyn =5.*ms
            
            
            
            
            # external drive--------------------------------------------------------------------------
            
            AmpStim=NAmp*5.+60  #80. #92.
            plat = 1000
            def heaviside(x):
                return 0.5 * (1 + np.sign(x))
            
            def input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
                        # t1_exc=10. # time of the maximum of external stimulation
                        # tau1_exc=20. # first time constant of perturbation = rising time
                                # tau2_exc=50. # decaying time
                                # ampl_exc=20. # amplitude of excitation
                inp = ampl_exc * (np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc)) + \
                                              heaviside(-(t - (t1_exc+plateau))) * heaviside(t - (t1_exc))+ \
                                              np.exp(-(t - (t1_exc+plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc+plateau)))
                return inp
                
                
            t2 = np.arange(0, TotTime, DT)
            test_input = []
            TauP=25.+10*NbS
            for ji in t2:
               # print(i), print (input_rate(i))
               test_input.append(6.+input_rate(ji, 1000., TauP, TauP, AmpStim, plat))
                
            stimulus=TimedArray(test_input*Hz, dt=DT*ms)
              #      print(max(test_input))
            P_ed=PoissonGroup(8000, rates='stimulus(t)')
            
            #P_edINH=PoissonGroup(8000, rates='stimulusI(t)')
            
            
            
            # connections-----------------------------------------------------------------------------
            
            Qi=5.0*nS
            Qe=1.5*nS
            
            prbC= 0.05 #0.05
            prbC2=0.05
            S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV')
            S_12.connect('i!=j', p=prbC2)
            
            S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')
            S_11.connect('i!=j',p=prbC2)
            
            S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
            S_21.connect('i!=j',p=prbC)
            
            S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
            S_22.connect('i!=j', p=prbC)
            
            
            
            
            S_ed_in = Synapses(P_ed, G1, on_pre='GsynE_post+=Qe')
            S_ed_in.connect(p=prbC)
            
            S_ed_ex = Synapses(P_ed, G2, on_pre='GsynE_post+=Qe')
            S_ed_ex.connect(p=prbC)#0.05)
            
            
            
            #S_ed_in2 = Synapses(P_edINH, G1, on_pre='GsynI_post+=Qi')
            #S_ed_in2.connect(p=prbC)
            
            #S_ed_ex2 = Synapses(P_edINH, G2, on_pre='GsynI_post+=Qi')
            #S_ed_ex2.connect(p=prbC)#0.05)
            

            Vtt=1
#
           # M1G1 = SpikeMonitor(G1)
      #  M2G1 = StateMonitor(G1, 'v', record=range(Vtt))
     #   M3G1 = StateMonitor(G1, 'w', record=range(Vtt))
   #     M4G1 = StateMonitor(G1, 'GsynE', record=range(Vtt))
    #    M5G1 = StateMonitor(G1, 'GsynI', record=range(Vtt))
           # FRG1 = PopulationRateMonitor(G1)

            #M1G2 = SpikeMonitor(G2)
     #   M2G2 = StateMonitor(G2, 'v', record=range(Vtt))
    #    M3G2 = StateMonitor(G2, 'w', record=range(Vtt))
    #    M4G2 = StateMonitor(G2, 'GsynE', record=range(Vtt))
    #    M5G2 = StateMonitor(G2, 'GsynI', record=range(Vtt))
            FRG2 = PopulationRateMonitor(G2)
            FRPed= PopulationRateMonitor(P_ed)
        #print(len(S_22.GsynE))

            print('simu #'+str(Cnt))
            #print('--##Start simulation##--')
            run(duration)#1500*ms)#2020*ms)#1250*ms)
           # print('--##End simulation##--')
   # Nbsyn=int(0.05*len(S_22.GsynE))
   # print(Nbsyn)
   # S_22.GsynE[0:Nbsyn]=0.*nS
        #S_22.GsynE=0.*nS
   # print(S_22.GsynE)

   # print('--##Start simulation##--')
   # run(3500*ms) #3500*ms)#2980*ms)#3750*ms)
   # print('--##End simulation##--')

        # Plots -------------------------------------------------------------------------------
        #trainG1=M1G1.spike_trains()
        #isi_mu_G1=[]
        #isi_std_G1=[]
        #for i in range(N1):
         #   Tr=diff(trainG1[i])
         #   if len(Tr)!=0:
     #       isi_mu_G1.append(mean(Tr))
         #       isi_std_G1.append(std(Tr))

        #fig3=plt.figure(figsize=(12,4))
        #ax33=fig3.add_subplot(111)
        #ax33.errorbar(range(len(isi_mu_G1)), isi_mu_G1, yerr=isi_std_G1)
        #plot(isi_mu_G1, 'o')
        #ax33.plot(isi_mu_G1, 'o')
        #print("CV G1=", CV_G1)
        #trainG2=M1G2.spike_trains()
        #CV_G2=diff(trainG2)
        #print("CV G2=", CV_G2)


          #  RasG1 = np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]])
          #  RasG2 = np.array([M1G2.t/ms, M1G2.i])
          #  plt.figure()
          #  plt.plot(RasG1[0], RasG1[1], ',r')
          #  plt.plot(RasG2[0], RasG2[1], ',g')


      #  LVG1=[]
      #  LwG1=[]
      #  LVG2=[]
      #  LwG2=[]

     #   LgseG1=[]
     #   LgsiG1=[]
      #  LgseG2=[]
      #  LgsiG2=[]
#
      # for a in range(Vtt):
      #      LVG1.append(array(M2G1[a].v/mV))
      #      LwG1.append(array(M3G1[a].w/mamp))
      #      LVG2.append(array(M2G2[a].v/mV))
      #      LwG2.append(array(M3G2[a].w/mamp))
      #      LgseG1.append(array(M4G1[a].GsynE/nS))
      #      LgsiG1.append(array(M5G1[a].GsynI/nS))
      #      LgseG2.append(array(M4G2[a].GsynE/nS))
      #      LgsiG2.append(array(M5G2[a].GsynI/nS))


            BIN=10
            time_array = np.arange(int(TotTime/DT))*DT



            LfrG2=np.array(FRG2.rate/Hz)
            TimBinned,popRateG2=bin_array(time_array, BIN, time_array),bin_array(LfrG2, BIN, time_array)
            LfrPed=np.array(FRPed.rate/Hz)
            TimBinned,popRatePed=bin_array(time_array, BIN, time_array),bin_array(LfrPed, BIN, time_array)

           # LfrG1=np.array(FRG1.rate/Hz)
           # TimBinned,popRateG1=bin_array(time_array, BIN, time_array),bin_array(LfrG1, BIN, time_array)

     #   Lt1G1=array(M2G1.t/ms)
     #   Lt2G1=array(M3G1.t/ms)
     #       Lt1G2=array(M2G2.t/ms)
     #       Lt2G2=array(M3G2.t/ms)
          #  DiffG2Ped= [popRateG2[i]-popRatePed[i] for i in range(len(popRatePed))]
          #  plt.figure()
          #  plt.plot(TimBinned, popRateG2, 'g')
          #  plt.plot(TimBinned, popRateG1, 'r')
          #  plt.plot(TimBinned, popRatePed, 'b')
          #  plt.show()
       # np.save('Results/AD_spikeInh_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'.npy', RasG1)
       # np.save('Results/AD_spikeExc_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'.npy', RasG2)
          #  np.save('ResultsTest/AD_popRateInh_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'Nseed_'+str(Nseed)+'.npy', popRateG1)
            np.save('ResultsHH/AD_popRateExc_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'Nseed_'+str(Nseed)+'.npy', popRateG2)
            np.save('ResultsHH/AD_popRatePed_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'Nseed_'+str(Nseed)+'.npy', popRatePed)
       # np.save('Results/AD_GtotInh_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'.npy', monGinh.Gtot[0])
       # np.save('Results/AD_GtotExc_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'.npy', monGexc.Gtot[0])
       # np.save('Results/AD_VtotInh_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'.npy', MVinh.Vtot[0])
       # np.save('Results/AD_VtotExc_Sim_'+str(TauP)+'_Amp_'+str(NAmp)+'.npy', MVexc.Vtot[0])

