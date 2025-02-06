
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import scipy.fftpack



for Nseed in range(50):
    for NbS in range(20):
        NbSim=NbS
        Nsim=NbS
        print('simulation #'+str(NbSim*Nseed))
        def bin_array(array, BIN, time_array):
            N0 = int(BIN/(time_array[1]-time_array[0]))
            N1 = int((time_array[-1]-time_array[0])/BIN)
            return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

   # LT=-50.+Nsim/2.
      #  Nseed=1
        seed(Nseed)
        start_scope()
        DT=0.1
        defaultclock.dt = DT*ms
        N1 = 2000#2000
        N2 = 8000#8000

        #tau = 10*ms
        #v0_max = 3.
        TotTime=3000
        duration = TotTime*ms


        #neuron_params={'gl':2.0,'El':-60.e-3,'a':0.5,'tau_w':100,'Dt':2.0, 'Vt':0}  ####### -GsynE*(v-Ee)-GsynI*(v-Ei)

        eqs=eqs='''
        dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w*(v-Ea) + Is)/Cm : volt (unless refractory)
        dw/dt = (ga/(1.0+exp((Vc-v)/Vd))-w)/tau_w:siemens
        dGsynI/dt = -GsynI/Tsyn : siemens
        dGsynE/dt = -GsynE/Tsyn : siemens
        Is:ampere
        Cm:farad
        gl:siemens
        El:volt
        Ea:volt
        tau_w:second
        Dt:volt
        Vt:volt
        Ee:volt
        Ei:volt
        Vc:volt
        Vd:volt
        ga:siemens
        Tsyn:second
        '''#% neuron_params

        # Populations----------------------------------------------------------------------------------
        #NAME=='FS-cell':
        #params = {'name':name, 'N':number,\
        #                  'Gl':10., 'Cm':200.,'Trefrac':5,\
    #                      'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':0.5,'ampnoise':0.,\
        #                  'a':0., 'b': 0., 'tauw':1e9}
    #NAME=='RS-cell':
        #params = {'name':name, 'N':number,\
        #                  'Gl':10., 'Cm':200.,'Trefrac':5,\
        #                  'El’:-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
        #                  'a':0., 'b':60., 'tauw':1000.}



        Par1={'gl': 100.e-09, 'El': -0.065, 'Vt': -0.05, 'ga': 0.e-09, 'Ea': -0.070, 'Tsyn': 0.005, 'b': 0.0, 'Ee': 0.0, 'Cm': 2e-10, 'Dt': 0.0005, 'Vc': -0.045, 'Is': 0.0, 'refractory': 0.005, 'Ei': -0.08, 'Vreset': -0.065, 'tau_w': 0.0001, 'Vd': 0.005}
        # Population 1 - FS
        b1 = 0.0*nS
        G1 = NeuronGroup(N1, eqs, threshold='v > -47.5*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
        #init:
        G1.v = -50*mV
        G1.w = 0.0*nS
        G1.GsynI=0.0*nS
        G1.GsynE=0.0*nS
        #parameters
        G1.Cm     = Par1['Cm']*farad
        G1.gl     = Par1['gl']*siemens
        G1.El     = Par1['El']*volt
        G1.Vt     = Par1['Vt']*volt
        G1.Dt     = Par1['Dt']*volt
        G1.tau_w  = Par1['tau_w']*second
        G1.Is     = Par1['Is']*amp  
        G1.Ee     = Par1['Ee']*volt
        G1.Ei     = Par1['Ei']*volt
        G1.Tsyn   = Par1['Tsyn']*second
        G1.Vc     = Par1['Vc']*volt
        G1.Vd     = Par1['Vd']*volt
        G1.ga     = Par1['ga']*siemens
        G1.Ea     = Par1['Ea']*volt




        Par2={'gl': 10.e-09, 'El': -0.065, 'Vt': -0.05, 'ga': 1.e-09, 'Ea': -0.065, 'Tsyn': 0.001, 'b': 1.e-09, 'Ee': 0.0, 'Cm': 2e-10, 'Dt': 0.002, 'Vc': -0.03, 'Is': 0.0, 'refractory': 0.005, 'Ei': -0.08, 'Vreset': -0.065, 'tau_w': 1.0, 'Vd': 0.001}

        # Population 2 - RS
        b2 = Par2['b']*siemens
        G2 = NeuronGroup(N2, eqs, threshold='v > -40.*mV', reset='v = -65*mV; w += b2', refractory='5*ms',  method='heun')
        G2.v = -41.*mV
        G2.w = 0.0*nS
        G2.GsynI=0.0*nS
        G2.GsynE=0.0*nS
        G2.Cm     = Par2['Cm']*farad
        G2.gl     = Par2['gl']*siemens
        G2.El     = Par2['El']*volt
        G2.Vt     = Par2['Vt']*volt
        G2.Dt     = Par2['Dt']*volt
        G2.tau_w  = Par2['tau_w']*second
        G2.Is     = Par2['Is']*amp  
        G2.Ee     = Par2['Ee']*volt
        G2.Ei     = Par2['Ei']*volt
        G2.Tsyn   = Par2['Tsyn']*second
        G2.Vc     = Par2['Vc']*volt
        G2.Vd     = Par2['Vd']*volt
        G2.ga     = Par2['ga']*siemens
        G2.Ea     = Par2['Ea']*volt

        # external drive--------------------------------------------------------------------------
        AmpStim=80. #92.
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
        TauP=20.+15*NbS
        for ji in t2:
            # print(i), print (input_rate(i))
            test_input.append(6.+input_rate(ji, 1000., TauP, TauP, AmpStim, plat))

        stimulus=TimedArray(test_input*Hz, dt=DT*ms)
  #  print(max(test_input))
        P_ed=PoissonGroup(8000, rates='stimulus(t)')

        # connections-----------------------------------------------------------------------------

        Qi=5.0*nS
        Qe=1.5*nS

        prbC= 0.05 #0.05
        prbC2= 0.05#0.065
        S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV')
        S_12.connect('i!=j', p=prbC)

        S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')
        S_11.connect('i!=j',p=prbC)

        S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
        S_21.connect('i!=j',p=prbC)

        S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
        S_22.connect('i!=j', p=prbC)




        S_ed_in = Synapses(P_ed, G1, on_pre='GsynE_post+=Qe')
        S_ed_in.connect(p=prbC2)

        S_ed_ex = Synapses(P_ed, G2, on_pre='GsynE_post+=Qe')
        S_ed_ex.connect(p=prbC)#0.05)




        #FinalPre = {key: value + two[key] + [three[key]] for key, value in one.iteritems()}

        eqGtot=''' 
        Gt1:siemens
        Gt2:siemens
        Gtot=Gt1+Gt2:siemens
        '''
        Ginh=NeuronGroup(1, eqGtot, method='rk4')
        Gexc=NeuronGroup(1, eqGtot, method='rk4')
        ScInh1=Synapses(G1, Ginh, 'Gt1_post = GsynI_pre : siemens (summed)')
        ScInh1.connect(p=1)
        ScInh2=Synapses(G2, Ginh, 'Gt2_post = GsynI_pre : siemens (summed)')
        ScInh2.connect(p=1)
        ScExc1=Synapses(G1, Gexc, 'Gt1_post = GsynE_pre : siemens (summed)')
        ScExc1.connect(p=1)
        ScExc2=Synapses(G2, Gexc, 'Gt2_post = GsynE_pre : siemens (summed)')
        ScExc2.connect(p=1)

        monGinh = StateMonitor(Ginh, 'Gtot', record=0)
        monGexc = StateMonitor(Gexc, 'Gtot', record=0)
        GV_inh = NeuronGroup(1, 'Vtot : volt', method='rk4')
        GV_exc = NeuronGroup(1, 'Vtot : volt', method='rk4')
        SvInh1=Synapses(G1, GV_inh, 'Vtot_post = v_pre : volt (summed)')
        SvInh1.connect(p=1)
        SvExc1=Synapses(G2, GV_exc, 'Vtot_post = v_pre : volt (summed)')
        SvExc1.connect(p=1)
        MVinh = StateMonitor(GV_inh, 'Vtot', record=0)
        MVexc = StateMonitor(GV_exc, 'Vtot', record=0)

        Vtt=1

        M1G1 = SpikeMonitor(G1)
        M2G1 = StateMonitor(G1, 'v', record=range(Vtt))
        M3G1 = StateMonitor(G1, 'w', record=range(Vtt))
        M4G1 = StateMonitor(G1, 'GsynE', record=range(Vtt))
        M5G1 = StateMonitor(G1, 'GsynI', record=range(Vtt))
        FRG1 = PopulationRateMonitor(G1)

        M1G2 = SpikeMonitor(G2)
        M2G2 = StateMonitor(G2, 'v', record=range(Vtt))
        M3G2 = StateMonitor(G2, 'w', record=range(Vtt))
        M4G2 = StateMonitor(G2, 'GsynE', record=range(Vtt))
        M5G2 = StateMonitor(G2, 'GsynI', record=range(Vtt))
        FRG2 = PopulationRateMonitor(G2)

        #print(len(S_22.GsynE))


        print('--##Start simulation##--')
        run(duration)#1500*ms)#2020*ms)#1250*ms)
        print('--##End simulation##--')
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


        RasG1 = np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]])
        RasG2 = np.array([M1G2.t/ms, M1G2.i])


        LVG1=[]
        LwG1=[]
        LVG2=[]
        LwG2=[]

        LgseG1=[]
        LgsiG1=[]
        LgseG2=[]
        LgsiG2=[]

        for a in range(Vtt):
            LVG1.append(array(M2G1[a].v/mV))
            LwG1.append(array(M3G1[a].w/mamp))
            LVG2.append(array(M2G2[a].v/mV))
            LwG2.append(array(M3G2[a].w/mamp))
            LgseG1.append(array(M4G1[a].GsynE/nS))
            LgsiG1.append(array(M5G1[a].GsynI/nS))
            LgseG2.append(array(M4G2[a].GsynE/nS))
            LgsiG2.append(array(M5G2[a].GsynI/nS))


        BIN=10
        time_array = np.arange(int(TotTime/DT))*DT



        LfrG2=np.array(FRG2.rate/Hz)
        TimBinned,popRateG2=bin_array(time_array, BIN, time_array),bin_array(LfrG2, BIN, time_array)

        LfrG1=np.array(FRG1.rate/Hz)
        TimBinned,popRateG1=bin_array(time_array, BIN, time_array),bin_array(LfrG1, BIN, time_array)

        Lt1G1=array(M2G1.t/ms)
        Lt2G1=array(M3G1.t/ms)
        Lt1G2=array(M2G2.t/ms)
        Lt2G2=array(M3G2.t/ms)

        np.save('Results/AD_spikeInh_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', RasG1)
        np.save('Results/AD_spikeExc_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', RasG2)
        np.save('Results/AD_popRateInh_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', popRateG1)
        np.save('Results/AD_popRateExc_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', popRateG2)
        np.save('Results/AD_GtotInh_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', monGinh.Gtot[0])
        np.save('Results/AD_GtotExc_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', monGexc.Gtot[0])
        np.save('Results/AD_VtotInh_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', MVinh.Vtot[0])
        np.save('Results/AD_VtotExc_Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.npy', MVexc.Vtot[0])

        fig2=plt.figure(figsize=(12,4))
        ax1=fig2.add_subplot(221)
        ax2=fig2.add_subplot(222)

        for a in range(len(LVG1)):
            ax1.plot(Lt1G1, LVG1[a],'r')
            ax2.plot(Lt2G1, LwG1[a],'r')
            ax1.plot(Lt1G2, LVG2[a],'g')
            ax2.plot(Lt2G2, LwG2[a],'g')


        ax1.set_ylim([-100, 0])
        ax3=fig2.add_subplot(223)
        ax3.plot(RasG1[0], RasG1[1], ',r')
        ax3.plot(RasG2[0], RasG2[1], ',g')

        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Neuron index')
        ax4=fig2.add_subplot(224)
        ax4.plot(TimBinned,popRateG1, 'r')
        ax4.plot(TimBinned,popRateG2, 'g')

        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('FR')
        plt.show()
        plt.savefig('SimPlot/Sim_'+str(TauP)+'_seed_'+str(Nseed)+'.png')
        plt.clf()
        plt.cla()
        plt.close('all')
#np.save('SimPlot/Results', ResultsTotal )

        #fig2=plt.figure(figsize=(12,4))
        #ax21=fig2.add_subplot(111)
        #ST_popRateG1=popRateG1[int(len(popRateG1)/5)::]
        #ST_popRateG2=popRateG2[int(len(popRateG2)/5)::]
        #ax21.hist(ST_popRateG1, color='r', normed=True, bins=20, alpha=0.5)
        #ax21.hist(ST_popRateG2, color='g', normed=True, bins=20, alpha=0.5)
        #ax21.axvline(x=mean(ST_popRateG1), color='r', lw=2.0)
        #ax21.axvline(x=mean(ST_popRateG2), color='g', lw=2.0)


#        yfG1 = scipy.fftpack.fft(popRateG1[int(len(popRateG1)/5)::])
#        yfG2 = scipy.fftpack.fft(popRateG2[int(len(popRateG2)/5)::])
#        N2=len(popRateG1[int(len(popRateG1)/5)::])
#        ax21.plot(2.0/N2 * np.abs(yfG1[:N2//2]), 'r')
#        ax21.plot(2.0/N2 * np.abs(yfG2[:N2//2]), 'g')


 #   figRP=plt.figure(figsize=(12,4))
 #   axRP=figRP.add_subplot(111)

  #  axRP.plot(RasG1[0][int(len(popRateG1)/10)::], RasG1[1][int(len(popRateG1)/10)::], ',r')
  #  axRP.plot(RasG2[0][int(len(popRateG1)/10)::], RasG2[1][int(len(popRateG1)/10)::], ',g')
#

#        axRP.set_xlabel('Time (ms)')

        #for a in range(len(LVG1)):
        ##        ax21.plot(Lt1G1, LgseG1[a])
        #        ax21.plot(Lt2G1, LgsiG1[a])
        #        ax21.plot(Lt1G1, LgseG2[a] )
        #        ax21.plot(Lt2G1, LgsiG2[a])
 #   plt.show()

