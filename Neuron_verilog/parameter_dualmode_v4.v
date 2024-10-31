//////////////////////////////////////////////////
// File name: parameter_dualmode           
// Designer: Jiatong Guo                       
// Date: 2024.06.06                           
// Version: v4 
// Description: AdEx_v5
//////////////////////////////////////////////////

`define Bitwidth        27  // 1+7+19
`define Bitwidth_1      26  // Bitwidth - 1 
`define int_Bitwidth    8
`define fra_Bitwidth    19

// `define v_init      -27'b0_1000001_0000000000000000000   // -65 mv
//`define v_init          -27'b0_0111010_0000000000000000000   // -58 mV
`define v_init  0
/**'''''''''''''''''''''HH neuron'''''''''''''''''''''**/
`define m_init      27'b0_0000000_0000110110001010111    // 0.0529
`define n_init      27'b0_0000000_0101000101010100110    // 0.3177
`define h_init      27'b0_0000000_1001100010011010000    // 0.5961
`define Cm          27'b0_0000001_0000000000000000000    // 1/Cm = 1
`define vth_HH      0
//`define i_syn       27'd20480  // 5 pA >>> 7   
//`define i_syn       27'd81920  // 20 pA >>> 7   
//`define i_syn       27'd163840  // 40 pA >>> 7   
//`define i_syn       27'd245760  // 60 pA >>> 7   
`define i_syn       27'd327680  // 80 pA >>> 7   

// Set 1
`define EK          -27'd37822136   // -72.14 mv
`define EL          -27'd25910313   // -49.42 mv
`define ENa         27'd28898755    // 55.12 mv
`define gK          27'd18874368    // 36 mS
`define gL          27'd157286      // 0.3 mS
`define gNa         27'd62914560    // 120 mS

// Set 2
// `define EK          -27'd52428800   // -100 mv
// `define EL          -27'd44564480   // -85 mv
// `define ENa         27'd26214400    // 50 mv
// `define gK          27'd2621440     // 5 mS
// `define gL          27'd52429       // 0.1 mS
// `define gNa         27'd26214400    // 50 mS

/**'''''''''''''''''''''AdEx neuron'''''''''''''''''''''**/
`define w_init      0
`define v_r         -27'b0_0111010_0000000000000000000   // -58 mV
`define b_in        0
`define VT          27'd26214400    // 50 mV
`define vth_AdEx    0
`define int_thre    'd10485760 // 20

`define A1          27'd524025  // A1 = 1 - gL*dt / C        
`define A2          27'd524     // A2 = gL*dt*delta_T / C     
`define A3          -27'd5243   // A3 = (I+gL*EL)dt / C       
`define A4          -27'd53   // A4 = -dt / C  <<1               
`define B1          27'd524113  // B1 = 1 - dt / τ_w          
`define B2          27'd349   // B2 = alpha*dt / τ_w         
`define B3          27'd24466  // B3 = -alpha*EL*dt / τ_w     

// # AdEx Tonic spiking
// dt       = 1/100
// C        = 200   # 200pF
// gL       = 10    # 10nS
// El       = -70   # -70mV
// VT       = -50   # -50mV
// delt_T   = 2     # 2mV
// alpha    = 2     # 0nS
// tw       = 30    # 30mS
// b        = 0     # -58pA
// Vr       = -58   # 500mV
// I        = 500   # 2pA