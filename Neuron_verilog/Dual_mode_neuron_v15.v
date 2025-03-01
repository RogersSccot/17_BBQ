//////////////////////////////////////////////////
// File name: Dual_mode_neuron           
// Designer: Jiatong Guo                       
// Date: 2024.06.06                           
// Version: v15                             
// Description: based on v14, AdEx_v5
//////////////////////////////////////////////////

`include "parameter_dualmode_v4.v"

module Dual_mode_neuron (
    input wire clk,
    input wire rstn,
    input wire mode,    // 1:HH 0:AdEx
    
    output reg signed [`Bitwidth_1 : 0] v,
    output reg signed [26:0] h_0,
    output reg done
);

reg  signed [`Bitwidth_1 : 0] v_0, w_0;
reg  [26:0] m_0, n_0;
reg  [ 5:0] cnt;
wire    cordic_done1, cordic_done2, cordic_done3;
reg     cordic_done_r1, cordic_done_r2, cordic_done_r3;
reg     cordic_en1, cordic_en2, cordic_en3;
reg     cordic_init1, cordic_init2, cordic_init3;
reg     cordic_mode1, cordic_mode2, cordic_mode3;
reg  signed [20 : 0] cordic_x1, cordic_x2, cordic_x3, cordic_y1, cordic_y2, cordic_y3;  //CORDIC input  1+4+16bit
wire signed [22 : 0] cordic_out1, cordic_out2, cordic_out3;                             //CORDIC output 1+6+16bit
reg  signed [`Bitwidth_1 : 0] cordic_out_r1, cordic_out_r2, cordic_out_r3;
reg  signed [26 : 0] temp1;
reg  signed [23 : 0] temp2;
reg  signed [20 : 0] temp3, temp4;
reg  signed [22 : 0] temp5;

/*******************************Multiplier*******************************/
reg  signed [`Bitwidth_1 : 0] mult_1, mult_2;
wire signed [`Bitwidth_1 : 0] mult_out;
reg  signed [`Bitwidth*2-1 : 0] mult_out_temp;

always @(posedge clk or negedge rstn) begin
    if(!rstn) mult_out_temp <= 0;
    else      mult_out_temp <= mult_1 * mult_2;
end

assign mult_out = {mult_out_temp[`Bitwidth*2-1], mult_out_temp[(`Bitwidth + `fra_Bitwidth - 2) : `fra_Bitwidth]}; // abandon low frac_bit

/********************************Adder**************************/
reg signed [`Bitwidth_1 : 0]  add_1, add_2, add_out1;

always @(posedge clk or negedge rstn) begin
    if(!rstn) add_out1 <= 0;
    else      add_out1 <= add_1 + add_2;
end

/***************************Pipeline Control**************************/
always @(posedge clk or negedge rstn) begin
    if(!rstn)                                                                                       cnt <= 0;
    else if(mode) begin // HH mode
        if(cnt == 11 && (cordic_done_r1 != 1 || cordic_done_r2 != 1 || cordic_done_r3 != 1))        cnt <= cnt;
        else if(cnt == 21 && (cordic_done_r1 != 1 || cordic_done_r2 != 1 || cordic_done_r3 != 1))   cnt <= cnt;
        else if(cnt == 30 && cordic_done_r3 != 1)                                                   cnt <= cnt;
        else if(cnt == 35)                                                                          cnt <= 0;
        else                                                                                        cnt <= cnt + 1;
    end
    else begin  // AdEx mode
        if(cnt == 5 && cordic_done_r1 != 1)                                 cnt <= cnt;
        else if(cnt == 7 || (cnt == 1 && add_out1 >= $signed(`int_thre)))   cnt <= 0;    
        else                                                                cnt <= cnt + 1;
    end
end

/**************** input of multiplier and adders ****************/
always @(*) begin
    if(mode) begin  //HH mode
        case(cnt)
        6'd0:begin
            mult_1 = $signed(`Bitwidth'd52429);   //0.1
            mult_2 = v_0;
            add_1  = v_0;
            add_2  = $signed(`Bitwidth'd31457280);  //60
        end
        6'd1:begin
            mult_1 = add_out1;  // V+60
            mult_2 = $signed(`Bitwidth'd26215);   // 1/20
            add_1  = 0;
            add_2  = 0;
        end
        6'd2:begin
            mult_1 = temp1; // V+60
            mult_2 = $signed(`Bitwidth'd6554);   // 1/80
            add_1  = 0; 
            add_2  = 0;
        end
        6'd3:begin
            mult_1 = temp1;   // V+60
            mult_2 = $signed(`Bitwidth'd29127); // 1/18
            add_1  = v_0;
            add_2  = $signed(`Bitwidth'd18350080);  // 35
        end
        6'd4:begin
            mult_1 = add_out1; // V+35
            mult_2 = $signed(`Bitwidth'd52429); // 0.1
            add_1  = v_0;
            add_2  = $signed(`Bitwidth'd26214400);  // 50
        end
        6'd5:begin
            mult_1 = add_out1;  // V+50
            mult_2 = $signed(`Bitwidth'd5243);  // 0.01
            add_1  = $signed(`EL);
            add_2  = ~v_0 + 1;
        end
        6'd6:begin
            mult_1 = $signed(`gL);  
            mult_2 = (add_out1 >>> 7);  // (EL - V) * dt
            add_1  = 0;
            add_2  = 0;
        end
        6'd7:begin
            mult_1 = n_0;
            mult_2 = n_0;
            add_1  = 0;
            add_2  = 0;
        end
        6'd8:begin 
            mult_1 = mult_out;  // n^2
            mult_2 = mult_out;
            add_1  = 0;
            add_2  = 0;
        end
        6'd9:begin
            mult_1 = mult_out;  // n^4
            mult_2 = $signed(`gK);
            add_1  = $signed(`EK);
            add_2  = ~v_0 + 1;
        end
        6'd10:begin
            mult_1 = mult_out;  // n^4 * gK
            mult_2 = (add_out1 >>> 7);  // (EK - V) * dt
            add_1  = 0;
            add_2  = 0;
        end
        6'd11:begin // wait CORDIC
            mult_1 = mult_out;  // IK * dt
            mult_2 = $signed(`Bitwidth'd524288); // 1
            add_1  = 0;
            add_2  = 0;
        end
        6'd12:begin 
            mult_1 = cordic_out_r1; // e^-(0.1V+3)
            mult_2 = $signed(`Bitwidth'd317997);   // e^-0.5 
            add_1  = $signed(`Bitwidth'd524288);
            add_2  = cordic_out_r1;
        end
        6'd13:begin
            mult_1 = cordic_out_r1; // e^-(0.1V+3)
            mult_2 = $signed(`Bitwidth'd70955);   // e^-2 
            add_1  = $signed(`Bitwidth'd524288);
            add_2  = ~mult_out+1;   // -e^-(0.1V+3.5)
        end
        6'd14:begin
            mult_1 = m_0;
            mult_2 = m_0;
            add_1  = $signed(`Bitwidth'd524288);
            add_2  = ~mult_out+1;   // -e^-(0.1V+5)
        end
        6'd15:begin 
            mult_1 = mult_out;
            mult_2 = m_0;
            add_1  = 0;
            add_2  = 0;
        end
        6'd16:begin
            mult_1 = mult_out;
            mult_2 = h_0;
            add_1  = 0;
            add_2  = 0;
        end
        6'd17:begin
            mult_1 = mult_out;
            mult_2 = $signed(`gNa);
            add_1  = $signed(`ENa);
            add_2  = ~v_0+1;
        end
        6'd18:begin
            mult_1 = mult_out;  // GNa
            mult_2 = (add_out1 >>> 7);
            add_1  = { {6{temp4[20]}} , temp4}; // IL *dt        
            add_2  = { {4{temp5[22]}} , temp5}; // IK *dt
        end
        6'd19:begin
            mult_1 = { {3{temp2[23]}} , temp2}; // to be changed
            mult_2 = $signed(`Bitwidth'd36700); // 0.07
            add_1  = add_out1;  // dt * (IL + IK)
            add_2  = mult_out;  // dt * INa
        end
        6'd20:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = add_out1;  // dt * (IL + IK + INa)
            add_2  = $signed(`i_syn);  // Isyn * dt
        end
        6'd21:begin // wait CORDIC
            mult_1 = 0;
            mult_2 = 0;
            add_1  = add_out1;  // dt * (IL + IK + INa + Isyn)
            add_2  = 0;        
        end
        6'd22:begin
            mult_1 = add_out1;  // dt * (IL + IK + INa + Isyn)
            mult_2 = $signed(`Cm);
            add_1  = cordic_out_r1;
            add_2  = { {6{temp3[20]}} , temp3} >>> 3;
        end
        6'd23:begin
            mult_1 = add_out1;  // An + Bn
            mult_2 = n_0;
            add_1  = v_0;
            add_2  = mult_out;  // delta_V
        end
        6'd24:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = cordic_out_r1;
            add_2  = ~mult_out + 1;
        end
        6'd25:begin 
            mult_1 = 0;
            mult_2 = 0;
            add_1  = add_out1 >>> 7;  // delta_n
            add_2  = n_0;
        end
        6'd26:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = cordic_out_r2;
            add_2  = { {3{temp2[23]}} , temp2} <<< 2;
        end
        6'd27:begin
            mult_1 = add_out1;
            mult_2 = m_0;
            add_1  = 0;
            add_2  = 0;
        end
        6'd28:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = cordic_out_r2;    
            add_2  = ~mult_out + 1;
        end
        6'd29:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = add_out1 >>> 7;
            add_2  = m_0;
        end
        6'd30:begin // wait CORDIC 3
            mult_1 = 0;
            mult_2 = 0;
            add_1  = add_out1;
            add_2  = 0;
        end
        6'd31:begin 
            mult_1 = 0;
            mult_2 = 0;
            add_1  = { {6{temp4[20]}} , temp4};
            add_2  = cordic_out_r3;
        end
        6'd32:begin
            mult_1 = add_out1;
            mult_2 = h_0;
            add_1  = 0;
            add_2  = 0;
        end
        6'd33:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = { {6{temp4[20]}} , temp4};
            add_2  = ~mult_out + 1;
        end
        6'd34:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = add_out1 >>> 7;    // delta_h * dt
            add_2  = h_0;
        end
        default:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = 0;
            add_2  = 0;
        end
        endcase
    end
    else begin  //AdEx mode
        case(cnt)
        6'd0:begin
            mult_1 = `B2;  
            mult_2 = v_0;
            add_1  = v_0;
            add_2  = `VT; 
        end
        6'd1:begin
            mult_1 = `B1;
            mult_2 = w_0;
            add_1  = mult_out;
            add_2  = `B3;
        end
        6'd2:begin
            mult_1 = `A1;
            mult_2 = v_0;
            add_1  = mult_out;
            add_2  = add_out1;
        end
        6'd3:begin
            mult_1 = `A4;
            mult_2 = w_0;
            add_1  = mult_out;
            add_2  = `A3;
        end
        6'd4:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = mult_out >>> 1;
            add_2  = add_out1;
        end
        6'd5:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = 0;
            add_2  = add_out1;
        end
        6'd6:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = cordic_out_r1;
            add_2  = add_out1;
        end
        default:begin
            mult_1 = 0;
            mult_2 = 0;
            add_1  = 0;
            add_2  = 0;
        end
        endcase
    end
end


/*******************************************************************************************************************************/
/********************************'''''''''''''''''''''''''''''''CORDIC1'''''''''''''''''''''''''''''''**************************/
/*******************************************************************************************************************************/
RFC_CORDIC cordic1(
    .clk(clk),              // input
    .rst_n(rstn),
    .init(cordic_init1),
    .x_init(cordic_x1),
    .y_init(cordic_y1),
    .mode(cordic_mode1),
    .result(cordic_out1),   // output
    .done(cordic_done1)
);

always @(posedge clk, negedge rstn) begin
    if(!rstn)                       cordic_en1 <= 0;
    else if(mode) begin
        if(cnt == 1 || cnt == 15)   cordic_en1 <= 1;
        else                        cordic_en1 <= 0;
    end
    else begin
        if(cnt == 1)                cordic_en1 <= 1;
        else                        cordic_en1 <= 0;
    end
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                       cordic_init1 <= 0;
    else if(cordic_en1)             cordic_init1 <= 1;
    else                            cordic_init1 <= 0;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                       cordic_mode1 <= 0;
    else if(mode) begin
        if(cnt == 1)                cordic_mode1 <= 1;
        else if(cnt == 15)          cordic_mode1 <= 0;
        else                        cordic_mode1 <= cordic_mode1;
    end
    else                            cordic_mode1 <= 1;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
            cordic_x1 <= 0;
            cordic_y1 <= 0;
    end
    else if(mode && cnt == 1) begin
            cordic_x1 <= ~({mult_out[26],mult_out[22:3]} + 21'd196608) + 1;   // -(0.1V + 3)
            cordic_y1 <= 0;
        end
    else if(mode && cnt == 15) begin
            cordic_x1 <= { {6{temp3[20]}}, temp3[19:5]};       // 0.01(V+50) >> 2
            cordic_y1 <= {add_out1[26], add_out1[24:5]};    // 1-e^-(0.1V+5) >> 2
    end
    else if(!mode && cnt == 1) begin
            cordic_x1 <= ( {add_out1[26], add_out1[23:4]} - $signed(21'd452706));  // 6.90775
            cordic_y1 <= 0;
        end
    else if(cordic_en1) begin
            cordic_x1 <= cordic_x1;
            cordic_y1 <= cordic_y1;
    end
    else begin
            cordic_x1 <= 0;
            cordic_y1 <= 0;
    end
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                               cordic_done_r1 <= 0;
    else if(cordic_done1)                   cordic_done_r1 <= 1;
    else if(mode &&(cnt == 12 ||cnt == 22)) cordic_done_r1 <= 0;
    else if(!mode && cnt == 6)              cordic_done_r1 <= 0;
    else                                    cordic_done_r1 <= cordic_done_r1;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)               cordic_out_r1 <= 0;
    else if(cordic_done1)   cordic_out_r1 <= {{2{cordic_out1[22]}}, cordic_out1[21:0], {3{cordic_out1[22]}}};   // ???
    else                    cordic_out_r1 <= cordic_out_r1;
end

/*******************************************************************************************************************************/
/********************************'''''''''''''''''''''''''''''''CORDIC2'''''''''''''''''''''''''''''''**************************/
/*******************************************************************************************************************************/

RFC_CORDIC cordic2(
    .clk(clk),              // input
    .rst_n(rstn),
    .init(cordic_init2),
    .x_init(cordic_x2),
    .y_init(cordic_y2),
    .mode(cordic_mode2),
    .result(cordic_out2),   // output
    .done(cordic_done2)
);

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                   cordic_en2 <= 0;
    else if(mode && (cnt == 2 || cnt == 14))    cordic_en2 <= 1;
    else                                        cordic_en2 <= 0;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                   cordic_init2 <= 0;
    else if(cordic_en2)                         cordic_init2 <= 1;
    else                                        cordic_init2 <= 0;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                   cordic_mode2 <= 0;
    else if(mode && cnt == 2)                   cordic_mode2 <= 1;
    else if(mode && cnt == 14)                  cordic_mode2 <= 0;
    else                                        cordic_mode2 <= cordic_mode2;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
            cordic_x2 <= 0;
            cordic_y2 <= 0;
    end
    else if(mode && cnt == 2) begin
            cordic_x2 <= ~({mult_out[26],mult_out[22:3]}) + 1;
            cordic_y2 <= 0;
        end
    else if(mode && cnt == 14) begin
            cordic_x2 <= { {3{temp2[23]}}, temp2[22:5]};    // 0.1(V+35) >>> 2
            cordic_y2 <= {add_out1[26], add_out1[24:5]};    // 1-e^-(0.1V+3.5) >>> 2
    end
    else begin
            cordic_x2 <= cordic_x2;
            cordic_y2 <= cordic_y2;
    end
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                               cordic_done_r2 <= 0;
    else if(cordic_done2)                   cordic_done_r2 <= 1;
    else if(mode &&(cnt==12 || cnt==22))    cordic_done_r2 <= 0;
    else                                    cordic_done_r2 <= cordic_done_r2;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                               cordic_out_r2 <= 0;
    else if(cordic_done2)                   cordic_out_r2 <= {{2{cordic_out2[22]}}, cordic_out2[21:0], {3{cordic_out2[22]}}};
    else                                    cordic_out_r2 <= cordic_out_r2;
end


/*******************************************************************************************************************************/
/********************************'''''''''''''''''''''''''''''''CORDIC3'''''''''''''''''''''''''''''''**************************/
/*******************************************************************************************************************************/
RFC_CORDIC cordic3(
    .clk(clk),              // input
    .rst_n(rstn),
    .init(cordic_init3),
    .x_init(cordic_x3),
    .y_init(cordic_y3),
    .mode(cordic_mode3),
    .result(cordic_out3),   // output
    .done(cordic_done3)
);

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                               cordic_en3 <= 0;
    else if(mode && (cnt == 3 || cnt == 12 || cnt ==22))    cordic_en3 <= 1;
    else                                                    cordic_en3 <= 0;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                               cordic_init3 <= 0;
    else if(cordic_en3)                                     cordic_init3 <= 1;
    else                                                    cordic_init3 <= 0;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                               cordic_mode3 <= 0;
    else if(mode && (cnt == 3 || cnt == 12))                cordic_mode3 <= 1;
    else if(mode && cnt == 22)                              cordic_mode3 <= 0;
    else                                                    cordic_mode3 <= cordic_mode3;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
            cordic_x3 <= 0;
            cordic_y3 <= 0;
    end
    else if(mode && cnt == 3) begin
            cordic_x3 <= ~ ({mult_out[26],mult_out[22:3]}) + 1; // -(V+60)/80
            cordic_y3 <= 0;
        end
    else if(mode && cnt == 12) begin
            cordic_x3 <= {temp1[26],temp1[22:3]};   // -(V+60)/18
            cordic_y3 <= 0;
    end
    else if(mode && cnt == 22) begin
            cordic_x3 <= $signed(21'd16384);       // 1 >> 2
            cordic_y3 <= {temp1[26],temp1[24:5]};  // 1+e^-(0.1V+3) >> 2
    end
    else begin
            cordic_x3 <= cordic_x3;
            cordic_y3 <= cordic_y3;
    end
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                       cordic_done_r3 <= 0;
    else if(cordic_done3)                           cordic_done_r3 <= 1;
    else if(mode &&(cnt==12 || cnt==22 || cnt==31)) cordic_done_r3 <= 0;
    else                                            cordic_done_r3 <= cordic_done_r3;
end

always @(posedge clk, negedge rstn) begin
    if(!rstn)                                       cordic_out_r3 <= 0;
    else if(cordic_done3)                           cordic_out_r3 <= {{2{cordic_out3[22]}}, cordic_out3[21:0], {3{cordic_out3[22]}}};
    else                                            cordic_out_r3 <= cordic_out_r3;
end


/*******************************************************************************************************************************/
/****************************'''''''''''''''''''''''''''''''reg memory'''''''''''''''''''''''''''''''**************************/
/*******************************************************************************************************************************/

always @(posedge clk, negedge rstn) begin
    if(!rstn)               temp1 <= 0;
    else if(mode) begin
        if(cnt == 1)        temp1 <= add_out1;          // V+60
        else if(cnt == 4)   temp1 <= ~mult_out + 1;     // -(V+60)/18
        else if(cnt == 13)  temp1 <= add_out1;          // 1+e^-(0.1V+3)
        else                temp1 <= temp1;
    end
    else   
        if(cnt == 3)        temp1 <= add_out1;  // W_update
        else                temp1 <= temp1;
end


always @(posedge clk, negedge rstn) begin
    if(!rstn)               temp2 <= 0; // 1+4+19 bit
    else if(mode) begin
        if(cnt == 5)        temp2 <= {mult_out[26], mult_out[22 : 0]};          // 0.1(V+35)
        else if(cnt == 15)  temp2 <= {cordic_out_r2[26], cordic_out_r2[22 : 0]}; //e^-(V+60)/20
        else if(cnt == 22)  temp2 <= {cordic_out_r3[26], cordic_out_r3[22 : 0]}; //e^-(V+60)/18
        else                temp2 <= temp2;
    end
    else                    temp2 <= 0;
end


always @(posedge clk, negedge rstn) begin
    if(!rstn)               temp3 <= 0; // 1+1+19 bit
    else if(mode) begin
        if(cnt == 6)        temp3 <= {mult_out[26], mult_out[19 : 0]};  // 0.01(V+50)
        else if(cnt == 19)  temp3 <= {cordic_out_r3[26], cordic_out_r3[19 : 0]}; //e^-(V+60)/80
        else                temp3 <= temp3;
    end
    else                    temp3 <= 0;
end


always @(posedge clk, negedge rstn) begin
    if(!rstn)                           temp4 <= 0; // 1+1+19 bit
    else if(mode) begin
        if(cnt == 7 || cnt == 20)       temp4 <= {mult_out[26], mult_out[19 : 0]};  // EK-V * dt or alpha_h
        else                            temp4 <= temp4;
    end
    else                                temp4 <= 0;
end


always @(posedge clk, negedge rstn) begin
    if(!rstn)               temp5 <= 0; // 1+3+19 bit
    else if(mode) begin
        if(cnt == 11)       temp5 <= {mult_out[26], mult_out[21 : 0]};  // IK = n^4*gk*(Ek-V)
        else                temp5 <= temp5;
    end
    else                    temp5 <= 0;
end

/********************************Update**************************/
always @(posedge clk or negedge rstn) begin
    if(!rstn)                                       v_0 <= `v_init;
    else if(mode)  begin
        if(cnt == 24)                               v_0 <= add_out1;
        else                                        v_0 <= v_0;
    end
    else begin
        if(cnt == 7 && add_out1 < `vth_AdEx)        v_0 <= add_out1;
        else if((cnt == 7 && add_out1 >= `vth_AdEx) 
        || (cnt == 1 && add_out1 >= $signed(`int_thre)))         v_0 <= `v_r;
        else                                        v_0 <= v_0;
    end
end

always @(posedge clk or negedge rstn) begin
    if(!rstn)                                       w_0 <= `w_init;
    else if(!mode) begin
        if(cnt == 7 && add_out1 < `vth_AdEx)        w_0 <= temp1;
        else if((cnt == 7 && add_out1 >= `vth_AdEx) 
        || (cnt == 1 && add_out1 >= $signed(`int_thre)))         w_0 <= temp1 + `b_in;
        else                                        w_0 <= w_0;
    end
    else                                            w_0 <= 0;
end

always @(posedge clk or negedge rstn) begin
    if(!rstn)                   m_0 <= `m_init;
    else if(mode && cnt == 30)  m_0 <= add_out1;
    else                        m_0 <= m_0;
end

always @(posedge clk or negedge rstn) begin
    if(!rstn)                   n_0 <= `n_init;
    else if(mode && cnt == 26)  n_0 <= add_out1;
    else                        n_0 <= n_0;
end

always @(posedge clk or negedge rstn) begin
    if(!rstn)                   h_0 <= `h_init;
    else if(mode && cnt == 35)  h_0 <= add_out1;
    else                        h_0 <= h_0;
end


/********************************Output**************************/
always @(posedge clk or negedge rstn) begin
    if(!rstn)   v <= `v_init;
    else        v <= v_0;
end

always @(posedge clk or negedge rstn) begin
    if(!rstn)                                   done <= 0;
    else if(mode && cnt == 35)                  done <= 1;
    else if(!mode && cnt == 7)                  done <= 1;
    else if(!mode && (cnt == 1 && add_out1 >= $signed(`int_thre))) done <= 1;
    else                                        done <= 0;
end

 reg [20:0] itr_cnt;
 always @(posedge clk or negedge rstn) begin
     if(!rstn)                                   itr_cnt <= 0;
     else if(mode && cnt == 35)                  itr_cnt <= itr_cnt + 1'b1;
     else if(!mode && cnt == 7)                  itr_cnt <= itr_cnt + 1'b1;
     else                                        itr_cnt <= itr_cnt;
 end


endmodule