//////////////////////////////////////////////////////////
// File name: RFC_CORDIC                                
// Designer: Jiatong Guo                               
// Date: 2024.06.05                                     
// Version: v15                                        
// Description: based on v14, optimized exp-length revision     
//////////////////////////////////////////////////////////

`define Bitwidth        21  // 1+4+16
`define Bitwidth_1      20  // `Bitwidth - 1 (signbit)
`define Bitwidth_2      19  // `Bitwidth - 2 (intbit_start)
`define out_Bitwidth    23
`define out_Bitwidth_1  22
`define int_Bitwidth    5
`define fra_Bitwidth    16

module RFC_CORDIC(
    input wire clk, 
    input wire rst_n, 
    input wire init,
    input wire mode, // 1:exp, 0:div_x/y
    input wire signed [`Bitwidth_1 : 0] x_init, // 1+4+16
    input wire signed [`Bitwidth_1 : 0] y_init, // 1+4+16
    
    output reg signed [`out_Bitwidth_1 : 0] result, // 1+6+16 
    output reg done
    );

    parameter iter_num = 5'd16;  // cordic depth

    wire flag;      // iteration direction
    reg frac_done;  // exp_fraction done
    reg signed  [`Bitwidth_1 : 0] x, y, z, angle, out; // ???
    reg signed  [`Bitwidth : 0 ] x_tmp;
    reg         [3 : 0] exp_int;        // bit-changed
    reg signed  [`out_Bitwidth_1 : 0] exp_out;

    (*dont_touch = "yes"*) reg [4:0] i;
    
    /** '''''''''''''''''''''''''''''control signal''''''''''''''''''''''''''''' **/
    wire condition; // stop cordic when reach cordic depth
    assign condition = (i >= iter_num) || (!mode & (y <= 'd1)) || (mode & (z <= 'd1)); 
    
    wire flag_exp;  // exp_int done
    assign flag_exp  = (exp_int == 'd0);

    reg en, en_0;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)              en <= 1'd0;
        else if(init)           en <= 1'd1;
        else if(condition)      en <= 1'd0;
        else                    en <= en;
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)                      en_0 <= 1'd0;
        else if(init)                   en_0 <= 1'd1;
        else if(frac_done & flag_exp)   en_0 <= 1'd0;
        else                            en_0 <= en_0;
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)                                      frac_done <= 1'b0;
        else if(mode & condition & en)                  frac_done <= 1'b1;
        else if(mode & frac_done & flag_exp & en_0)     frac_done <= 1'b0;
        else                                            frac_done <= frac_done;
    end
    
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)                                      done <= 1'b0;
        else if((!mode) && condition && en)             done <= 1'b1;
        else if(mode && frac_done && flag_exp && en_0)  done <= 1'b1;
        else                                            done <= 1'b0;
    end


    /** '''''''''''''''''''''''''''''input_sign reg''''''''''''''''''''''''''''' **/
    reg x_sign, y_sign;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            x_sign <= 'd0;
            y_sign <= 'd0;
        end
        else if(init) begin
            x_sign <= x_init[`Bitwidth_1];
            y_sign <= y_init[`Bitwidth_1];
        end
        else begin
            x_sign <= x_sign;
            y_sign <= y_sign;
        end
    end


    /** '''''''''''''''''''''''''''''iteration equation''''''''''''''''''''''''''''' **/
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)
        begin
            x <= `Bitwidth'd0;
            y <= `Bitwidth'd0;
            z <= `Bitwidth'd0;
        end
        else if(init)
            begin
                if(mode) begin
                    x <= `Bitwidth'd65536;                              // to be changed when bitchange
                    y <= `Bitwidth'd0;                     
                    z <= { {5{x_init[`Bitwidth_1]}} , x_init[15:0] };   // to be changed when bitchange
                end
                else begin
                    x <= y_init[`Bitwidth_1] ? ~y_init+1 : y_init;  
                    y <= x_init[`Bitwidth_1] ? ~x_init+1 : x_init;
                    z <= `Bitwidth'd0;                              
                end
            end
        else if(en) begin
            if(flag)
                begin
                    x <= mode? x - (y >>> i) : x;
                    y <= y - (x >>> i);
                    z <= z + angle;
                end
            else
                begin
                    x <= mode? x + (y >>> i) : x;
                    y <= y + (x >>> i);
                    z <= z - angle;
                end
        end
    end

//////////////////////////////find the optimum angle//////////////////////////////////   
    always @(posedge clk or negedge rst_n) begin // only operate when div
        if(!rst_n)                      x_tmp <= 0;
        else if(init & (mode == 1'b0))  x_tmp <= (x_init[`Bitwidth_1] ? ~x_init+1 : x_init) + ((x_init[`Bitwidth_1] ? ~x_init+1 : x_init) >>> 1); 
        else if(en & (mode == 1'b0))    x_tmp <= x + (x >>> 1);
        else                            x_tmp <= 0;
    end

    wire signed [37:0] x_tmp_extend;                            // to be changed
    assign x_tmp_extend = {{13{x_tmp[`Bitwidth]}}, x_tmp};    // to be changed
    
    wire signed [`Bitwidth_1:0] target;
    wire signed [`Bitwidth_1:0] target_abs;
    assign target = mode? z: y;
    assign target_abs = target[`Bitwidth_1] ? ~target+`Bitwidth'd1 : target;    // absolute the rest of angle

    assign flag = mode? target[`Bitwidth_1]: ~target[`Bitwidth_1];  

    wire [16:0] rank; // determine the range of rest angle
    assign rank[16] = (target_abs >= (mode?'d26369  : x_tmp_extend[22:2]));  // to be changed
    assign rank[15] = (target_abs >= (mode?'d12487   : x_tmp_extend[23:3]));
    assign rank[14] = (target_abs >= (mode?'d6168   : x_tmp_extend[24:4]));
    assign rank[13] = (target_abs >= (mode?'d3075   : x_tmp_extend[25:5]));
    assign rank[12] = (target_abs >= (mode?'d1537   : x_tmp_extend[26:6]));
    assign rank[11] = (target_abs >= (mode?'d768    : x_tmp_extend[27:7]));
    assign rank[10] = (target_abs >= (mode?'d384    : x_tmp_extend[28:8]));
    assign rank[9] = (target_abs >=  (mode?'d192    : x_tmp_extend[29:9]));
    assign rank[8] = (target_abs >=  (mode?'d96     : x_tmp_extend[30:10]));
    assign rank[7] = (target_abs >=  (mode?'d48     : x_tmp_extend[31:11]));
    assign rank[6] = (target_abs >=  (mode?'d24     : x_tmp_extend[32:12]));
    assign rank[5] = (target_abs >=  (mode?'d12      : x_tmp_extend[33:13]));
    assign rank[4] = (target_abs >=  (mode?'d6      : x_tmp_extend[34:14]));
    assign rank[3] = (target_abs >=  (mode?'d3      : x_tmp_extend[35:15]));
    assign rank[2] = (target_abs >=  (mode?'d1      : x_tmp_extend[36:16]));
    assign rank[1] = (target_abs >=  (mode?'d0       : x_tmp_extend[37:17]));
    assign rank[0] = (target_abs <   (mode?'d0       : x_tmp_extend[37:17]));                                             

    always @(*) begin
        if(en) begin
        casez(rank)
            17'b1????????????????: i = 5'd1;
            17'b01???????????????: i = 5'd2;
            17'b001??????????????: i = 5'd3;
            17'b0001?????????????: i = 5'd4;
            17'b00001????????????: i = 5'd5;
            17'b000001???????????: i = 5'd6;
            17'b0000001??????????: i = 5'd7;
            17'b00000001?????????: i = 5'd8;
            17'b000000001????????: i = 5'd9;
            17'b0000000001???????: i = 5'd10;
            17'b00000000001??????: i = 5'd11;
            17'b000000000001?????: i = 5'd12;
            17'b0000000000001????: i = 5'd13;
            17'b00000000000001???: i = 5'd14;
            17'b000000000000001??: i = 5'd15;
            17'b0000000000000001?: i = 5'd16;
            17'b00000000000000001: i = 5'd17;
            default: i = 5'd0;
        endcase
        end 
        else         i = 5'd0;
    end

    always @(*) begin
        case(i)
            5'd1:  angle = mode? `Bitwidth'd35999  : (`Bitwidth'b0_0000_1000000000000000); // to be changed
            5'd2:  angle = mode? `Bitwidth'd16738  : (`Bitwidth'b0_0000_0100000000000000);
            5'd3:  angle = mode? `Bitwidth'd8235   : (`Bitwidth'b0_0000_0010000000000000);
            5'd4:  angle = mode? `Bitwidth'd4101   : (`Bitwidth'b0_0000_0001000000000000);
            5'd5:  angle = mode? `Bitwidth'd2048   : (`Bitwidth'b0_0000_0000100000000000); 
            5'd6:  angle = mode? `Bitwidth'd1024   : (`Bitwidth'b0_0000_0000010000000000);
            5'd7:  angle = `Bitwidth'd512;
            5'd8:  angle = `Bitwidth'd256;
            5'd9:  angle = `Bitwidth'd128;
            5'd10: angle = `Bitwidth'd64;
            5'd11: angle = `Bitwidth'd32;
            5'd12: angle = `Bitwidth'd16;
            5'd13: angle = `Bitwidth'd8;
            5'd14: angle = `Bitwidth'd4;
            5'd15: angle = `Bitwidth'd2;
            5'd16: angle = `Bitwidth'd1;
            5'd17: angle = `Bitwidth'd1;
            default: angle = `Bitwidth'd35999;
        endcase
    end

//////////////////////revise the length of exp rotation///////////////////
    reg [4:0] index_mem;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)     index_mem <= 5'd0;
        else if(init)   index_mem <= 5'd0;
        else begin
                case (i)
                    4'd1: index_mem <= index_mem + 5'b00001;
                    4'd2: index_mem <= index_mem + 5'b00100;
                    4'd3: index_mem <= index_mem + 5'b01000;
                    4'd4: index_mem <= index_mem + 5'b10000;
                    default: index_mem <= index_mem;
                endcase
        end
    end

    always @(*) begin
        if (mode && condition)
            case (index_mem) // scale factor in exp mode
                        5'd1:  out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 5) - ((x + y) >>> 9) + ((x + y) >>> 12) + ((x + y) >>> 13) + ((x + y) >>> 15);
                        5'd4:  out <= (x + y) + ((x + y) >>> 5) + ((x + y) >>> 10) + ((x + y) >>> 11) + ((x + y) >>> 14) + ((x + y) >>> 16);
                        5'd5:  out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 4) + ((x + y) >>> 8) + ((x + y) >>> 10) + ((x + y) >>> 13) + ((x + y) >>> 15) + ((x + y) >>> 16);
                        5'd8:  out <= (x + y) + ((x + y) >>> 7) + ((x + y) >>> 14) + ((x + y) >>> 15);
                        5'd9:  out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 5) + ((x + y) >>> 7) - ((x + y) >>> 12);
                        5'd10: out <= (x + y) + ((x + y) >>> 2) + ((x + y) >>> 4) + ((x + y) >>> 5) + ((x + y) >>> 13);
                        5'd12: out <= (x + y) + ((x + y) >>> 5) + ((x + y) >>> 7) + ((x + y) >>> 9) - ((x + y) >>> 14);
                        5'd13: out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 4) + ((x + y) >>> 6) - ((x + y) >>> 9) + ((x + y) >>> 11) + ((x + y) >>> 12);
                        5'd16: out <= (x + y) + ((x + y) >>> 9);
                        5'd17: out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 5) + ((x + y) >>> 11) + ((x + y) >>> 12) - ((x + y) >>> 14);
                        5'd20: out <= (x + y) + ((x + y) >>> 5) + ((x + y) >>> 8) - ((x + y) >>> 11) + ((x + y) >>> 13) + ((x + y) >>> 16);
                        5'd21: out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 4) + ((x + y) >>> 7) - ((x + y) >>> 11) + ((x + y) >>> 14);
                        5'd24: out <= (x + y) + ((x + y) >>> 7) + ((x + y) >>> 9) + ((x + y) >>> 13) - ((x + y) >>> 16);
                        5'd25: out <= (x + y) + ((x + y) >>> 3) + ((x + y) >>> 5) + ((x + y) >>> 7) + ((x + y) >>> 9) + ((x + y) >>> 14) + ((x + y) >>> 16);
//                        5'd1:  out <= (x + y) * 1.154693604;
//                        5'd4:  out <= (x + y) * 1.032791138;
//                        5'd5:  out <= (x + y) * 1.19255732;
//                        5'd8:  out <= (x + y) * 1.007904053;
//                        5'd9:  out <= (x + y) * 1.163820363;
//                        5'd10: out <= (x + y) * 1.343855928;
//                        5'd12: out <= (x + y) * 1.040954373;
//                        5'd13: out <= (x + y) * 1.201983356;
//                        5'd16: out <= (x + y) * 1.001953125;
//                        5'd17: out <= (x + y) * 1.156948864;
//                        5'd20: out <= (x + y) * 1.034808308;
//                        5'd21: out <= (x + y) * 1.194886534;
//                        5'd24: out <= (x + y) * 1.009872615;
//                        5'd25: out <= (x + y) * 1.166093449;
                        default: out <= x + y;
                    endcase
//            out = (x + y) 
//            +((index_mem == 5'd10)?((x + y) >>> 2):0)
//            +((index_mem == 5'd1 || index_mem == 5'd5  || index_mem == 5'd9  || index_mem == 5'd13 || index_mem == 5'd17 || index_mem == 5'd21 || index_mem == 5'd25)?((x + y)  >>> 3):0)
//            +((index_mem == 5'd5 || index_mem == 5'd10 || index_mem == 5'd13 || index_mem == 5'd21)?((x + y)  >>> 4):0)
//            +((index_mem == 5'd1 || index_mem == 5'd4  || index_mem == 5'd9  || index_mem == 5'd10 || index_mem == 5'd12 || index_mem == 5'd17 || index_mem == 5'd20 || index_mem == 5'd25)?((x + y)  >>> 5):0)
//            +((index_mem == 5'd13)?((x + y)  >>> 6):0)
//            +((index_mem == 5'd8 || index_mem == 5'd9  || index_mem == 5'd12 || index_mem == 5'd21 || index_mem == 5'd24 || index_mem == 5'd25)?((x + y)  >>> 7):0)
//            +((index_mem == 5'd5 || index_mem == 5'd20)?((x + y)  >>> 8):0)
//            +((index_mem == 5'd12|| index_mem == 5'd16 || index_mem == 5'd24 || index_mem == 5'd25)?((x + y)  >>> 9):0)
//            +((index_mem == 5'd4 || index_mem == 5'd5)?((x + y)  >>> 10):0)
//            +((index_mem == 5'd4 || index_mem == 5'd13 || index_mem == 5'd17)?((x + y)  >>> 11):0)
//            +((index_mem == 5'd1 || index_mem == 5'd13 || index_mem == 5'd17)?((x + y)  >>> 12):0)
//            +((index_mem == 5'd1 || index_mem == 5'd5  || index_mem == 5'd10 || index_mem == 5'd20 || index_mem == 5'd24)?((x + y)  >>> 13):0)
//            +((index_mem == 5'd4 || index_mem == 5'd8  || index_mem == 5'd21 || index_mem == 5'd25)?((x + y)  >>> 14):0)
//            +((index_mem == 5'd1 || index_mem == 5'd5  || index_mem == 5'd8)?((x + y)  >>> 15):0)
//            +((index_mem == 5'd4 || index_mem == 5'd5  || index_mem == 5'd20 || index_mem == 5'd25)?((x + y)  >>> 16):0)
//            -((index_mem == 5'd1 || index_mem == 5'd13)?((x + y)  >>> 9):0)
//            -((index_mem == 5'd20|| index_mem == 5'd21)?((x + y)  >>> 11):0)
//            -((index_mem == 5'd9)?((x + y)  >>> 12):0)
//            -((index_mem == 5'd12|| index_mem == 5'd17)?((x + y)  >>> 14):0)
//            -((index_mem == 5'd24)?((x + y)  >>> 16):0);
//            out = (x + y) 
//            +((index_mem == 4'b1010)?((x + y) >>> 2):0)
//            +((index_mem == 4'b1101 || index_mem == 4'b0101 || index_mem == 4'b1001 || index_mem == 4'b0001)?((x + y)  >>> 3):0)
//            +((index_mem == 4'b1101 || index_mem == 4'b0101 || index_mem == 4'b1010)?((x + y)  >>> 4):0)
//            +((index_mem == 4'b1001 || index_mem == 4'b0100 || index_mem == 4'b1100)?((x + y)  >>> 5):0)
//            +((index_mem == 4'b0001 || index_mem == 4'b1010)?((x + y)  >>> 6):0)
//            +((index_mem == 4'b1101 || index_mem == 4'b1001 || index_mem == 4'b1100 || index_mem == 4'b0001 || index_mem == 4'b1010 || index_mem == 4'b1000)?((x + y)  >>> 7):0)
//            +((index_mem == 4'b1101 || index_mem == 4'b1001 || index_mem == 4'b0101 || index_mem == 4'b0001 || index_mem == 4'b1010)?((x + y)  >>> 8):0)
//            +((index_mem == 4'b1001 || index_mem == 4'b0001 || index_mem == 4'b1010)?((x + y)  >>> 9):0)
//            +((index_mem == 4'b1101 || index_mem == 4'b1001 || index_mem == 4'b1100 || index_mem == 4'b0100 || index_mem == 4'b1010)?((x + y)  >>> 10):0);
        else
            out = `Bitwidth'd0;
    end

/*'''''''''''''''''''''''''''''''''''''exp_int'''''''''''''''''''''''''''''''''''''*/
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)   begin
            exp_out <= 'd0;
            exp_int <= 'd0;
        end
        else if(mode) begin
            if(init) begin
                exp_out <= 'd0;
                exp_int <= x_init[`Bitwidth_1] ? ~(x_init[`Bitwidth_2 : `fra_Bitwidth]) : x_init[`Bitwidth_2 : `fra_Bitwidth];
            end
            else if(condition) begin
                exp_out <= {3'b0, out};
                exp_int <= exp_int;
            end
            else if(frac_done) begin
                if(!flag_exp) begin
                    if(x_sign) begin
//                        exp_out <= exp_out * 0.3678794412;
                        exp_out <= (exp_out>>>2) + (exp_out>>>3) - (exp_out>>>7) + (exp_out>>>11) + (exp_out>>>13) + (exp_out>>>16) + (exp_out>>>18) + (exp_out>>>20) + (exp_out>>>21);  // 1/e = 0.36787944 = 0.3676757813
                        exp_int <= exp_int - 1;
                    end
                    else begin
//                        exp_out <= exp_out * 2.718281828;
                        exp_out <= (exp_out<<<1) + (exp_out>>>1) + (exp_out>>>3) + (exp_out>>>4) + (exp_out>>>5) - (exp_out>>>11) + (exp_out>>>16) + (exp_out>>>18) + (exp_out>>>20); // e = 2.71828183 = 2.718261719
                        exp_int <= exp_int - 1;
                    end
                end
                else begin
                    exp_out <= exp_out;
                    exp_int <= exp_int;
                end
            end
            else begin
                exp_out <= exp_out;
                exp_int <= exp_int;
            end
        end
        else begin
            exp_out <= 'd0;
            exp_int <= 'd0;
        end
    end

/*'''''''''''''''''''''''''''''''''''''output'''''''''''''''''''''''''''''''''''''*/

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)                                      result <= 'd0;
        else if(!mode & condition)                      result <= (x_sign ^ y_sign) ? {3'b1,~z[19:0]+1} : {3'b0,z[19:0]}; // div
        else if(mode && frac_done && flag_exp && en_0)  result <= exp_out;  // exp
        else                                            result <= 'd0;
    end

endmodule