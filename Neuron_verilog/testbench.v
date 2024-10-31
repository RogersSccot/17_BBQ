`timescale 1ps/1ps

module test;

parameter Bitwidth = 27;
integer file_out;
reg clk, rstn, mode;
wire signed [26 : 0] v,h;
wire done;

Dual_mode_neuron test(
    .clk(clk),
    .rstn(rstn),
    .mode(mode),    // 1:HH 2:AdEx

    .v(v),
    .h_0(h),
    .done(done)
);

initial begin

    file_out = $fopen("v_HH_80.txt");
    if (!file_out) begin
        $display("can't open file");
        $finish;
    end
    
//    file_out = $fopen("h_hardware.txt");
//    if (!file_out) begin
//        $display("can't open file");
//        $finish;
//    end
    
    clk = 0;
    rstn = 0;
    mode = 1;   // change this to change mode
    #10 rstn = 1;
//    #2000000 mode = 0;
end

always #5 clk = ~ clk;

wire signed [26 : 0] dout_s = v;
wire rst_write = clk & rstn;         //复位期间不应写入数据
reg done_1;
always @ (posedge clk)
    done_1 <= done;
    
always @ (posedge rst_write)
    if (done)   begin
        $fdisplay(file_out, "%d", dout_s);
//        $fdisplay(file_out, "%d", h);
    end

endmodule