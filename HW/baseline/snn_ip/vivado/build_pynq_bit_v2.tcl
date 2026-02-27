set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]

set proj_name "snn_pynq_v2"
set proj_dir [file join $script_dir $proj_name]
set out_dir [file join $script_dir output_v2]
set ip_repo [file join $root_dir snn_top_prj_v2_conv2 solution_v2_conv2 impl ip]

file mkdir $out_dir

create_project $proj_name $proj_dir -part xczu7ev-ffvc1156-2-e -force
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list [file normalize $ip_repo]] [current_project]
update_ip_catalog

set snn_ipdefs [get_ipdefs -all *:hls:snn_top:*]
if {[llength $snn_ipdefs] == 0} {
    error "Cannot find HLS IP snn_top in IP catalog. Check ip_repo path: $ip_repo"
}
set snn_vlnv [lindex $snn_ipdefs end]
puts "Using HLS IP: $snn_vlnv"

create_bd_design "design_1"

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps8_0_99M
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
create_bd_cell -type ip -vlnv $snn_vlnv snn_top_0
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0
create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_0

set_property -dict [list \
    CONFIG.NUM_SI {3} \
    CONFIG.NUM_MI {2} \
] [get_bd_cells smartconnect_0]

set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
] [get_bd_cells axi_dma_0]

# 256 KB BRAM for DMA source/sink buffers
set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Write_Width_A {32} \
    CONFIG.Read_Width_A {32} \
    CONFIG.Write_Depth_A {65536} \
    CONFIG.Write_Width_B {32} \
    CONFIG.Read_Width_B {32} \
] [get_bd_cells blk_mem_gen_0]

# Clock/reset
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins rst_ps8_0_99M/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_ps8_0_99M/ext_reset_in]

foreach p {zynq_ultra_ps_e_0/maxihpm0_lpd_aclk smartconnect_0/aclk axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk axi_dma_0/m_axi_s2mm_aclk axi_bram_ctrl_0/s_axi_aclk snn_top_0/ap_clk} {
    connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins $p]
}

foreach p {smartconnect_0/aresetn axi_dma_0/axi_resetn axi_bram_ctrl_0/s_axi_aresetn snn_top_0/ap_rst_n} {
    connect_bd_net [get_bd_pins rst_ps8_0_99M/peripheral_aresetn] [get_bd_pins $p]
}

# Keep SNN running: ap_start tied high
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_ap_start
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] [get_bd_cells const_ap_start]
connect_bd_net [get_bd_pins const_ap_start/dout] [get_bd_pins snn_top_0/ap_start]

# AXI memory-mapped fabric
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins smartconnect_0/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# BRAM connection
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]

# AXI Stream path
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins snn_top_0/in_stream]
connect_bd_intf_net [get_bd_intf_pins snn_top_0/out_stream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
validate_bd_design
save_bd_design

make_wrapper -files [get_files ${proj_dir}/${proj_name}.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ${proj_dir}/${proj_name}.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

open_run impl_1
set bit_file [get_property BITSTREAM.FILE [current_design]]
set hwh_file "${proj_dir}/${proj_name}.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"

if {$bit_file eq ""} {
    set bit_file "${proj_dir}/${proj_name}.runs/impl_1/design_1_wrapper.bit"
}

if {![file exists $bit_file]} {
    error "Bitstream file not found: $bit_file"
}
if {![file exists $hwh_file]} {
    error "HWH file not found: $hwh_file"
}

file copy -force $bit_file "${out_dir}/snn_v2.bit"
file copy -force $hwh_file "${out_dir}/snn_v2.hwh"

puts "BIT: ${out_dir}/snn_v2.bit"
puts "HWH: ${out_dir}/snn_v2.hwh"

close_project
