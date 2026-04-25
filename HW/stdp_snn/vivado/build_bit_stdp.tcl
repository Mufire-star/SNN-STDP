set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]

set proj_name "snn_pynq_stdp"
set proj_dir [file join $script_dir $proj_name]
set out_dir [file join $script_dir output_stdp]
set ip_repo [file join $root_dir snn_top_prj_stdp solution_stdp impl ip]

if {[info exists ::env(VIVADO_JOBS)] && $::env(VIVADO_JOBS) ne ""} {
    set build_jobs $::env(VIVADO_JOBS)
} else {
    set build_jobs 1
}

catch {set_param general.maxThreads $build_jobs}

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

# ====== 1) Configure PS: Enable M_AXI_HPM0_FPD and S_AXI_HP0_FPD ======
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]

# Enable M_AXI_HPM0_FPD for PS to access PL peripherals (DMA/BRAM control)
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
] [get_bd_cells zynq_ultra_ps_e_0]

# Enable S_AXI_HP0_FPD for DMA to access DDR memory (KEY FIX for DMADecErr)
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
] [get_bd_cells zynq_ultra_ps_e_0]

# ====== 2) Create all IP blocks ======
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps8_0_99M
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
create_bd_cell -type ip -vlnv $snn_vlnv snn_top_0

# Control interconnect: PS -> DMA control registers only
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_ctrl
set_property -dict [list \
    CONFIG.NUM_SI {1} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_interconnect_ctrl]

# Data interconnect: DMA -> PS DDR (KEY FIX for DMADecErr)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_data
set_property -dict [list \
    CONFIG.NUM_SI {2} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_interconnect_data]

# ====== 3) Configure DMA ======
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
] [get_bd_cells axi_dma_0]

# ====== 4) Connect clocks and resets ======
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins rst_ps8_0_99M/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_ps8_0_99M/ext_reset_in]

# Connect all IP clocks to pl_clk0 (100MHz)
foreach p { \
    zynq_ultra_ps_e_0/maxihpm0_fpd_aclk \
    zynq_ultra_ps_e_0/maxihpm0_lpd_aclk \
    zynq_ultra_ps_e_0/saxihp0_fpd_aclk \
    axi_interconnect_ctrl/ACLK \
    axi_interconnect_ctrl/S00_ACLK \
    axi_interconnect_ctrl/M00_ACLK \
    axi_interconnect_data/ACLK \
    axi_interconnect_data/S00_ACLK \
    axi_interconnect_data/S01_ACLK \
    axi_interconnect_data/M00_ACLK \
    axi_dma_0/s_axi_lite_aclk \
    axi_dma_0/m_axi_mm2s_aclk \
    axi_dma_0/m_axi_s2mm_aclk \
    snn_top_0/ap_clk \
} {
    connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins $p]
}

# Connect peripheral resets
foreach p { \
    axi_interconnect_ctrl/ARESETN \
    axi_interconnect_ctrl/S00_ARESETN \
    axi_interconnect_ctrl/M00_ARESETN \
    axi_interconnect_data/ARESETN \
    axi_interconnect_data/S00_ARESETN \
    axi_interconnect_data/S01_ARESETN \
    axi_interconnect_data/M00_ARESETN \
    axi_dma_0/axi_resetn \
    snn_top_0/ap_rst_n \
} {
    connect_bd_net [get_bd_pins rst_ps8_0_99M/peripheral_aresetn] [get_bd_pins $p]
}

# ====== 5) Keep SNN always running ======
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_ap_start
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] [get_bd_cells const_ap_start]
connect_bd_net [get_bd_pins const_ap_start/dout] [get_bd_pins snn_top_0/ap_start]

# ====== 6) AXI Control Path: PS -> DMA ======
# PS M_AXI_HPM0_FPD -> interconnect -> DMA S_AXI_LITE
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins axi_interconnect_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M00_AXI] \
                    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# ====== 7) AXI Data Path: DMA -> PS DDR (KEY FIX for DMADecErr) ======
# DMA M_AXI_MM2S/S2MM -> interconnect -> PS S_AXI_HP0_FPD
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
                    [get_bd_intf_pins axi_interconnect_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
                    [get_bd_intf_pins axi_interconnect_data/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_data/M00_AXI] \
                    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

# ====== 8) AXI Stream: DMA <-> SNN ======
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
                    [get_bd_intf_pins snn_top_0/in_stream]
connect_bd_intf_net [get_bd_intf_pins snn_top_0/out_stream] \
                    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# ====== 9) Address assignment and validation ======
assign_bd_address

# Verify critical address mappings:
# - zynq_ultra_ps_e_0/Data(M_AXI_HPM0_FPD) should see:
#   * axi_dma_0/S_AXI_LITE (typically 0x8000_0000 range)
# - axi_dma_0/Data_MM2S(M_AXI_MM2S) should see:
#   * zynq_ultra_ps_e_0/SAXIGP2 (0x0000_0000 - DDR memory)
# - axi_dma_0/Data_S2MM(M_AXI_S2MM) should see:
#   * zynq_ultra_ps_e_0/SAXIGP2 (0x0000_0000 - DDR memory)

puts "\n========== Address Map =========="
puts "Check Address Editor in Vivado GUI to verify:"
puts "1. PS M_AXI_HPM0_FPD can access DMA @ 0x8000_0000"
puts "2. DMA M_AXI_MM2S can access DDR @ 0x0000_0000 - 0x7FFF_FFFF"
puts "3. DMA M_AXI_S2MM can access DDR @ 0x0000_0000 - 0x7FFF_FFFF"
puts "=================================\n"

validate_bd_design
save_bd_design

make_wrapper -files [get_files ${proj_dir}/${proj_name}.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ${proj_dir}/${proj_name}.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

launch_runs impl_1 -to_step write_bitstream -jobs $build_jobs
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

file copy -force $bit_file "${out_dir}/snn_stdp.bit"
file copy -force $hwh_file "${out_dir}/snn_stdp.hwh"

puts "BIT: ${out_dir}/snn_stdp.bit"
puts "HWH: ${out_dir}/snn_stdp.hwh"

close_project
