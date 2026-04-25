open_project -reset snn_top_prj_stdp
set_top snn_top
add_files snn_top.cpp
add_files snn_top.h
add_files -tb snn_top_tb.cpp
open_solution -reset solution_stdp
set_part xczu7ev-ffvc1156-2-e
if {[info exists ::env(HLS_CLOCK_NS)] && $::env(HLS_CLOCK_NS) ne ""} {
    set hls_clock_ns $::env(HLS_CLOCK_NS)
} else {
    set hls_clock_ns 10.0
}
create_clock -period $hls_clock_ns -name default
if {![info exists ::env(HLS_SKIP_CSIM)] || $::env(HLS_SKIP_CSIM) ne "1"} {
    csim_design
} else {
    puts "Skipping csim_design because HLS_SKIP_CSIM=1"
}
csynth_design
export_design -format ip_catalog -rtl verilog -version 1.0.0
exit
