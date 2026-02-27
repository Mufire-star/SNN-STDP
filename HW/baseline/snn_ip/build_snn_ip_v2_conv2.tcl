open_project -reset snn_top_prj_v2_conv2
set_top snn_top
add_files snn_top.cpp
add_files snn_top.h
add_files weights/weights_generated.h
add_files -tb snn_top_tb.cpp
open_solution -reset solution_v2_conv2
set_part xczu7ev-ffvc1156-2-e
create_clock -period 5.0 -name default
csim_design
csynth_design
export_design -format ip_catalog -version 2.0.0
exit
