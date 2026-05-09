open_project -reset projection_out_prj

add_files projection.cpp
add_files -tb projection_test.cpp
set_top projection_out

open_solution -reset solution1
set_part {xcvu37p-fsvh2892-2L-e}
create_clock -period "75MHz"

csim_design

exit
