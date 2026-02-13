#!/bin/sh   

#Set Output Folder 
#Output_folder_path="tmp_$(date +%Y%m%d)"

avoid_interval=1   #mounth

# #########################   72_to_72   #############################
Output_folder_path="ID42_P(obs_WRF_CMAQ)+F(WRF_CMAQ)(JAN-Identity)"
mkdir -p "$Output_folder_path"

site_sets="[42]"

Input_timesteps=72

Output_timesteps=72

python origin_datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python origin_train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python origin_inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 


