import sys 
import json 
import pandas as pd 
from nsys_recipe import data_service, log 
from nsys_recipe.data_service import DataService 
 
# To run this script, be sure to add the Nsight Systems package directory to your PYTHONPATH, similar to this: 
# export PYTHONPATH=/opt/nvidia/nsight-systems/2023.4.1/target-linux-x64/python/packages 


# export PYTHONPATH=~/packages
# export NSYS_DIR=/opt/nvidia/nsight-systems/2023.4.4
 
def compute_utilization(filename, freq=10000): 
    service=DataService(filename) 
    table_column_dict = { 
        "GPU_METRICS": ["typeId", "metricId", "value"], 
        "TARGET_INFO_GPU_METRICS": ["metricName", "metricId"], 
        "META_DATA_CAPTURE": ["name", "value"] 
    } 
    hints={"format":"sqlite"} 
    df_dict = service.read_tables(table_column_dict, hints=hints) 
    df = df_dict.get("GPU_METRICS", None) 
    if df is None: 
        print(f"{filename} does not contain GPU metric data.") 
        return
    tgtinfo_df = df_dict.get("TARGET_INFO_GPU_METRICS", None) 
    if tgtinfo_df is None: 
        print(f"{filename} does not contain TARGET_INFO_GPU_METRICS table.") 
        return
    metadata_df = df_dict.get("META_DATA_CAPTURE", None) 
    if metadata_df is not None: 
        if "GPU_METRICS_OPTIONS:SAMPLING_FREQUENCY" in metadata_df['name'].values: 
            report_freq = metadata_df.loc[ metadata_df['name']=='GPU_METRICS_OPTIONS:SAMPLING_FREQUENCY']['value'].iat[0] 
            if isinstance(report_freq, (int,float)): 
                freq = report_freq 
                print("Setting GPU Metric sample frequency to value in report file. new frequency=",freq)  

    possible_smactive=['SMs Active', 'SM Active', 'SM Active [Throughput %]'] 
    smactive_name_mask = tgtinfo_df['metricName'].isin(possible_smactive) 
    smactive_row = tgtinfo_df[smactive_name_mask] 
    smactive_name = smactive_row['metricName'].iat[0] 
    smactive_id = tgtinfo_df.loc[tgtinfo_df['metricName']==smactive_name,'metricId'].iat[0] 
    smactive_df = df.loc[ df['metricId'] == smactive_id ] 
      
    usage = smactive_df['value'].sum() 
    count = len(smactive_df['value']) 
    count_nonzero = len(smactive_df.loc[smactive_df['value']!=0]) 
    count_zero = len(smactive_df.loc[smactive_df['value']==0]) 
    count_100 = len(smactive_df.loc[smactive_df['value']==100])
    avg_gross_util = usage/count 
    avg_net_util = usage/count_nonzero 
    effective_util = usage/freq/100
      
    #print(f"Avg gross GPU utilization:\t%lf %%" % avg_gross_util) 
    #print(f"Avg net GPU utilization:\t%lf %%" % avg_net_util) 
    #print(f"Effective GPU utilization time:\t%lf s" % effective_util) 
    #print("Number of times where 100% SMs were active: ", count_100)
    #print("Number of times where 0% SMs were active: ", count_zero)
    
    smactive = smactive_df.loc[smactive_df['value']!=0]
    smactive_index = smactive.index.to_list()

    start_loc = smactive_index[0]
    end_loc = smactive_index[-1]
    ingestion_smactive = smactive_df.loc[list(range(start_loc, end_loc + 1, 14))]
    ingestion_len = len(ingestion_smactive.index)
    print("Total captured freq for ingestion: ", ingestion_len)
    print(f"Avg net GPU utilization:\t%lf %%" % (usage / ingestion_len)) 
    print("Number of times where 100% SMs were active: ", len(ingestion_smactive.loc[ingestion_smactive['value']==100]))
    print("Number of times where 0% SMs were active: ", len(ingestion_smactive.loc[ingestion_smactive['value']==0]) )

    return metadata_df 
      
if __name__ == '__main__': 
    if len(sys.argv)==2: 
        compute_utilization(sys.argv[1]) 
    elif len(sys.argv)==3: 
        compute_utilization(sys.argv[1], freq=float(sys.argv[2])) 