#!pip install wget
#pip install urllib
import os
import pandas as pd
import re
import numpy as np
from urllib import request
import gzip
import multiprocessing

out_file = 'D:/documents_andres/CPT_downloads/pruebita.tsv.gz'
url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.OCNF/.surface/.TMP/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.OCNF/.surface/.TMP/appendstream/S/%280000%201%20Dec%201982-2022%29VALUES/L/%281.5%29VALUES/M/1/24/RANGE%5BM%5Daverage/Y/%2812%29%28-12%29RANGEEDGES/X/%28258%29%28333%29RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BS/L/add%5Dcptv10.tsv.gz"


def download_CFSV2_CPT_1(first_year,last_year,i_month,ic,dir_save,area1,lg):
    """Function to downloas SST CSFV2 from CPT data servers

    Parameters:
    first_year (numeric): first year for data to download 
    i_month (int): Initial month for data download.
    lg (int): Length of season to average.
    ic (int): Month forecast were initialized (Lead Time month number)
    last_year (int): Last years for data to download (numeric).
    dir_save (character):  full file path where to save data .
    area1 (array): Numeric vector for area sub-domain.
    Returns:
    str: SST CSFV2 data downloaded and unzipped from CPT data servers

    """
    month_abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    lg_s = lg -1 # length of season
    lead = i_month-ic #lead time raw
    if lead<0 :
        lead     = lead + 12  
        last_year=last_year-1
    month_lab = month_abb[ic] #month abbreviated name
    full_period_length = lead+lg_s #full period length  

    if lg_s == 0:
        route = f"http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.OCNF/.surface/.TMP/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.OCNF/.surface/.TMP/appendstream/S/%280000%201%20{month_lab}%20{first_year}-{last_year}%29VALUES/L/%28{lead}.5%29VALUES/M/1/24/RANGE%5BM%5Daverage/Y/%28{area1[3]}%29%28{area1[2]}%29RANGEEDGES/X/%28{area1[0]}%29%28{area1[1]}%29RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BS/L/add%5Dcptv10.tsv.gz"
    else:
        route = f"http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.OCNF/.surface/.TMP/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.OCNF/.surface/.TMP/appendstream/S/%280000%201%20{month_lab}%20{first_year}-{last_year}%29VALUES/L/{lead}.5/{full_period_length}.5/RANGE%5BL%5D//keepgrids/average/M/1/24/RANGE%5BM%5Daverage/Y/%28{area1[3]}%29%28{area1[2]}%29RANGEEDGES/X/%28{area1[0]}%29%28{area1[1]}%29RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BS/L/add%5Dcptv10.tsv.gz"

    
    
    trimestrel = list(range((ic+lead),(ic+lead+lg_s+1), 1) )

    if sum([i > 12 for i in trimestrel] )>0 :
        for k in range(len(trimestrel)):
            if trimestrel[k] > 12:
                trimestrel[k] = trimestrel[k]-12
        
    if len(trimestrel)>1:

        path_save = dir_save+"/"+month_abb[ic]+"_"+"-".join([month_abb[i] for i in trimestrel])+".tsv.gz"


    else:
        path_save = dir_save+"/"+month_abb[ic]+ "_"+month_abb[trimestrel[0]]+".tsv.gz"



    unzipped_fl_name = path_save.replace(".gz", "")
    responsex = request.urlretrieve(route ,path_save)
    
    op = open(unzipped_fl_name,"w") 

    with gzip.open(path_save,"rb") as ip_byte:
        op.write(ip_byte.read().decode("utf-8"))
        op.close()
        ip_byte.close()
    
    os.remove(path_save)


    return print('successful download: '+unzipped_fl_name)


download_CFSV2_CPT_1(
first_year = 1982,
last_year = 2014,
i_month = 1,
ic = 11,
dir_save =  "D:/documents_andres/CPT_downloads",
area1 = [180, 350, -20, 40],
lg = 1,
)
2+2