#pip install monthdelta
import os
import pandas as pd
import re
import glob
import datetime
from datetime import date
from monthdelta import monthdelta
import json
import numpy as np
import gzip
import shutil

def read_Files(ruta, skip):
    
    dataframe = pd.read_csv(ruta, sep='/t', engine='python', header=None)  # * El separador debe ser '\t' para indicar una tabulación
    
    idx = np.where(dataframe.iloc[:,0].str.find("cpt:T") != -1)

    idx = [x+1 for x in idx]

    dataframe.iloc[idx[0]] = "\t"+dataframe.iloc[idx[0]]
    dataframe_ajustado=dataframe.iloc[:,0].str.split('\t', expand=True)
    dataframe_ajustado = dataframe_ajustado.drop(range(skip)).reset_index(drop=True)
    
    return dataframe_ajustado

def get_cpt_dates(df):
    """
    Function to get dates from CPT input files

    Parameters:
     df (pandas.DataFrame): input CPT file loaded used read_Files function

    Returns:
    pandas.Serie with formatted dates
    """
    df = df.drop(range(2))
    years = df.iloc[0,1:].str.replace("(T[0-9]{2}:[0-9]{2})", "")
    if all(years.isnull()):
        pos = np.where(df.iloc[:,0].str.contains("cpt:T"))[0] 
        tms = df.iloc[pos, 0]
        #tms <- gsub("/", "-", tms)
        tms  = tms.str.extract("cpt:T=([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{2}/[0-9]{2}|[0-9]{4}-[0-9]{2}/[0-9]{4}-[0-9]{2})").reset_index(drop = True).squeeze()
        #algunos archivos tienen dos anos e.g Nov_Dec-Jan va a tener 2013(Dic) y 2014(Jan)
        cond = tms.str.contains(r"[0-9]{4}-[0-9]{2}/[0-9]{4}-[0-9]{2}")
        if any(cond):
            to_change =  tms.iloc[np.where(cond)[0]].str.extract(r"(/[0-9]{4}-[0-9]{2})").reset_index(drop = True).squeeze()
            tms.iloc[np.where(cond)[0]] =    to_change.str.replace("/", "")+"-01" 
        years = tms 
    
    
    return years.iloc[np.where(~years.isnull())[0]]


def files_y(y_d, names):
    """ Function to convert station precipitation file to CPT data format and write it to tsv define file

    Parameters:
    y_d (pandas dataFrame): Dataframe with station precipitation data
    names (str): name of the station
    Returns:
    It's write CPT formatted tsv file in defined directory and returns the first 8 characters of the
    names of sample station points 
    """
  
    #y_d = data_y[list(data_y.keys())[1]]
    #names = list(data_y.keys())[1]
    print(">>>> procesando: "+names)
    y_d["month"] = ["{:02d}".format(x) for x in y_d["month"]]
    y_m = y_d.apply(lambda x: str(int(x["year"]))+ '-' + x["month"], axis = 1)
    data = y_d.drop(labels = ["year", "month"], axis = 1)
    p1="cpt:field=prcp, cpt:nrow="+str(data.shape[0])+", cpt:ncol="+str(data.shape[1])+", cpt:row=T, cpt:col=index, cpt:units=mm, cpt:missing=-999.000000000"
    data.insert(0, " ", y_m)

    p="xmlns:cpt=http://iri.columbia.edu/CPT/v10/"
    #os.makedirs(os.path.join(main_dir,"run_CPT",names), exist_ok=True)
    name_file= os.path.join(main_dir,"run_CPT",names,"y_"+names+".txt")  

    with open(name_file, "w") as fl:
        fl.write(p)
        fl.write("\n")
        fl.write("cpt:nfields=1")
        fl.write("\n")
        fl.write(p1)
        fl.write("\n")
        fl.write("\t".join(data.columns))
        fl.write("\n")
        for i in range(data.shape[0]):
            fl.write("\t".join([str(x) for x in data.iloc[i]]))
            fl.write("\n")
    return([x[0:8] for x in data.columns if x != " "])

def load_json(pth):
    """Function to read input params json file
    Parameters:
    pth (str): full path to input params json file

    Returns:
    dict object with initial params for processing
    """
    with open(pth, 'r') as js:
        to_ret = json.load(js)
    return(to_ret[0])

def get_season_months(season_type, month, month_abb):
    """Function to generate month season labs for folders names based on season type 
    Parameters:
    season_type (str): text identifier for season, extracted from imput params json file
    month (int): Month number of inital month
    month_abb (list): Year months three letter abbreviation

    Returns:
    list with the first letter of each month for each season
    """

    mnths = [month+x for x in [0,1,2,3,4,5]]
    mnths = [x - 12 if x>12 else x for x in mnths]
    if season_type == 'tri':
        tri1 =  "".join([month_abb[x-1][0] for x in mnths[0:3]])
        tri2 =  "".join([month_abb[x-1][0] for x in mnths[3:]])
        to_ret = [tri1, tri2]
    elif season_type == "bi":
        bi1 =  "".join([month_abb[x-1][0] for x in mnths[0:2]])
        bi2 =  "".join([month_abb[x-1][0] for x in mnths[2:4]])
        bi3 =  "".join([month_abb[x-1][0] for x in mnths[4:]])
        to_ret = [bi1, bi2, bi3]
    else:
        raise ValueError("Invalid seaso type")
        
    return(to_ret)

def get_season_years(season_type, month, year):
    """Function to generate year season
    Parameters:
    season_type (str): text identifier for season, extracted from imput params json file
    month (int): Month number of inital month
    year (int): Year of system date

    Returns:
    list with season years
    """
    mnths = [month+x for x in [0,1,2,3,4,5]]
    years = [year if x <= 12 else year+1 for x in mnths]
    if season_type == 'tri':
        to_ret =  [years[x] for x in [0,3]]
    elif season_type == "bi":
        to_ret =  [years[x] for x in [0,2,4]]
    else:
        raise ValueError("Invalid seaso type")
        
    return(to_ret)
    

######### Run #################
start_time = date.today()
#options(timeout=180)
##############################

#define some global variables (some paths should be already defined in runMain so may not be necesary here)
month_abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
root_dir = os.path.join("D:"+os.sep, "documents_andres", "pr_1", "Colombia","inputs")
main_dir  = os.path.join(root_dir, "prediccionClimatica")
path_dpto = os.path.join(main_dir, 'estacionesMensuales')#dir_response
dir_save  = os.path.join(main_dir, "descarga") #paste0(dirPrediccionInputs, "descarga", sep = "", collapse = NULL)
os.makedirs(os.path.join(main_dir, "run_CPT"), exist_ok=True)
os.makedirs(dir_save, exist_ok=True)
#dirCurrent <- paste0(dirWorkdir, currentCountry, "/")
#dirInputs <- paste0(dirCurrent, "inputs/", sep = "", collapse = NULL)
#main_dir = "paste0(dirInputs, "prediccionClimatica/", sep = "", collapse = NULL)"
#dir_response <- paste0(dirPrediccionInputs, "estacionesMensuales", sep = "", collapse = NULL)
#dir_runCPT <- paste0(dirPrediccionInputs, "run_CPT", sep = "", collapse = NULL)
#dir_response <- paste0(dirPrediccionInputs, "estacionesMensuales", sep = "", collapse = NULL)
#dir_stations <- paste0(dirPrediccionInputs, "dailyData", sep = "", collapse = NULL)
#dir_inputs_nextgen <- paste0(dirPrediccionInputs, "NextGen/", sep = "", collapse = NULL)


dir_names = os.listdir(path_dpto)

path_json = glob.glob(f"{path_dpto}\\**\\cpt_areas.json", recursive = True)
init_params = {k: load_json(pth) for k,pth in zip(dir_names, path_json)}

month  =  int(date.today().strftime("%m"))
year   =  int(date.today().strftime("%Y"))
season = {k: get_season_months(season_type = value["type"], month = month, month_abb = month_abb) for k,value in init_params.items()}


start_date = date.today()+ datetime.timedelta(days = 30)
years = {k: get_season_years(season_type = value["type"], month = month, year = year) for k,value in init_params.items()}

path_months_l = {x: os.path.join(main_dir, "run_CPT", x) for x in dir_names}
for ky,pth in path_months_l.items():
    seas = season[ky]
    path_months_l[ky] = [os.path.join(pth, i) for i in seas]
    for k in path_months_l[ky]:
        os.makedirs(k, exist_ok=True)
        os.makedirs(os.path.join(k, "output"), exist_ok=True)

path_down = {k: os.path.join(dir_save, k) for k in dir_names}#paste(dir_save,list.files(path_dpto),sep="/")
for value in path_down.values():
    os.makedirs(value, exist_ok=True)

path_areas = path_json
#data_areas = #MV
#data_areas_l = #MV
#areas_final = #MV
#n_areas_l = MV
#O_empty_2 = download.cpt = MV

print(" \n Archivos de entrada Descargados \n")

all_path_down = {k: glob.glob(f"{v}\\**.tsv.gz") for k,v in path_down.items()}
for k,v in all_path_down.items():
    for pth in v:
        if len(v) > 0:
            try:
                with gzip.open(pth, 'rb') as f_in:
                    with open(pth.replace(".gz", ""), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except:
                print('Error when unzziping')

all_path_unzziped = {k: glob.glob(f"{v}\\**.tsv") for k,v in path_down.items()}

tsm_o = {k: { os.path.basename(pth): read_Files(pth, skip = 0) for pth in v}   for k,v in all_path_unzziped.items()}
time_sel = {k: {nm: get_cpt_dates(df) for nm,df in v.items()} for k,v in tsm_o.items()}

print("\n Archivos de entrada cargados")
#df = tsm_o['58504322333cb94a800f809b']['Jun_Nov-Dec.tsv']


#data_x =  JM

path_stations = glob.glob(f"{path_dpto}\\**\\stations.csv", recursive = True)
data_y = {k: pd.read_csv(fl) for k,fl in zip(dir_names, path_stations)}
part_id = {k: files_y(df, k) for k,df in data_y.items()}

print("\n Archivos de entrada y estaciones construidos para CPT \n")

confi_l = {k: [x for x in v["modes"].values()] for k,v in init_params.items()}
p_data=   {k: v.shape[1]-2 for k,v in data_y.items() }  



######################################################
######################################################
######################################################
dates = read_Files("Y:/CPT_merged/2014/input/merged/Apr_Jul-Aug.tsv", skip = 2)
data_cpt1 = dates

# Eliminar DataFrames donde la primera columna contiene el texto "cpt"

data_cpt1 = data_cpt1[~data_cpt1.iloc[:, 0].str.contains("cpt")]#[data_cpt1[~data_cpt1.iloc[:, 0].str.contains("cpt")] for data_cpt1 in data_cpt1]
pos = np.where(data_cpt1.iloc[:, 0] == "")[0]
# Generar la secuencia de identificadores para cada capa
nfields = len(data_cpt1)

pos = [i for i in range(1, nfields + 1)]

#genear la secuencia de los ids con el año y el nombre del campo e.j [1982_TSM, ..., 2023_TSM, 1982_URGD, ...]. Dim == len(pos)
#unlist(lapply(1:2, function(i){paste0(year, "_", nfieldsname[i])}))
np.repeat(["apr", "jun", "Jul"], repeats = 3, axis = 1)

to_save = {}
for i in range(len(pos)): 
    if i == (len(pos)-1): 
        end = data_cpt1.shape[0]
    else:
        end =pos[(i+1)]
    
    to_save["id"+str(i)] = data_cpt1.iloc[pos[i]:end]
# Guardar los identificadores 
