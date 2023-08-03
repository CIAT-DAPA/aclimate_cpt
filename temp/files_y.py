#pip install monthdelta
import os
import platform
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
from functools import reduce

def read_Files(ruta, skip):
    
    dataframe = pd.read_csv(ruta, sep='/t', engine='python', header=None)  # * El separador debe ser '\t' para indicar una tabulación
    
    #idx = np.where(dataframe.iloc[:,0].str.find("cpt:T") != -1)
    cp = re.compile("cpt:T=|cpt:field=")
    idx = np.where(pd.Series([len(cp.findall(x)) for x in dataframe.iloc[:,0]]) != 0)

    idx = [x+1 for x in idx]

    dataframe.iloc[idx[0]] = "\t"+dataframe.iloc[idx[0]]
    dataframe_ajustado=dataframe.iloc[:,0].str.split('\t', expand=True)
    dataframe_ajustado = dataframe_ajustado.drop(range(skip)).reset_index(drop=True)
    
    return dataframe_ajustado

def correl(x, y):
    """
    Function to calculate the weigthed correlation for cc_load_x.txt files

    Parameters:
    x (str): full path to the correlation file cca_cc.txt
    y (str): full path to the CCA load file cca_load_x.txt

    Retruns:
    dict object with pandas.Dataframes for each field found
    
    """
    #x = path_cc['58504322333cb94a800f809b'][0] #"Y:\CPT_merged\2022\output\raw_output\May_Oct-Nov_0.5_cca_load_x.txt"
    #y = path_load['58504322333cb94a800f809b'][0]
    
    loads = read_Files(y, skip = 0)
    cor = read_Files(x, skip = 2)  # * El separador debe ser '\t' para indicar una tabulación
    cor= cor.drop(0, axis =1).drop(0, axis = 0).reset_index(drop=True).squeeze() # * El separador debe ser '\t' para indicar una tabulación
    
    lg = any(loads.iloc[:,0].str.contains("cpt:nfields"))
    
    if lg:

        nfields = int(loads.iloc[:,0].str.extract(r"cpt:nfields=([0-9]{1})").dropna().squeeze())
        loadings = loads.drop(range(2)).reset_index(drop= True)
        field_names = pd.Series(loadings.iloc[:, 0].str.extract("cpt:field=([A-Za-z]+)").dropna().squeeze().unique())
        ps = [np.where( loadings.iloc[:, 0].str.contains("cpt:field="))[0][x] for x in range(len(np.where( loadings.iloc[:, 0].str.contains("cpt:field="))[0]))]
        ps = [ps[x] for x in range(0, len(ps), int(len(ps)/nfields))]
        ps = ps + [loadings.shape[0]] 
        spliteed = {field_names.iloc[n]: loadings.iloc[ps[n]:ps[n+1]].reset_index(drop= True).drop(range(1)).reset_index(drop= True) for n in range(len(ps)-1)}
        
        
        to_ret = {k: get_cca_tables(v, cor) for k,v in spliteed.items()}


    else:
        y.iloc[0, 0] = ""
        loadings = loads.drop(range(2)).reset_index(drop= True)
        
        to_ret = get_cca_tables(loadings, cor)
    
    
    return to_ret


def sum_df(df1, df2):
            """
            Function extract tables from cca_loas_x.txt files and calculate the weigthed correlation

            Parameters:
            df1 (pandas.DataFrame): first Dataframe
            df2 (pandas.DataFrame): Second DataFrame

            Returns:
            DataFrame
            """
            sm = df1+df2
            return(sm)

def get_cca_tables(loadings, cor):
    """
    Function extract tables from cca_loads_x.txt files and calculate the weigthed correlation

    Parameters:

    loadings (pandas.DataFrame): cca_loads_x.txt file
    cor (pandas.Series): Correlation values for each CCA Mode

    Returns:
    pandas.DataFrame with weighted correlation

    """

    loadings = loadings[~loadings.iloc[:, 0].str.contains("cpt")]
        #loadings = loadings.replace(0, np.nan)
    pos = np.where(loadings.iloc[:, 0] == "")[0]
    if len(pos) == 1:
        w_cor = loadings.dropna(axis = 1)
    else:
        tables = []
        for idx in range(len(pos)):
            #df_key = "id" + str(idx)
            start_pos = pos[idx]
            if idx ==  len(pos)-1:
                end_pos = loadings.shape[0]
            else:
                end_pos = pos[idx+1]
            
            cor_val = float(cor.iloc[idx])
            tmp_df  = loadings.iloc[start_pos:end_pos].reset_index(drop=True)
            tmp_df  = tmp_df.drop(0, axis = 0).drop(0, axis = 1).astype(float)
            tmp_df  = tmp_df.replace(-999, np.nan)
            tables.append(abs(tmp_df)*cor_val)

        if len(pos) != len(tables):
            raise ValueError("Lonigtud de tablas es diferente a la original")

        cor_ca = cor.iloc[range(len(tables))]
        w_cor  = reduce(sum_df, tables)
        w_cor  = w_cor*(1/np.sum([float(x) for x in cor_ca]))
    return(w_cor)

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

def check_file_integrity(pth):
    """
    Function to check CPT downloaded integrity, looking at the number of rows

    Parameters:
    pth (str): Full path to the downloaded file

    Returns:
    status (str): Error if number of rows does not agree with the number of fields or Ok otherwise.


    """

    data_cpt1 = read_Files(pth, skip = 0)
    nfields = data_cpt1.iloc[:,0].str.extract(r'cpt:nfields=([0-9]{1})').dropna()
    data_cpt1 = data_cpt1.drop(range(2))
    data_cpt1 = data_cpt1[~data_cpt1.iloc[:, 0].str.contains("cpt")]
    # Resetear el índice después de eliminar las filas
    data_cpt1.reset_index(drop=True, inplace=True)
    # Obtener posiciones de filas que tienen un valor vacío en la primera columna
    pos_para_dividir = np.where(data_cpt1.iloc[:, 0] == "")[0]
    to_check = np.unique(np.diff(pos_para_dividir))
    if len(to_check) != nfields:
        status = "Error"
    else:
        status = "Ok"

    return(status)



def data_raster(dates):
    """
    Function to split CPT layers into a dict
    
    Parameters:
    dates (pd.DataFrame) Dataframe produced by read_Files function

    Returns:
    Dict with dataFrames for each individual layer
    
    """

    data_cpt1 = dates.copy()
    nfields = data_cpt1.iloc[:,0].str.extract(r'cpt:nfields=([0-9]{1})').dropna()
    data_cpt1 = data_cpt1.drop(range(2))
    year_month = data_cpt1.iloc[0, :].dropna()
    if int(nfields.iloc[0,0])>1:  # Si solo hay un valor en la primera fila, es un archivo merged
        pos = np.where(data_cpt1.iloc[:,0].str.contains("cpt:T="))[0]
        tms= data_cpt1.iloc[pos,0]
        tms=tms.str.replace("/", "-")
        tms = tms.str.extract(r'cpt:T=([0-9]{4}-[0-9]{2}-[0-9]{2})') # uso de expreiones regulares para determinar las fechas que hay en el archivo
        # se usa str.extract de la libreria pandas para la lectura de expresiones regulares cpt:T=YYYY-MM-DD
        year_month = tms  
        

    # Eliminar filas donde la primera columna contiene el texto "cpt"
    data_cpt1 = data_cpt1[~data_cpt1.iloc[:, 0].str.contains("cpt")]

    # Resetear el índice después de eliminar las filas
    data_cpt1.reset_index(drop=True, inplace=True)

    # Obtener posiciones de filas que tienen un valor vacío en la primera columna
    pos_para_dividir = np.where(data_cpt1.iloc[:, 0] == "")[0]
    #np.unique(np.diff(pos_para_dividir))
    # Dividir el DataFrame en DataFrames más pequeños basados en la secuencia de identificadores generada
    to_save = {}
    for idx in range(len(pos_para_dividir)):
        df_key = "id" + str(idx)
        start_pos = pos_para_dividir[idx]
        if idx ==  len(pos_para_dividir)-1:
            end_pos = data_cpt1.shape[0]
        else:
            end_pos = pos_para_dividir[idx+1]
        to_save[df_key] = data_cpt1.iloc[start_pos:end_pos].reset_index(drop=True)
    return(to_save)




def files_y(y_d, names, main_dir):
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
    return(to_ret)


def run_cpt(path_x, path_y, run, output, confi, p, type_trans):

    """
    Function to Run CPT batch commands.

    It creates .bat file with all the input parameters to run CPT_batch.exe and execute it, all outputs
    are stored in ouput folder

    Parameters:
    path_x (str): Full path to input x file
    path_y (str): Full path to predictor y file
    run (str): Full path to batch file
    output (str): Full path of output folder
    conf (dict): dict of len 3 with x,y and cca parameters
    p (int): total number of variables (stations) present in predictor y file
    type_tran (logical): Wheter to Gamma transform the data

    Returns:
    Ok str value
    
    """

    nfields = pd.read_csv(path_x, sep='/t', engine='python', header=None, nrows = 2)  # * El separador debe ser '\t' para indicar una tabulación
    nfields = int(nfields.iloc[:,0].str.extract(r'cpt:nfields=([0-9]{1})').dropna().iloc[0,0])

    modes_x = int(confi["x"])
    mode_y  = int(confi["y"])
    if p<10:
        mode_y=p
    mode_cca = int(confi["cca"])
    if p<5:
        mode_cca=p

    if mode_cca > modes_x or mode_cca > mode_y:
        mode_cca = np.min([modes_x, mode_y])
        #raise Warning("Mode x and Mode y cannot be greather than Mode CCA, setting mode_cca to min [Mode x , Mode y]")
    

    tr_type  = 2
    if type_trans:
        t = 541
    else:
        t = " "
        
    
    path_GI=(output+"GI.txt")
    path_pear=(output+"pearson.txt")
    path_2afc=(output+"2afc.txt")
    path_prob=(output+"prob.txt")
    path_roc_a=(output+"roc_a.txt")
    path_roc_b=(output +"roc_b.txt")
    path_pca_eigen_x=(output+"pca_eigen_x.txt")
    path_pca_load_x=(output+"pca_load_x.txt")
    path_pca_scores_x=(output+"pca_scores_x.txt")
    path_pca_eigen_y=(output+"pca_eigen_y.txt")
    path_pca_load_y=(output+"pca_load_y.txt")
    path_pca_scores_y=(output+"pca_scores_y.txt")
    path_cca_load_x=(output+"cca_load_x.txt")
    path_cca_cc=(output+"cca_cc.txt")
    path_cca_scores_x=(output+"cca_scores_x.txt")
    path_cca_load_y=(output+"cca_load_y.txt")
    path_cca_scores_y=(output+"cca_scores_y.txt")
    path_hit_s=(output+"hit_s.txt")
    path_hit_ss=(output+"hit_ss.txt")
    to_check = [path_GI, path_pear, path_2afc, path_prob, path_roc_a, path_roc_b, path_pca_eigen_x, path_pca_load_x,
                path_pca_scores_x, path_pca_eigen_y, path_pca_load_y, path_pca_scores_y, path_cca_load_x, path_cca_cc,
                path_cca_scores_x, path_cca_load_y, path_cca_scores_y, path_hit_s, path_hit_ss]
    
    
    if platform.system() == "Windows":
        cpt_batch = "CPT_batch.exe"
    elif platform.system() == "Linux":
        cpt_batch = "/forecast/models/CPT/15.5.10/bin/CPT.x"
        
    if nfields == 1:

        cmd = f"""@echo off
        (
        echo 611
        echo 545
        echo 1
        echo {path_x} 
        echo /
        echo /
        echo /
        echo /
        echo 1
        echo {modes_x}
        echo 2
        echo {path_y}
        echo 1
        echo {mode_y}
        echo 1
        echo {mode_cca}
        echo 9
        echo 1
        echo 532
        echo /
        echo /
        echo N
        echo 2
        echo 554
        echo {tr_type}
        echo {t}
        echo 112
        echo {path_GI}
        echo 311
        echo 451
        echo 455
        echo 413
        echo 1
        echo {path_pear}
        echo 413
        echo 3
        echo {path_2afc}
        echo 413
        echo 4 
        echo {path_hit_s}
        echo 413  
        echo 5
        echo {path_hit_ss} 
        echo 413
        echo 10
        echo {path_roc_b}
        echo 413
        echo 11
        echo {path_roc_a}
        echo 111
        echo 301
        echo {path_pca_eigen_x}
        echo 302
        echo {path_pca_load_x}
        echo 303
        echo {path_pca_scores_x}
        echo 311
        echo {path_pca_eigen_y}
        echo 312
        echo {path_pca_load_y}
        echo 313
        echo {path_pca_scores_y}
        echo 401
        echo {path_cca_cc}
        echo 411
        echo {path_cca_load_x}
        echo 412
        echo {path_cca_scores_x}
        echo 421
        echo {path_cca_load_y}
        echo 422
        echo {path_cca_scores_y}
        echo 501
        echo {path_prob}
        echo 0
        echo 0
        ) | {cpt_batch}"""
    elif nfields == 2:

        cmd = f"""@echo off
        (
        echo 611
        echo 545
        echo 1
        echo {path_x} 
        echo /
        echo /
        echo /
        echo /
        echo /
        echo /
        echo /
        echo /
        echo 1
        echo {modes_x}
        echo 2
        echo {path_y}
        echo 1
        echo {mode_y}
        echo 1
        echo {mode_cca}
        echo 9
        echo 1
        echo 532
        echo /
        echo /
        echo N
        echo 2
        echo 554
        echo {tr_type}
        echo {t}
        echo 112
        echo {path_GI}
        echo 311
        echo 451
        echo 455
        echo 413
        echo 1
        echo {path_pear}
        echo 413
        echo 3
        echo {path_2afc}
        echo 413
        echo 4 
        echo {path_hit_s}
        echo 413  
        echo 5
        echo {path_hit_ss}
        echo 413
        echo 10
        echo {path_roc_b}
        echo 413
        echo 11
        echo {path_roc_a}
        echo 111
        echo 301
        echo {path_pca_eigen_x}
        echo 302
        echo {path_pca_load_x}
        echo 303
        echo {path_pca_scores_x}
        echo 311
        echo {path_pca_eigen_y}
        echo 312
        echo {path_pca_load_y}
        echo 313
        echo {path_pca_scores_y}
        echo 401
        echo {path_cca_cc}
        echo 411
        echo {path_cca_load_x}
        echo 412
        echo {path_cca_scores_x}
        echo 421
        echo {path_cca_load_y}
        echo 422
        echo {path_cca_scores_y}
        echo 501
        echo {path_prob}
        echo 0
        echo 0
        ) | {cpt_batch}"""

    with open(run, "w") as fl:
            fl.write(cmd)
    if platform.system() == "Windows":
        os.system(run)
  
    elif platform.system() == "Linux":
        os.system("chmod +x"+run)
        os.system(run)


    logic = [not os.path.exists(pth) for pth in to_check]
    if any(logic):
        raise ValueError("Error when running CPT.bat file")

    return("ok")
        
        
    


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
ext_exe = ".bat"
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
season = {k: [x["season"] for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}

start_date = date.today()+ datetime.timedelta(days = 30)
years = {k: get_season_years(season_type = value[0]["type"], month = month, year = year) for k,value in init_params.items()}

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
part_id = {k: files_y(df, k, main_dir = main_dir) for k,df in data_y.items()}

print("\n Archivos de entrada y estaciones construidos para CPT \n")

confi_l    = {k: [v[x]["modes"] for x in range(len(v)) if len(v[x]["modes"]) != 0] for k,v in init_params.items()}
transform  =  {k: [v[x]["transformation"][0]["gamma"] for x in range(len(v)) if len(v[x]["transformation"]) != 0] for k,v in init_params.items()}
p_data     = {k: v.shape[1]-2 for k,v in data_y.items() }  

path_x     = {x: glob.glob(f"{os.path.join(dir_save,x)}\\**.tsv", recursive = True) for x in os.listdir(dir_save)}   # lapply(list.files(dir_save,full.names = T),function(x)list.files(x,recursive = T,full.names = T))
path_zone  = {dir_names[x]: glob.glob(f"{os.path.join(main_dir, 'run_CPT')}\\**\\**.txt", recursive = True)[x] for x in range(len(dir_names))} #list.files(paste0(main_dir,"run_CPT"),full.names = T) %>% paste0(.,"/y_",list.files(path_dpto),".txt")
path_output_pred = {k: [ os.path.join(pth, "output","0_") for pth in v] for k,v in path_months_l.items()}
path_run         = {k: [ os.path.join(pth, "run_0"+ext_exe) for pth in v] for k,v in path_months_l.items()}#lapply(path_months_l,function(x)paste0(x,"/output/0_"))

for k in dir_names:
    for j in  range(len(path_x[k])):
        print(f">>> Processing: {os.path.basename( path_x[k][j])}")
        run_cpt( path_x = path_x[k][j],
                path_y = path_zone[k],
                run = path_run[k][j],
                output = path_output_pred[k][j],
                confi = confi_l[k][j],
                p = p_data[k],
                type_trans = transform[k][j])

print("\n Primera corrida realizada")


path_cc   = {k: [x+"cca_cc.txt" for x in v] for k,v in path_output_pred.items()}
path_load = {k: [x+"cca_load_x.txt" for x in v] for k,v in path_output_pred.items()}
 

#lapply(path_cc,function(x)lapply(x,function(x1)read.table(x1,sep="\t",dec=".",header = T,row.names = 1,skip =2,fill=TRUE,na.strings =-999,stringsAsFactors=FALSE)))


#####################################################
######################################################
######################################################
dates = read_Files("Y:/CPT_merged/2014/input/merged/Apr_Jul-Aug.tsv", skip = 0)

path_x = path_x[ '58504322333cb94a800f809b'][0]
path_y = path_zone['58504322333cb94a800f809b']
run = path_run['58504322333cb94a800f809b'][0]
output = path_output_pred['58504322333cb94a800f809b'][0]
confi = confi_l['58504322333cb94a800f809b'][0]
p = p_data['58504322333cb94a800f809b']
type_trans = transform['58504322333cb94a800f809b'][0]

run_cpt( path_x = path_x[ '58504322333cb94a800f809b'][0],
path_y = path_zone['58504322333cb94a800f809b'],
run = path_run['58504322333cb94a800f809b'][0],
output = path_output_pred['58504322333cb94a800f809b'][0],
confi = confi_l['58504322333cb94a800f809b'][0],
p = p_data['58504322333cb94a800f809b'],
type_trans = transform['58504322333cb94a800f809b'][0])
