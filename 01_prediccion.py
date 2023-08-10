#pip install monthdelta
import os
import platform
import pandas as pd
import re
import glob
import datetime
from datetime import date
import json
import numpy as np
import gzip
import shutil
from functools import reduce
import itertools
from monthdelta import monthdelta

def bind_rows(df1,df2):
    """
    Function for bind two dataframes by row
    
    Parameters:
    df1 (pandas.DataFrame): Dataframe 1
    df2 (pandas.DataFrame): Dataframe 2

    Returns:

    df_f (pandas.DataFrame) row binded dataframe
    """
    df_f = pd.concat([df1, df2])
    return df_f

def merge_list(lst1, lst2):
            """Funciton to merge lists
            Params:
            lst1 (list) objetc
            lst2 (list) object

            Returns:
            merged list
            """
            ls = lst1 + lst2
            return(ls)

def read_Files(ruta, skip):

    """
    Function to read CPT format input data
    
    Parameters (str): Full path to file
    Skip (int): Number of rows to ommit

    Returns:

    Pandas.DataFrame of CPT input file
    """
    
    dataframe = pd.read_csv(ruta, sep='/t', engine='python', header=None)  # * El separador debe ser '\t' para indicar una tabulación
    
    #idx = np.where(dataframe.iloc[:,0].str.find("cpt:T") != -1)
    cp = re.compile("cpt:T=|cpt:field=")
    idx = np.where(pd.Series([len(cp.findall(x)) for x in dataframe.iloc[:,0]]) != 0)

    idx = [x+1 for x in idx]

    dataframe.iloc[idx[0]] = "\t"+dataframe.iloc[idx[0]]
    dataframe_ajustado=dataframe.iloc[:,0].str.split('\t', expand=True)
    dataframe_ajustado = dataframe_ajustado.drop(range(skip)).reset_index(drop=True)
    
    return dataframe_ajustado

def sum_df(df1, df2):
            """
            Function to Summ/add two dataframes element wise

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
    array with weighted correlation

    """

    loadings = loadings[~loadings.iloc[:, 0].str.contains("cpt")]
        #loadings = loadings.replace(0, np.nan)
    pos = np.where(loadings.iloc[:, 0] == "")[0]
    if len(pos) == 1:
        w_cor = loadings.dropna(axis = 1)
        w_cor = w_cor.drop(0, axis = 0).drop(0, axis = 1).astype(float)
        w_cor = w_cor.replace(-999, np.nan)
        w_cor = abs(w_cor)
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
            tmp_df  = tmp_df.dropna(axis = 1)
            tmp_df  = tmp_df.replace(-999, np.nan)
            tables.append(abs(tmp_df)*cor_val)

        if len(pos) != len(tables):
            raise ValueError("Lonigtud de tablas es diferente a la original")

        cor_ca = cor.iloc[range(len(tables))]
        w_cor  = reduce(sum_df, tables)
        w_cor  = w_cor*(1/np.sum([float(x) for x in cor_ca]))
        
        
    w_cor  = w_cor.reset_index(drop = True)        
    #w_cor  = w_cor.to_numpy().flatten().tolist()

    return w_cor

def correl(x, y):
    """
    Function to calculate the weigthed correlation for cc_load_x.txt files

    Parameters:
    x (str): full path to the correlation file cca_cc.txt
    y (str): full path to the CCA load file cca_load_x.txt

    Retruns:
    dict object with pandas.Dataframes for each field found
    
    """
    #x = path_cc['58504314333cb94a800f8098'][0] #"Y:\CPT_merged\2022\output\raw_output\May_Oct-Nov_0.5_cca_load_x.txt"
    #y = path_load['58504314333cb94a800f8098'][0]
    
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
    

        to_ret = {k: get_cca_tables(spliteed[k], cor) for k,v in spliteed.items()}
        
    else:
        
        loadings = loads.drop(range(2)).reset_index(drop= True)
        
        to_ret = get_cca_tables(loadings, cor)
    
    
    return to_ret

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

def run_optimization(raster, cor, out_file, path_y, dir_out_path, config, n_cols, data_trans):
    """
    Funtion to create files_X for area optimization process and run CPT batch.

    Parameters:
    raster (pandas.DataFrame) with CPT predictors data loaded using readFiles function
    cor (dict) with correlation matrix for each cell in raster
    out_file (str) full_path to predictor data source
    path_y (str) full path to the predictand file (file_y , path_zone)
    dir_out_path (str) full path to output folder (path_months_l)
    config (dict) dictionary with configuration values for run CPT (confi_l)
    n_cols (int) number of variables present in file y data (p_data)
    data_trans (logical) whether gamma transform predictand data


    Returns:
    .txt files with selected cell for area optimization
    All CPT outputs
    """
    #how to run
    # raster = tsm_o['58504322333cb94a800f809b'][1]
    # cor    = cor_tsm['58504322333cb94a800f809b'][1]
    # out_file = path_x['58504322333cb94a800f809b'][1]
    # path_y = path_zone['58504322333cb94a800f809b']
    # dir_out_path = path_months_l['58504322333cb94a800f809b'][1]
    # config  = confi_l['58504322333cb94a800f809b'][1]
    # n_cols = p_data['58504322333cb94a800f809b']
    # data_trans = transform['58504322333cb94a800f809b'][1]

    out_file_name = out_file.replace(".tsv", "") 
    print(f"  >> Season {os.path.basename(out_file_name)}")
    #na = names_selec[29]
    #years = time_sel[[29]]
    

    nfields = int(raster.iloc[:,0].str.extract(r"cpt:nfields=([0-9]{1})").dropna().squeeze())
    field_names = raster.iloc[:, 0].str.extract("cpt:field=([A-Za-z]+)").dropna().squeeze().unique().tolist()
    tmp_df = raster.drop(range(2)).reset_index(drop= True)
    tag_add = tmp_df.iloc[np.where(tmp_df.iloc[:,0].str.match(r"cpt:T$") )[0].tolist(), : ].fillna("")

    if len(tag_add) == 1:
        tmp_df = tmp_df.drop(range(1)).reset_index(drop= True)
        tag_add_p = "\n"+"\t".join([str(x) for x in tag_add.iloc[0]])
    elif len(tag_add) > 1:
        raise ValueError("Multiple cpt:T matched when should be 1")
    else:
        tag_add_p = ""
    cp = re.compile("cpt:T=|cpt:field=")
    idx = np.where(pd.Series([len(cp.findall(x)) for x in tmp_df.iloc[:,0]]) != 0)[0].tolist()
        
    if nfields > 1:
        
        change_point =  [x for x in np.diff(idx)]
        change_point = np.cumsum([change_point.count(x) for x in np.unique(change_point)])
        change_point = change_point[ range(0, int(len(change_point)/nfields))].tolist() 
        change_point = change_point + [len(idx) - change_point[len(change_point)-1]] 
    
        labs = reduce(merge_list,[np.repeat(field_names[x],change_point[x]).tolist() for x in range(len(change_point)) ])
        if len(labs) != len(idx):
            raise ValueError("Number of identified rows do not match field labels array length")
    else:
        cor = {field_names[0]: cor}
        labs = np.repeat(field_names[0], len(idx)).tolist()
    
    cor_vec = reduce(merge_list, [cor[x].to_numpy().flatten().tolist() for x in cor.keys()])
    idx = idx + [tmp_df.shape[0]]

    for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"    Processing quantile {perc}")
        thr = np.nanquantile(cor_vec, q = perc)
        tmp_df_perc = tmp_df.copy()
        empty_counter = []
        for i in range(len(idx)-1):
            k       = labs[i]
            cor_tmp = cor[k]
            cor_tmp = cor_tmp.mask(cor_tmp < thr, np.nan)
            if cor_tmp.isna().values.all():
                empty_counter.append(False)
            else:
                empty_counter.append(True)

            cor_tmp = cor_tmp.mask(~cor_tmp.isna(), 1)
            #cor_tmp = np.where(cor_tmp.isna() )
            #tmp_df = tmp_df.copy()
            #hay un problema caundo los datos no son rregulares
            
            to_remove = np.where(tmp_df_perc.iloc[(idx[i]+2):idx[i+1], 1:].fillna(np.nan).isna().sum() != 0 )[0].tolist()
            if len(to_remove) != 0:
                to_keep = min(to_remove) + 1
            else:
                to_keep = tmp_df.shape[1]
        
            tmp_df_perc.iloc[(idx[i]+2):idx[i+1], 1:to_keep] = tmp_df_perc.iloc[(idx[i]+2):idx[i+1], 1:to_keep].dropna(axis = 1).reset_index(drop=True).astype(float).mul(cor_tmp.reset_index(drop = True), axis = 0).fillna(-999)
            #for pos in range(len(cor_tmp[0])):
            #    tmp_df.iloc[(idx[i]+2):idx[i+1], 1:].iloc[cor_tmp[0][pos], cor_tmp[1][pos]] = -999
        if nfields > 1:
            new_field_names = np.unique([labs[x]  for x in range(len(labs)-1) if empty_counter[x]]).tolist()
            nfields_new = "cpt:nfields="+str(len(new_field_names))
            if len(new_field_names) == 1:
                to_remove = [idx[x] for x in range(len(idx)-1) if empty_counter[x]]
                start_from = to_remove[0]
                tmp_df_perc = tmp_df_perc.iloc[start_from:]
        else:
            nfields_new = "cpt:nfields="+str(nfields)

        tmp_df_perc = tmp_df_perc.fillna("")
        pth_to_save = out_file_name+f"_{perc}.txt"
        with open(pth_to_save, "w") as fl:
            fl.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/")
            fl.write("\n")
            fl.write(nfields_new+tag_add_p)
            fl.write("\n")
            for i in range(tmp_df_perc.shape[0]):
                fl.write("\t".join([str(x) for x in tmp_df_perc.iloc[i]]))
                fl.write("\n")

        if not os.path.exists(pth_to_save):
            raise ValueError(f"File {pth_to_save} does not exists.")
        else:
            print(f"    >Runnig CPT_batch for {perc}")
            run_cpt(path_x     = pth_to_save,
                    path_y     = path_y,
                    run        = os.path.join(dir_out_path, f"run_{perc}.bat"),
                    output     = os.path.join(dir_out_path, "output" , f"{perc}_"),
                    confi      = config,
                    p          = n_cols,
                    type_trans = data_trans )


    return("Ok")

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

    return([x for x in data.columns if x != " "])

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

def proba(
    root_path:str
    ,month:int
    ,season_type:str
    ,predictand:str
    ,true_col_names:list
    ,years_season:str):
    """
    Function to generate probability files from optimization process

    Parameters:
    root_path (str): full path to the optimal CPT output 
    month (int): month number for season type (if season = tri --> month is the middle number else month is the first number)
    season_type (str): long season type (Jul-Aug-Sep or Jul-Aug)
    predictand (str): predictand used for forecasting
    true_col_names (list): true names for files y stations
    year_season (str): year for the season in str format

    Returns:
    df_final: dataFrame with probabilities
    """ 
    df_raw = pd.read_csv(root_path.replace("GI.txt", "prob.txt")
                        ,skiprows=3 
                        ,header=None
                        ,sep='\t'
                        , float_precision="high")
    col_names = df_raw.iloc[0]
    if (len(col_names)-1) != len(true_col_names):
        raise ValueError("numero de columnas no concuerda con las del archivo de probabilidades")
    
    col_names[0] = "id"
    col_names[1:] = true_col_names
    df_raw = df_raw.iloc[[1,4, 7]]
    df_raw.columns = col_names
    df_raw.iloc[:,0] = ["below", "normal", "above"]
    df_final = df_raw.transpose().reset_index()
    df_final.columns =  df_final.iloc[0]
    df_final = df_final[1:]
    df_final['year'] = years_season
    df_final["month"] = month
    df_final["season"] = season_type
    df_final["predictand"] = predictand

    return df_final

def metricas(
        root_path:str
        ,month:int
        ,season_type:str
        ,predictand:str
        ,true_col_names:str
        ,years_season:str
        ):
    """
    Function to get metrics from CPT outputs
        
    Parameters:
    root_path (str): full path to best_GI file
    month (int): month number for season type (if season = tri --> month is the middle number else month is the first number)
    season_type (str): long season type (Jul-Aug-Sep or Jul-Aug)
    predictand (str): predictand used for forecasting
    true_col_names (list): true names for files y stations
    years_season (str): year for the season in str format

    Returns:
    df_final: dataFrame with metrics
        
    """
    pearsonDf = pd.read_csv(root_path.replace("GI.txt", "pearson.txt")
                            ,skiprows=3 
                            ,header=None
                            ,sep='\t'
                            , float_precision="high"
                            )
    afcDf = pd.read_csv(root_path.replace("GI.txt", "2afc.txt")
                            ,skiprows=3 
                            ,header=None
                            ,sep='\t'
                            , float_precision="high"
                            )
    ccaDf = pd.read_csv(root_path.replace("GI.txt", "cca_cc.txt")
                            ,skiprows=3 
                            ,header=None
                            ,sep='\t'
                            , float_precision="high"
                            )
    giDf = pd.read_fwf(root_path
                            ,skiprows=7 
                            ,header=None
                            ,sep='\t'
                            , infer_nrows=True
                            )
    pearsonDf.columns    = ['Station','pearson']
    pearsonDf['Station'] = true_col_names
    afcDf.columns = ['Station','afc2']
    afcDf['Station'] = true_col_names
    ccaDf.columns = ['id','correlation']
    giDf.columns = ['current_x','current_y','current_cca','current_index'
                    ,'optimum_x','optimum_y','optimum_cca','optimum_index']
    pearsonacfDf  = pearsonDf.merge(afcDf, on='Station') #,suffixes=['_pearson','_afc']
    canonica= ccaDf[ccaDf.id  == 1].correlation.iloc[0]
    goodness = giDf.iloc[giDf.shape[0]-1].optimum_index
    pearsonacfDf['month'] = month
    pearsonacfDf['year'] = years_season
    pearsonacfDf['goodness']  = goodness
    pearsonacfDf['canonica'] = canonica
    pearsonacfDf.rename(columns = {'Station':'id'}, inplace = True)
    #pearsonacfDf['seson'] =  season_type
    #pearsonacfDf['predictand'] = predictand

    return pearsonacfDf

def get_season_months(season_lab, month_abb):
    """Function to get month season number from json input file
    Parameters:
    season_lab (str): text identifier for season, extracted from imput params json file
    month_abb (list): Year months three letter abbreviation

    Returns:
    Month number int (if season type is trimestral then month number will be the middle month else will be the first month number of season)
    """

    mnths = season_lab.split("-")
    if len(mnths) > 2:
        mn_abb = mnths[1]
    else:
        mn_abb = mnths[0]
    
    to_ret = [x+1 for x in range(len(month_abb)) if month_abb[x] == mn_abb]
        #raise ValueError("Invalid seaso type")
    if to_ret[0] > 12 or to_ret[0] < 0:
        raise ValueError("Month number out of [0,12] range")

    return(to_ret[0])

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

def best_GI(rutas):
    """
    Function to get Best GI filename from all optimization output files.

    Parameters:

    Rutas (list): list with all optimization output file paths

    Returns:
    
    Filename of Best_GI
    """

    valores = []  # Lista para almacenar los valores

    for ruta in rutas:
        datos = []  # Lista para almacenar los datos

        with open(ruta, 'r') as archivo:
            for linea in archivo:
                columnas = linea.strip().split()
                datos.append(columnas)

        # Convertimos los datos en un DataFrame usando pandas
        df = pd.DataFrame(datos)
        
        # Obtenemos el valor de la última fila en la última columna
        ultimo_valor = df.iloc[-1, -1]
        
        valores.append(ultimo_valor)
    valores = [float(x) for x in valores]
    best_gi = [rutas[x] for x in np.where(valores == np.max(valores))[0]]
    if len(best_gi) > 1:
        #best_gi = best_gi[0]
        print("tow or more fiel where found, selecting first one")#raise Warning("more than one file where found, selection the first one")
        
    return best_gi[0]

######### Run #################
start_time = date.today()
month_abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#options(timeout=180)

#########################################################
########## NOT RUN ######################################
#########################################################
print(os.path.join("D:/", "andres"))
#define some global variables (some paths should be already defined in runMain so may not be necesary here)
root_dir = os.path.join("D:"+os.sep, "documents_andres", "pr_1", "Colombia","inputs")
main_dir  = os.path.join(root_dir, "prediccionClimatica")
path_dpto = os.path.join(main_dir, 'estacionesMensuales')#dir_response
dir_save  = os.path.join(main_dir, "descarga") #paste0(dirPrediccionInputs, "descarga", sep = "", collapse = NULL)
os.makedirs(os.path.join(main_dir, "run_CPT"), exist_ok=True)
os.makedirs(dir_save, exist_ok=True)
ext_exe = ".bat"
dirOutputs  = os.path.join("D:"+os.sep, "documents_andres", "pr_1", "Colombia", "outputs")
dirPrediccionOutputs  = os.path.join(dirOutputs, "prediccionClimatica")
path_save = os.path.join(dirPrediccionOutputs, "probForecast")
os.makedirs(path_save, exist_ok=True)

#dirCurrent <- paste0(dirWorkdir, currentCountry, "/")
#dirInputs <- paste0(dirCurrent, "inputs/", sep = "", collapse = NULL)
#main_dir = "paste0(dirInputs, "prediccionClimatica/", sep = "", collapse = NULL)"
#dir_response <- paste0(dirPrediccionInputs, "estacionesMensuales", sep = "", collapse = NULL)
#dir_runCPT <- paste0(dirPrediccionInputs, "run_CPT", sep = "", collapse = NULL)
#dir_response <- paste0(dirPrediccionInputs, "estacionesMensuales", sep = "", collapse = NULL)
#dir_stations <- paste0(dirPrediccionInputs, "dailyData", sep = "", collapse = NULL)
#dir_inputs_nextgen <- paste0(dirPrediccionInputs, "NextGen/", sep = "", collapse = NULL)
#dirOutputs <- paste0(dirCurrent, "outputs/", sep = "", collapse = NULL)
#dirPrediccionOutputs <- paste0(dirOutputs, "prediccionClimatica/", sep = "", collapse = NULL)
#path_save <- paste0(dirPrediccionOutputs, "probForecast", sep = "", collapse = NULL)
#########################################################
########## END NOT RUN ##################################
#########################################################


dir_names = os.listdir(path_dpto)
path_json = glob.glob(f"{path_dpto}\\**\\cpt_areas.json", recursive = True)
init_params = {k: load_json(pth) for k,pth in zip(dir_names, path_json)}

month        =  int(date.today().strftime("%m"))
year         =  int(date.today().strftime("%Y"))
season       = {k: [x["season"] for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}
month_season =  {k: [get_season_months(x["season"], month_abb = month_abb) for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}
predictands  =  {k: [x["predictand"] for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}

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

tsm_o = {k: [read_Files(pth, skip = 0) for pth in v]   for k,v in all_path_unzziped.items()}
#time_sel = {k: {nm: get_cpt_dates(df) for nm,df in v.items()} for k,v in tsm_o.items()}

print("\n Archivos de entrada cargados")

path_stations = glob.glob(f"{path_dpto}\\**\\stations.csv", recursive = True)
data_y = {k: pd.read_csv(fl) for k,fl in zip(dir_names, path_stations)}
part_id = {k: files_y(df, k, main_dir = main_dir) for k,df in data_y.items()}

print("\n Archivos de entrada y estaciones construidos para CPT \n")

confi_l    = {k: [v[x]["modes"] for x in range(len(v)) if len(v[x]["modes"]) != 0] for k,v in init_params.items()}
transform  =  {k: [v[x]["transformation"][0]["gamma"] for x in range(len(v)) if len(v[x]["transformation"]) != 0] for k,v in init_params.items()}
p_data     = {k: v.shape[1]-2 for k,v in data_y.items() }  

path_x     = {x: glob.glob(f"{os.path.join(dir_save,x)}\\**.tsv", recursive = True) for x in os.listdir(dir_save)}   # lapply(list.files(dir_save,full.names = T),function(x)list.files(x,recursive = T,full.names = T))
path_zone  = {dir_names[x]: glob.glob(f"{os.path.join(main_dir, 'run_CPT')}\\**\\y_**.txt", recursive = True)[x] for x in range(len(dir_names))} #list.files(paste0(main_dir,"run_CPT"),full.names = T) %>% paste0(.,"/y_",list.files(path_dpto),".txt")
path_output_pred = {k: [ os.path.join(pth, "output","0_") for pth in v] for k,v in path_months_l.items()}
path_run         = {k: [ os.path.join(pth, "run_0"+ext_exe) for pth in v] for k,v in path_months_l.items()}#lapply(path_months_l,function(x)paste0(x,"/output/0_"))

print("\n Iniciando Primera corrida de CPT")
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
cor_tsm   = {k: [correl(path_cc[k][n], path_load[k][n]) for n in range(len(path_cc[k]))   ] for k in dir_names}
names_selec =  {k: [os.path.basename(path_x[k][x]).replace(".tsv", "") for x in  range(len(path_x[k])) ] for k,v in path_x.items()}

print("\n Iniciando proceso de Optimización de área predictora")
for k in dir_names:
    print(f">>> Creating files_x for: {k}")
    for j in range(len(path_x[k])):
        run_optimization(raster = tsm_o[k][j]
                         ,cor = cor_tsm[k][j]
                         ,out_file = path_x[k][j]
                         ,path_y = path_zone[k]
                         ,dir_out_path = path_months_l[k][j]
                         ,config  = confi_l[k][j]
                         ,n_cols = p_data[k]
                         ,data_trans = transform[k][j] )

best_decil_l ={k: [best_GI(glob.glob(v[j]+ "\\output"+"\\**_GI.txt")) for j in range(len(v))]  for k,v in path_months_l.items()}


metricas_all = {k: [ metricas(root_path = best_decil_l[k][v],month = month_season[k][v],season_type = season[k][v],predictand = predictands[k][v], true_col_names = part_id[k], years_season = years[k][v])for v in range(len(best_decil_l[k]))] for k in dir_names}

metricas_final = []
for k,v in metricas_all.items():
    for j in range(len(v)):
        metricas_final.append(metricas_all[k][j])
metricas_final = reduce(bind_rows, metricas_final)

metricas_final.to_csv(
    os.path.join(path_save, "metrics.csv")
    ,float_format = float
    ,index = False
    )


prob_all = {k: [ proba(root_path = best_decil_l[k][v],month = month_season[k][v],season_type = season[k][v],predictand = predictands[k][v], true_col_names = part_id[k], years_season = years[k][v])for v in range(len(best_decil_l[k]))] for k in dir_names}

prob_final = []
for k,v in prob_all.items():
    for j in range(len(v)):
        prob_final.append(prob_all[k][j])
prob_final = reduce(bind_rows, prob_final)

prob_final.to_excel(
    os.path.join(path_save, "probabilities.xlsx")
    ,float_format = float
    ,index = False
    )

#####################################
###### END #########################
###################################
