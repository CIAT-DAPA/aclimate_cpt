#pip install monthdelta
#pip install tempfile
import os
import platform
import pandas as pd
import re
import glob
import datetime
import json
import numpy as np
import gzip
import shutil
from functools import reduce
import itertools
from monthdelta import monthdelta
import requests
import shutil
import time
from tqdm import tqdm
import concurrent.futures
from itertools import chain
from datetime import date, datetime, timedelta
import subprocess
import tempfile
import csv
from dateutil.relativedelta import relativedelta
from aclimate_cpt.tools_d_r import DownloadProgressBar,DirectoryManager

class AclimateDownloading():

    def __init__(self,path,country, month, year, cores):
     
     self.path = path
     self.country = country
     self.month = month
     self.year = year
     self.cores = cores
     self.path_inputs = os.path.join(self.path,self.country,"inputs")
     self.path_inputs_prediccion = os.path.join(self.path_inputs,"prediccionClimatica")
     self.path_inputs_monthly_stations = os.path.join(self.path_inputs_prediccion,"estacionesMensuales")
     self.path_inputs_downloads = os.path.join(self.path_inputs_prediccion,"descarga")
     self.path_inputs_run = os.path.join(self.path_inputs_prediccion,"run_CPT")
     self.path_outputs = os.path.join(self.path,self.country,"outputs")
     self.path_outputs_prediccion = os.path.join(self.path_outputs,"prediccionClimatica")
     self.path_save = os.path.join(self.path_outputs,"probForecast")

     pass



    def bind_rows(self,df1,df2):
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

    def merge_list(self,lst1, lst2):
                """Funciton to merge lists
                Params:
                lst1 (list) objetc
                lst2 (list) object

                Returns:
                merged list
                """
                ls = lst1 + lst2
                return(ls)

    def read_Files(self,ruta, skip):

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

    def sum_df(self, df1, df2):
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

    def get_cca_tables(self,loadings, cor):
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
            w_cor  = reduce(self.sum_df, tables)
            w_cor  = w_cor*(1/np.sum([float(x) for x in cor_ca]))
            
            
        w_cor  = w_cor.reset_index(drop = True)        
        #w_cor  = w_cor.to_numpy().flatten().tolist()

        return w_cor

    def correl(self,x, y):
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
        
        loads = self.read_Files(y, skip = 0)
        cor = self.read_Files(x, skip = 2)  # * El separador debe ser '\t' para indicar una tabulación
        cor= cor.drop(0, axis =1).drop(0, axis = 0).reset_index(drop=True).squeeze() # * El separador debe ser '\t' para indicar una tabulación
        

        lg = any(loads.iloc[:,0].str.contains("cpt:nfields"))
        
        if lg:

            nfields = int(loads.iloc[:,0].str.extract(r"cpt:nfields=([0-9]{1})").dropna().squeeze())
            loadings = loads.drop(range(2)).reset_index(drop= True)
            field_names = pd.Series(loadings.iloc[:, 0].str.extract("cpt:field=([A-Za-z]+)").dropna().squeeze().unique())
            #field_names = [field_names[x]+"_"+str(x) for x in range(len(field_names))]
            ps = [np.where( loadings.iloc[:, 0].str.contains("cpt:field="))[0][x] for x in range(len(np.where( loadings.iloc[:, 0].str.contains("cpt:field="))[0]))]
            ps = [ps[x] for x in range(0, len(ps), int(len(ps)/nfields))]
            ps = ps + [loadings.shape[0]] 
            spliteed = {field_names.iloc[n]: loadings.iloc[ps[n]:ps[n+1]].reset_index(drop= True).drop(range(1)).reset_index(drop= True) for n in range(len(ps)-1)}
        

            to_ret = {k: self.get_cca_tables(spliteed[k], cor) for k,v in spliteed.items()}
            # loadings = spliteed[k]
        else:
            
            loadings = loads.drop(range(2)).reset_index(drop= True)
            
            to_ret = self.get_cca_tables(loadings, cor)
        
        
        return to_ret

    def get_cpt_dates(self,df):
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

    def check_file_integrity(self,pth):
        """
        Function to check CPT downloaded integrity, looking at the number of rows

        Parameters:
        pth (str): Full path to the downloaded file

        Returns:
        status (str): Error if number of rows does not agree with the number of fields or Ok otherwise.


        """

        data_cpt1 = self.read_Files(pth, skip = 0)
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

    def data_raster(self,dates):
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
        #year_month = data_cpt1.iloc[0, :].dropna()
        if int(nfields.iloc[0,0])>1:  # Si solo hay un valor en la primera fila, es un archivo merged
            pos = np.where(data_cpt1.iloc[:,0].str.contains("cpt:T="))[0]
            tms= data_cpt1.iloc[pos,0]
            tms=tms.str.replace("/", "-")
            tms = tms.str.extract(r'cpt:T=([0-9]{4}-[0-9]{2}-[0-9]{2})') # uso de expreiones regulares para determinar las fechas que hay en el archivo
            # se usa str.extract de la libreria pandas para la lectura de expresiones regulares cpt:T=YYYY-MM-DD
            #year_month = tms  
            

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

    def run_optimization(self,raster, cor, out_file, path_y, dir_out_path, config, n_cols, data_trans):
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
            
            change_point = np.where(tmp_df.iloc[:,0].str.contains("cpt:field="))[0].tolist()# [x for x in np.diff(idx)]
            #change_point = np.cumsum([change_point.count(x) for x in np.unique(change_point)])
            #change_point = change_point[ range(0, int(len(change_point)/nfields))].tolist()
            change_point = [x for x in range(len(idx)) if idx[x] == change_point[1]] 
            change_point =  change_point + [len(idx) - change_point[0]]
        
            labs = reduce(self.merge_list,[np.repeat(field_names[x],change_point[x]).tolist() for x in range(len(change_point)) ])
            if len(labs) != len(idx):
                raise ValueError("Number of identified rows do not match field labels array length")
        else:
            cor = {field_names[0]: cor}
            labs = np.repeat(field_names[0], len(idx)).tolist()
        
        cor_vec = reduce(self.merge_list, [cor[x].to_numpy().flatten().tolist() for x in cor.keys()])
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
                    to_remove_idx = [x for x in range(len(idx)-1) if empty_counter[x]]
                    start_from = idx[to_remove_idx[0]]
                    end_pos    = idx[to_remove_idx[-1]+1]
                    tmp_df_perc = tmp_df_perc.iloc[start_from:end_pos]
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
                self.run_cpt(path_x     = pth_to_save,
                        path_y     = path_y,
                        run        = os.path.join(dir_out_path, f"run_{perc}.bat"),
                        output     = os.path.join(dir_out_path, "output" , f"{perc}_"),
                        confi      = config,
                        p          = n_cols,
                        type_trans = data_trans )
                
        return("Ok")

    def files_y(self, y_d, names, main_dir):
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
        na_pos = data.isna().all()
        if any(na_pos):
            print("Removing stations with full NA's")
            data = data.iloc[:, np.where(~na_pos)[0].tolist()]
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

    def load_json(self,pth):
        """Function to read input params json file
        Parameters:
        pth (str): full path to input params json file

        Returns:
        dict object with initial params for processing
        """
        with open(pth, 'r') as js:
            to_ret = json.load(js)
        return(to_ret)

    def run_cpt(self,path_x, path_y, run, output, confi, p, type_trans):

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

    def proba(self,
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

    def metricas(self,
            root_path:str
            ,month:int
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

    def get_season_months(self,season_lab, month_abb):
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

    def get_season_years(self,season_type, month, year):
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
            raise ValueError("Invalid season type")
            
        return(to_ret)

    def best_GI(self,rutas):
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


    def download_gz_predictor_file(self,url, path ,timeout=300):
        # url = urls[0]
        # path = all_path_down[0]
        code = False
        while code== False:
            try:
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()  # Check for HTTP errors
                
                if(response.status_code != 200):
                    raise Exception("Code status error")
                
                # Obtener el tamaño esperado del archivo del encabezado "Content-Length"
                expected_size = int(response.headers.get('Content-Length', 0))

                # Descargar el archivo y verificar el tamaño
                downloaded_size = 0
                
                progress_bar = tqdm(total=expected_size, unit='B', unit_scale=True, desc=path.split('/')[-1], miniters=1)

                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded_size += len(chunk)
                        f.write(chunk)
                        progress_bar.update(len(chunk))
                progress_bar.close()
                
                if downloaded_size == expected_size:
                    print(".gz file downloaded successfully")

                    # Descomprimir el archivo descargado
                    with gzip.open(path, 'rb') as f_in:
                        with open(path.replace('.gz',''), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(path)
                    print(".gz file uncompressed successfully")
                    code = True
                    return print("Process done successfully")
                    
                else:
                    print(f" Downloaded file size ({downloaded_size} bytes) doesn't match expected size ({expected_size} bytes).")
                    os.remove(path)
            except (requests.exceptions.RequestException, IOError) as e:
                print(f"Download failed: {e}")
                os.remove(path)
                

        return False


    def json_to_df(self,json, dpto):
        
        json_df = pd.json_normalize(json, ["transformation"]  ,['type','season','predictand',"areas",['modes','x'],['modes','y'],['modes','cca']],errors='ignore')
        json_df["dpto"]=dpto
        return json_df

    def make_url(self,path,dpto,typel,season,areas):
        #path = dir_save 
        # dpto =list(merged_df["dpto"])[0]
        # typel =list(merged_df["type"])[0]
        # season =list(merged_df["season"])[0]
        # areas =list(merged_df["areas"])[0]
        print(dpto)
        urls=[]
        paths = []
        path_descarga_dptos = os.path.join(path, dpto)
        month_abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        
        
        predictors = pd.json_normalize(areas)
        predic_final = predictors.groupby('predictor').agg({"predictor":"count",'x_min':[lambda x: x.iloc[0], lambda x: x.iloc[1] if len(x) > 1 else None ] , 'x_max':[lambda x: x.iloc[0],lambda x: x.iloc[1] if len(x) > 1 else None ],'y_min':[lambda x: x.iloc[0],lambda x: x.iloc[1] if len(x) > 1 else None],'y_max':[lambda x: x.iloc[0],lambda x: x.iloc[1] if len(x) > 1 else None]}).reset_index()
        
        predic_final.columns = predic_final.columns.map('_'.join)
        
        date_st = datetime.now()- timedelta(days=30)
        month_st = date_st.month
        year_st = date_st.year
        month_target = datetime.strptime(season.split('-')[0][:3], '%b').month
        diff_month =(month_target - month_st) % 12
        leng = 1 if typel=="bi" else 2
        
        for i  in range(predic_final.shape[0]):
            
            var_predic = predic_final.iloc[i,0]
            num_areas  = predic_final.iloc[i,1]
            
            x_min_1 = predic_final.iloc[i,2];  x_max_1 = predic_final.iloc[i,4]
            x_min_2 = predic_final.iloc[i,3];  x_max_2 = predic_final.iloc[i,5]
            
            y_min_1 = predic_final.iloc[i,6];  y_max_1 = predic_final.iloc[i,8]
            y_min_2 = predic_final.iloc[i,7];  y_max_2 = predic_final.iloc[i,9]
            
            source = "/.OCNF/.surface/.TMP/" if var_predic=="sst" else "/.PGBF/.pressure_level/.UGRD/"
            presion = "P/%28850%29/VALUES/"  if var_predic=="wind" else ""

            if num_areas==1:
                
                url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE"+ source +"SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE"+ source +"appendstream/S/%280000%201%20"+  month_abb[month_st-1] +"%201982-" + str(year_st) + "%29/VALUES/L/"+ str(diff_month) +".5/" + str(diff_month+leng) + ".5/RANGE/%5BL%5D//keepgrids/average/" + presion + "M/1/24/RANGE/%5BM%5Daverage/X/" + str(x_min_1) + "/" + str(x_max_1) + "/flagrange/Y/"+ str(y_min_1) + "/" + str(y_max_1) + "/flagrange/add/1/flaggt/0/maskle/mul/-999/setmissing_value/%5BX/Y%5D%5BS/L/add/%5Dcptv10.tsv.gz"
                
            if num_areas==2:
                
                url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE"+ source +"SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE"+ source +"appendstream/S/%280000%201%20"+  month_abb[month_st-1] +"%201982-" + str(year_st) + "%29/VALUES/L/"+ str(diff_month) +".5/" + str(diff_month+leng) + ".5/RANGE/%5BL%5D//keepgrids/average/"+ presion +"M/1/24/RANGE/%5BM%5Daverage/X/" + str(x_min_1) + "/" + str(x_max_1) + "/flagrange/Y/"+ str(y_min_1) + "/" + str(y_max_1) + "/flagrange/add/1/flaggt/X/" + str(x_min_2) + "/" + str(x_max_2) +"/flagrange/Y/"+ str(y_min_2) + "/"+ str(y_max_2) +"/flagrange/add/1/flaggt/add/0/maskle/mul/-999/setmissing_value/%5BX/Y%5D%5BS/L/add/%5Dcptv10.tsv.gz"
            
    
            file = str(month_abb[month_st-1]) + "_" + season + "_" + var_predic + ".tsv.gz"
            path_before = os.path.join(path_descarga_dptos,season)
            manager = DirectoryManager()
            manager.makedirs(path_before)
            path_last = os.path.join(path_before,file)
            
            urls.append(url)
            paths.append(path_last)
        total_predic = predic_final.shape[0]
        
        return urls, paths, total_predic

    
    def cpt_merge_x_files(self, file_paths):
        #file_path_1 = 'D:\\documents_andres\\pr_1\\Colombia\\inputs\\prediccionClimatica\\descarga\\58504314333cb94a800f8098\\Aug-Sep-Oct\\Jul_Aug-Sep-Oct_sst.tsv'
        # file_path_2 = 'D:\\documents_andres\\pr_1\\Colombia\\inputs\\prediccionClimatica\\descarga\\58504314333cb94a800f8098\\Aug-Sep-Oct\\Jul_Aug-Sep-Oct_wind.tsv'
        # merged_out_path = 'D:\\documents_andres\\pr_1\\Colombia\\inputs\\prediccionClimatica\\descarga\\58504314333cb94a800f8098\\Aug-Sep-Oct\\Jul_Aug-Sep-Oct_merged.tsv'
        # Nombre del archivo temporal y lugar donde se va a copiar este archivo
        #file_paths = ['Y:/CPT_merged/2014/input/sst_cfsv2/Apr_Jul-Aug.tsv',"Y:/CPT_merged/2014/input/U_wind/Apr_Jul-Aug_U_wind.tsv"]
        
        pos = np.where(np.array([x.find("wind") for x in file_paths])  >5)[0].tolist()
        if len(pos) != 0:
            pos = pos[0]
            if pos >= 1:
                file_paths.reverse()

            file_path_1 = file_paths[0]
            file_path_2 = file_paths[1]
            
            df_temp = self.read_Files(file_path_1, skip = 0)
            if df_temp.iloc[:,0].str.contains("aprod").any():
                df_temp.iloc[:,0] = df_temp.iloc[:,0].str.replace("aprod", "UGRD")
                df_temp.to_csv(file_path_1, sep ="\t", index = False, header= False,  quoting=csv.QUOTE_NONE)

            merged_out_path = re.sub("[a-zA-Z]+.tsv", "merged.tsv", file_path_1 )

            tmp_file = os.path.join(tempfile.gettempdir(), os.path.basename(tempfile.mktemp(suffix=".bat")))
            
            # argumentos para el batch
            if platform.system() == "Windows":
                cpt_batch = "CPT_batch.exe"
            elif platform.system() == "Linux":
                cpt_batch = "/forecast/models/CPT/15.5.3/bin/CPT.x"

            
            cmd_args = f"""@echo off
                (
                echo 611
                echo 801
                echo {file_path_1}
                echo /
                echo /
                echo /
                echo /
                echo /
                echo {file_path_2}
                echo /
                echo /
                echo /
                echo /
                echo /
                echo {merged_out_path}
                echo 0
                ) | {cpt_batch}"""
            
            with open(tmp_file, "w") as fl:
                    fl.write(cmd_args)
            try:
                if platform.system() == "Windows":
                    os.system(tmp_file)
        
                elif platform.system() == "Linux":
                    os.system("chmod +x"+tmp_file)
                    os.system(tmp_file)
                # Ejecución de CPT     
            
                
                # verificacion de que se creó el archivo tempora;l
                if not os.path.exists(tmp_file):
                    status = "Failed: Error en la creación del archivo temporal"
                else:
                    # copia del archivo a merged out path
                    #os.rename(tmp_file, merged_out_path)
                    os.remove(file_path_1)
                    os.remove(file_path_2)
                    status = "Success"
            except subprocess.CalledProcessError:
                status = "Failed: Error al ejecutar CPT_batch"
        else:
            print(f"No wind file was found on file_path list ommiting merge files process. {file_paths}")
            status = "Failed: Error al ejecutar CPT_batch for merging"
        
        return status


    def run_master(self):

        dir_names = os.listdir(self.path_inputs_monthly_stations)
        #path_json = glob.glob(f"{self.path_inputs_monthly_stations}\\**\\cpt_areas.json", recursive = True)
        path_json = glob.glob(os.path.join(self.path_inputs_monthly_stations, '**', 'cpt_areas.json'), recursive=True)
        init_params = {k: self.load_json(pth) for k,pth in zip(dir_names, path_json)}
        month_abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ext_exe = ".bat"

        season       = {k: [x["season"] for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}
        month_season =  {k: [self.get_season_months(x["season"], month_abb = month_abb) for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}
        predictands  =  {k: [x["predictand"] for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}
        predictors  =  {k: [ len(np.unique(pd.DataFrame(x["areas"])["predictor"].to_numpy().tolist())) for x in val if len(x['areas'] )!= 0]  for k,val in init_params.items()}

        #start_date = date.today()+ datetime.timedelta(days = 30)
        years = {k: self.get_season_years(season_type = value[0]["type"], month = self.month, year = self.year) for k,value in init_params.items()}

        path_months_l = {x: os.path.join(self.path_inputs_prediccion, "run_CPT", x) for x in dir_names}
        for ky,pth in path_months_l.items():
            seas = season[ky]
            path_months_l[ky] = [os.path.join(pth, i) for i in seas]
            for k in path_months_l[ky]:
                os.makedirs(k, exist_ok=True)
                os.makedirs(os.path.join(k, "output"), exist_ok=True)

        path_down = {k: os.path.join(self.path_inputs_downloads, k) for k in dir_names}#paste(dir_save,list.files(path_dpto),sep="/")
        for value in path_down.values():
            os.makedirs(value, exist_ok=True)

        #path_areas = path_json

        path_estaciones_dptos_list = list(map(lambda x: os.path.join(self.path_inputs_monthly_stations, x), dir_names))
        DirectoryManager()
        path_confi_cpt_nested = list( map(lambda x:  glob.glob(os.path.join(x, '*cpt*')),  path_estaciones_dptos_list ))
        path_confi_cpt = list(itertools.chain.from_iterable(path_confi_cpt_nested))
        json_list_nested = list( map(lambda x: self.load_json(x), path_confi_cpt ))
        resul = list(map(lambda x,y : self.json_to_df(x,y), json_list_nested,dir_names))
        merged_df = pd.concat(resul, ignore_index=True)

        urls_nested, paths_nested, total_predic_nested = zip(*map(self.make_url ,[self.path_inputs_downloads]*merged_df.shape[0] ,list(merged_df["dpto"]) ,list(merged_df["type"]) ,list(merged_df["season"]) , list(merged_df["areas"]) ))
        urls = list(chain(*urls_nested))
        all_path_down = list(chain(*paths_nested))
        #merge_when  = {k: [v[x] for x in range(len(v)) if int(v[x]) > 1 ] for k,v in predictors.items() }

        inicio = time.time()
        cores = self.cores

        with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
                    executor.map(self.download_gz_predictor_file, urls, all_path_down)

        fin = time.time()
        print(fin-inicio)

        print(" \n Archivos de entrada Descargados \n")



       # all_path_season_dir = {k: glob.glob(f"{v}\\**") for k,v in path_down.items()}
       # all_path_files = {k: [ glob.glob(f"{x}\\**.tsv")  for x in v] for k,v in all_path_season_dir.items()}
        
        for k,v in path_down.items():
            print(f"checkin for downloads path for {k} which is : {v}")

        all_path_season_dir = {k: glob.glob(os.path.join(v, '**')) for k, v in path_down.items()}
        all_path_files = {k: [glob.glob(os.path.join(x, '**/*.tsv')) for x in v] for k, v in all_path_season_dir.items()}
        
        for v in all_path_season_dir.values():
            for x in range(len(v)):
                print(f"path is: {v[x]}")
        

        for k,v in predictors.items():
            for x in range(len(v)):
                if v[x] > 1:
                    print(k)
                    print(f">{v[x]}")
                    print(f">>>{all_path_files[k][x]}")
                    self.cpt_merge_x_files(all_path_files[k][x])

        #all_path_unzziped = {k: glob.glob(f"{v}\\**\\**.tsv") for k,v in path_down.items()}
        all_path_unzziped = {k: glob.glob(os.path.join(v, '**', '**.tsv')) for k, v in path_down.items()}



        tsm_o = {k: [self.read_Files(pth, skip = 0) for pth in v]   for k,v in all_path_unzziped.items()}
        #time_sel = {k: {nm: get_cpt_dates(df) for nm,df in v.items()} for k,v in tsm_o.items()}

        print("\n Archivos de entrada cargados")
       
        #path_stations = glob.glob(f"{self.path_inputs_monthly_stations}\\**\\stations.csv", recursive = True)
        path_stations = glob.glob(os.path.join(self.path_inputs_monthly_stations, '**', 'stations.csv'), recursive=True)
        data_y = {k: pd.read_csv(fl) for k,fl in zip(dir_names, path_stations)}
        part_id = {k: self.files_y(df, k, main_dir = self.path_inputs_prediccion) for k,df in data_y.items()}

        print("\n Archivos de entrada y estaciones construidos para CPT \n")

        confi_l    = {k: [v[x]["modes"] for x in range(len(v)) if len(v[x]["modes"]) != 0] for k,v in init_params.items()}
        transform  =  {k: [v[x]["transformation"][0]["gamma"] for x in range(len(v)) if len(v[x]["transformation"]) != 0] for k,v in init_params.items()}
        p_data     = {k: v.shape[1]-2 for k,v in data_y.items() }  

        #path_x     = {x: glob.glob(f"{os.path.join(self.path_inputs_downloads,x)}\\**.tsv", recursive = True) for x in os.listdir(self.path_inputs_downloads)}   # lapply(list.files(dir_save,full.names = T),function(x)list.files(x,recursive = T,full.names = T))
        path_x = {x: glob.glob(os.path.join(self.path_inputs_downloads, x, '**.tsv'), recursive=True) for x in os.listdir(self.path_inputs_downloads)}

        #path_zone  = {dir_names[x]: glob.glob(f"{os.path.join(self.path_inputs_prediccion, 'run_CPT')}\\**\\y_**.txt", recursive = True)[x] for x in range(len(dir_names))} #list.files(paste0(main_dir,"run_CPT"),full.names = T) %>% paste0(.,"/y_",list.files(path_dpto),".txt")
        path_zone = {dir_names[x]: glob.glob(os.path.join(self.path_inputs_prediccion, 'run_CPT', '**', f'y_{list(files)[x]}.txt'), recursive=True)[x] for x in range(len(dir_names))}

        path_output_pred = {k: [ os.path.join(pth, "output","0_") for pth in v] for k,v in path_months_l.items()}
        path_run         = {k: [ os.path.join(pth, "run_0"+ext_exe) for pth in v] for k,v in path_months_l.items()}#lapply(path_months_l,function(x)paste0(x,"/output/0_"))

        print("\n Iniciando Primera corrida de CPT")


        for k in dir_names:
            for j in  range(len(path_x[k])):
                print(f">>> Processing: {os.path.basename( path_x[k][j])}")
                self.run_cpt( path_x = path_x[k][j],
                        path_y = path_zone[k],
                        run = path_run[k][j],
                        output = path_output_pred[k][j],
                        confi = confi_l[k][j],
                        p = p_data[k],
                        type_trans = transform[k][j])

        print("\n Primera corrida realizada")


        path_cc   = {k: [v[x]+"cca_cc.txt" for x in range(len(v))] for k,v in path_output_pred.items()}
        path_load = {k: [v[x]+"cca_load_x.txt" for x in range(len(v))] for k,v in path_output_pred.items()}
        cor_tsm   = {k: [self.correl(path_cc[k][n], path_load[k][n]) for n in range(len(path_cc[k]))   ] for k in dir_names}
        names_selec =  {k: [os.path.basename(path_x[k][x]).replace(".tsv", "") for x in  range(len(path_x[k])) ] for k,v in path_x.items()}


        print("\n Iniciando proceso de Optimización de área predictora")

        for k in dir_names:
            print(f">>> Creating files_x for: {k}")
            for j in range(len(path_x[k])):
                self.run_optimization(raster = tsm_o[k][j]
                                ,cor = cor_tsm[k][j]
                                ,out_file = path_x[k][j]
                                ,path_y = path_zone[k]
                                ,dir_out_path = path_months_l[k][j]
                                ,config  = confi_l[k][j]
                                ,n_cols = p_data[k]
                                ,data_trans = transform[k][j] )

       # best_decil_l ={k: [self.best_GI(glob.glob(v[j]+ "\\output"+"\\**_GI.txt")) for j in range(len(v))]  for k,v in path_months_l.items()}
        best_decil_l = {k: [self.best_GI(glob.glob(os.path.join(v[j], "output", '**_GI.txt'))) for j in range(len(v))] for k, v in path_months_l.items()}


        metricas_all = {k: [ self.metricas(root_path = best_decil_l[k][v],month = month_season[k][v],season_type = season[k][v],predictand = predictands[k][v], true_col_names = part_id[k], years_season = years[k][v])for v in range(len(best_decil_l[k]))] for k in dir_names}

        metricas_final = []
        for k,v in metricas_all.items():
            for j in range(len(v)):
                metricas_final.append(metricas_all[k][j])
        metricas_final = reduce(self.bind_rows, metricas_final)

        metricas_final.to_csv(
            os.path.join(self.path_save, "metrics.csv")
            ,float_format = float
            ,index = False
            )


        prob_all = {k: [ self.proba(root_path = best_decil_l[k][v],month = month_season[k][v],season_type = season[k][v],predictand = predictands[k][v], true_col_names = part_id[k], years_season = years[k][v])for v in range(len(best_decil_l[k]))] for k in dir_names}

        prob_final = []
        for k,v in prob_all.items():
            for j in range(len(v)):
                prob_final.append(prob_all[k][j])
        prob_final = reduce(self.bind_rows, prob_final)

        prob_final.to_csv(
            os.path.join(self.path_save, "probabilities.csv")
            ,float_format = float
            ,index = False
            )

        #####################################
        ###### END #########################
        ###################################