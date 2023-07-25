#pip install monthdelta
import os
import pandas as pd
import re
import glob
import datetime
from datetime import date
from monthdelta import monthdelta
import json


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
    with open(pth, 'r') as js:
        to_ret = json.load(js)
    return(to_ret[0])

def get_season_months(season_type, month, month_abb):
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
        to_ret = [tri1, tri2]





month_abb = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
root_dir = "D:/documents_andres/pr_1/Colombia/inputs"
#dirCurrent <- paste0(dirWorkdir, currentCountry, "/")
#dirInputs <- paste0(dirCurrent, "inputs/", sep = "", collapse = NULL)
#main_dir = "paste0(dirInputs, "prediccionClimatica/", sep = "", collapse = NULL)"


#dir_response <- paste0(dirPrediccionInputs, "estacionesMensuales", sep = "", collapse = NULL)
main_dir  = os.path.join(root_dir, "prediccionClimatica")
path_dpto = os.path.join(main_dir, 'estacionesMensuales')
dir_names = os.listdir(path_dpto)
path_stations = glob.glob(f"{path_dpto}\\**\\stations.csv", recursive = True)
path_json = glob.glob(f"{path_dpto}\\**\\cpt_areas.json", recursive = True)
init_params = {k: load_json(pth) for k,pth in zip(dir_names, path_json)}
season_type = [value["type"] for k,value in init_params.items()]

month  =  int(date.today().strftime("%m"))
year   =  int(date.today().strftime("%Y"))

season = [month+x for x in [1,4]]#month+[1,4] ADAPTAR PARA DIFERENTE SEASONS

start_date = date.today()+ datetime.timedelta(days = 30)
years      = [int((start_date+monthdelta(x)).strftime("%Y")) for x in range(6)]
years      = [years[x] for x in [0,3]] #years=format(seq(Sys.Date()+30, by = "month", length = 6) ,"%Y")[c(1,4)]
#years=get_season_years(month, year)
season     = [x - 12 if x>12 else x for x in season] #season[season>12]=season[season>12]-12

path_months_l = {os.path.join(main_dir, "run_CPT", x):{} for x in dir_names}
for ky in path_months_l.keys():
    path_months_l[ky] = [os.path.join(ky, month_abb[dr-1]) for dr in season]
    for k in path_months_l[ky]:
        os.makedirs(k, exist_ok=True)
        os.makedirs(os.path.join(k, "output"), exist_ok=True)



data_y = {k: pd.read_csv(fl) for k,fl in zip(dir_names, path_stations)}

part_id = {k: files_y(df, k) for k,df in data_y.items()}


  


