import sys
from datetime import date
from prediction_class import AclimateDownloading


if __name__ == "__main__":
    # Params
    # 0: Country
    # 1: Path root
    # 2: Cores

    parameters = sys.argv
    print("Reading inputs")
    country = parameters[0]
 
    path = parameters[1]
    month = int(date.today().strftime("%m"))
    year = int(date.today().strftime("%Y"))
    cores = int(parameters[2])

    ad = AclimateDownloading(path, country, month, year, cores = cores)
    ad.run_master()
 

