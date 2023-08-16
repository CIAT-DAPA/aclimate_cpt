import os
from datetime import date
import argparse
from aclimate_cpt.prediction_class import AclimateDownloading


def main():

    # Params
    # 0: Country
    # 1: Path root
    # 2: Cores

    parser = argparse.ArgumentParser(description="Resampling script")

    parser.add_argument("-C", "--country", help="Country name", required=True) # Country
    parser.add_argument("-p", "--path", help="Path to data directory", default=os.getcwd())# Path root
    parser.add_argument("-c", "--cores", type=int, help="Number of cores", default=6, required=True)# Cores

    args = parser.parse_args()
    print("Reading inputs")
    country = args.country
 
    path = args.path
    month = int(date.today().strftime("%m"))
    year = int(date.today().strftime("%Y"))
    cores = args.cores

    ad = AclimateDownloading(path, country, month, year, cores = cores)
    ad.run_master()
 

if __name__ == "__main__":
    main()