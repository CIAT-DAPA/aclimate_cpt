#pip install monthdelta
#pip install tempfile
import os
from tqdm import tqdm



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class DirectoryManager():
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Function which creates a folder. It checks if the folders exist before
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # (string) path: Path where the folder should be create
    def mkdir(self,path):
        if path != "" and not os.path.exists(path): 
            os.mkdir(path)
                
            
    def makedirs(self,path):
        if path != "" and not os.path.exists(path): 
            os.makedirs(path)
