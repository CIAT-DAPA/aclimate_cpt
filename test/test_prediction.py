import pytest
import pandas as pd
import os
import sys
import numpy as np

test_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(test_dir, '..', 'src'))
data_dir = os.path.abspath(os.path.join(test_dir, '..', 'data'))
sys.path.insert(0, src_dir)


from prediction_class import AclimateDownloading

def test_download_folder_creation():
    assert os.path.exists(data_dir)



