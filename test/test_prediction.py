import os
import sys
import unittest
import pandas as pd

test_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(test_dir, '..', 'src'))
sys.path.insert(0, src_dir)
from prediction_class import AclimateDownloading

class TestAclimateDownloading(unittest.TestCase):

    def setUp(self):
        self.testFolder =  test_dir
        self.dataDir = os.path.abspath(os.path.join(self.testFolder, '..', 'data'))
        self.outputPath = os.path.abspath(os.path.join(self.testFolder, '..', 'data/outputs/prediccionClimatica/probForecast'))
        print(self.outputPath)

    def test_output_folder_creation(self):

        #acclimate_downloading = AclimateDownloading()
        
        # Assert
        self.assertTrue(os.path.exists(self.outputPath))
    
    def test_metrics_file(self):
        # check for file existance

        _file = self.outputPath+'/'+'metrics.csv'
        self.assertTrue(os.path.exists(_file))
        # check for file columns:
        df = pd.read_csv(_file,sep=',')
        expectedColumns = ['id','pearson','afc2','month','year','goodness','canonica']
        self.assertEqual(list(df.columns), expectedColumns)

        


    def test_probabilities_file(self):
        _file = self.outputPath+'/'+'probabilities.xlsx'
        self.assertTrue(os.path.exists(_file))
        # check for file columns:
        df = pd.read_excel(_file)
        expectedColumns = ['id','below','normal'
                           ,'above','year','month','season','predictand']
        self.assertEqual(list(df.columns), expectedColumns)

if __name__ == '__main__':
    unittest.main()
