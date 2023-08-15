import os
import sys
import unittest

test_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(test_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from src.aclimate_prediction.prediction_class import AclimateDownloading

class TestAclimateDownloading(unittest.TestCase):

    def setUp(self):
        self.testFolder =  test_dir
        self.dataDir = os.path.abspath(os.path.join(self.testFolder, '..', 'data'))
        

    def test_download_folder_creation(self):

        #acclimate_downloading = AclimateDownloading()
        
        # Assert
        self.assertTrue(os.path.exists(self.dataDir))

if __name__ == '__main__':
    unittest.main()
