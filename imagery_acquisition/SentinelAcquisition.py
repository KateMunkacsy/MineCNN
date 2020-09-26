import pandas as pd
import numpy as np
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sentinelhub import SHConfig
from sentinelhub import DataSource
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox



# pull in mine coordinates
mines = pd.read_csv('data/mines.csv')
df = pd.DataFrame(mines)
minesarr = df.values
mine_lat = np.array(minesarr[:, 2])
mine_long = np.array(minesarr[:, 3])

# pull in non_mine coordinates
non_mines = pd.read_excel('data/nonMine_coords.xlsx')
df = pd.DataFrame(non_mines)
non_minesarr = df.values
non_mine_lat = np.array(non_minesarr[:, 1])
non_mine_long = np.array(non_minesarr[:, 0])


# was- 0.0354
def area_of_interest(lat, long):
  '''creating 50sqkm areas of interest'''
  coords = [[round(long[i]-0.0354, 4), round(lat[i]-0.0354, 4), 
           round(long[i]+0.0354, 4), round(lat[i]+0.0354, 4)] for i in range(len(lat))]
  coords = np.array(coords)
  np.random.shuffle(coords)
  coords = coords.tolist()
  return coords




def get_images(coords, folder):
  INSTANCE_ID = 'f4531504-b71f-4dc9-a931-73be0f9b97d0'  
  if INSTANCE_ID:
      config = SHConfig()
      config.instance_id = INSTANCE_ID
  else:
      config = None
  
  '''for i in range(int(len(coords)/2)):
      # area of interest
      mine_bbox = BBox(bbox=coords[i], crs=CRS.WGS84)

      wcs_true_color_request = WcsRequest(
          data_folder = folder,
          data_source = DataSource.SENTINEL2_L2A,
          layer='TRUE-COLOR-S2-L2A',
          bbox=mine_bbox,
          resx  = 15,
          resy = 15,
          maxcc = 0.1,
          config=config
          )

      wcs_true_color_img = wcs_true_color_request.get_data(save_data=True)'''


  for i in range(10000):
      # area of interest
      mine_bbox = BBox(bbox=coords[i], crs=CRS.WGS84)

      wms_true_color_request = WmsRequest(
          data_folder = folder,
          data_source = DataSource.SENTINEL2_L2A,
          layer='TRUE-COLOR-S2-L2A',
          bbox=mine_bbox,
          width = 700,
          height = 700,
          maxcc = 0.1,
          config=config
          )

      wms_true_color_img = wms_true_color_request.get_data(save_data=True)

      
def main():
  coords = area_of_interest(non_mine_lat, non_mine_long)
  get_images(coords, 'imagery_downloads/non_mine_dir_new')
  
if __name__ == "__main()__":
  main()