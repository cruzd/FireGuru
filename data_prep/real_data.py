#Take on the the real data file and modify it so there is 50& positives/negatives

import numpy as np
import pandas as pd 
from training import parameters as param

#File locations
fire_raw_path = "/data/fire/"
veg_raw_path = "/data/vegetation/"
nice_data_path = "/data/prepared/"

#First file to start file iteration
first_fire_file_fale = "HDF5_LSASAF_MSG_FDeM_MSG-Disk_201710011145"
first_veg_file_fale = "HDF5_LSASAF_MSG_LAI_MSG-Disk_201710010000"

#Function to return next file name
def nextFileName(currentFileName, minutesInterval):
    regexResult = re.search("(HDF5_LSASAF_MSG_LAI_MSG-Disk_|HDF5_LSASAF_MSG_FDeM_MSG-Disk_)(.+?)$",
                                  currentFileName)
    currentDateString = regexResult.group(2)
    if currentDateString is None:
        return None
    currentDate = datetime.datetime.strptime(currentDateString, "%Y%m%d%H%M")
    nextCurrentDate = currentDate + datetime.timedelta(minutes=minutesInterval)
    nextCurrentDateString = regexResult.group(1)+nextCurrentDate.strftime("%Y%m%d%H%M")
    return nextCurrentDateString

#Function to return next raw files
def nextFileNames(fireRawFileName, vegetationRawFileName):
    #Get the hour and minutes digits
    timeDigits = re.search(".*(\d\d\d\d)$", fireRawFileName).group(1)
    if timeDigits=="2345":
        #This means it's the next day. We need to update the vegetationRawFile too
        nextVegetationRawFileName = veg_raw_path+nextFileName(vegetationRawFileName,1440)
    else:
        #This means it's the same day. We can keep the same vegetationRawFile
        nextVegetationRawFileName = vegetationRawFileName
    nextFireRawFileName = fire_raw_path+nextFileName(fireRawFileName,15)
    return nextFireRawFileName, nextVegetationRawFileName

#Function to extract and save prepared data to a file
def extractNiceData(fireRawFileName, vegetationRawFileName, resultFile):
    print(fireRawFileName + "   " + vegetationRawFileName)
    #First thing to do is to get the nextFileName
    fireNextRawFileName = fire_raw_path+nextFileName(fireRawFileName, 15)
    
    #Get the raw files
    fireRawData = np.pad(np.matrix(h5py.File(fireRawFileName, 'r')["CF"]).getA(),((1,1),(1,1)), 'constant')
    fireNextRawData = np.pad(np.matrix(h5py.File(fireNextRawFileName, 'r')["CF"]).getA(),((1,1),(1,1)), 'constant')
    vegetationRawData = np.pad(np.matrix(h5py.File(vegetationRawFileName, 'r')["LAI"]).getA(),((1,1),(1,1)), 'constant')
    
    #Initialize the training array
    trainMatrix = []
    
    #Now let's extract the training array
    for row, row_value in enumerate(fireRawData):
        for column, column_value in enumerate(row_value):
            if(fireRawData[row][column]==1):
                if 2 in fireRawData[row-1:row+1,column-1:column+1]:
                    trainArray = np.array([fireRawData[row-1][column-1]-1,fireRawData[row-1][column]-1,fireRawData[row-1][column+1]-1,
                        fireRawData[row][column-1]-1,fireRawData[row][column]-1,fireRawData[row][column+1]-1,
                        fireRawData[row+1][column-1]-1,fireRawData[row+1][column]-1,fireRawData[row+1][column+1]-1,
                        vegetationRawData[row-1][column-1],vegetationRawData[row-1][column],vegetationRawData[row-1][column+1],
                        vegetationRawData[row][column-1],vegetationRawData[row][column],vegetationRawData[row][column+1],
                        vegetationRawData[row+1][column-1],vegetationRawData[row+1][column],vegetationRawData[row+1][column+1]])
                    if fireNextRawData[row][column]==2:
                        trainArray = np.append(trainArray,(0,1))
                    else:
                        trainArray = np.append(trainArray,(1,0))
                    trainMatrix.append(trainArray)
    #Convert this array
    trainMatrix = np.array(trainMatrix)
    
    np.savetxt(resultFile, trainMatrix, delimiter=',', fmt="%d")
    return None

fireRawFileName = fire_raw_path+first_fire_file_fale
vegetationRawFileName = veg_raw_path+first_veg_file_fale
with open(nice_data_path+"teste.txt", "a") as resultFile:
    while (fireRawFileName!=None) and (vegetationRawFileName!=None) :
        extractNiceData(fireRawFileName, vegetationRawFileName, resultFile)
        fireRawFileName, vegetationRawFileName = nextFileNames(fireRawFileName,vegetationRawFileName)