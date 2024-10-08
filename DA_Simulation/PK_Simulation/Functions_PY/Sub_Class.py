#!  /user/bin/env python

"""
# CoCo defined classes
# Update 20181210
#   Used for refined IM
# Update 20190209
#   Content:
#       Function description update
# Moved out classes:
    -   
"""
import copy
import datetime
import gc
import hashlib
import numpy
import os
import pywt
import scipy
import time
import zlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import cpu_count
from multiprocessing import Pool
from scipy import interpolate
from scipy import io
from scipy import signal
from scipy import special

from Parameter_Setting import Parameter_Str_Wavelet_Name

Flag_Debug = False
if Flag_Debug:
    from pandas import DataFrame

# class Class_Signal_Generation():
#     """
#     Class for signal generation including:
#         1. Artificiail synthesized
#         2. Measurement
#     ----------------------------------------------------------------------------
#     Funciton list:

#     """
#     def __init__(self):
#         self.Class_Name = 'Class for load both measured and synthesized signals'.
#         self.Bool_Flag_Debug = False

#     def Function_Impulse_Signal_Generation(self, Array_Signal):
#         Array_Signal_Impulse = numpy.zeros(Array_Signal.shape)
#         Array_Signal_Impulse[Array_Signal_Impulse.size // 2] = 1
#         Array_Signal_Impulse = (Array_Signal_Impulse - 1 / Array_Signal.size) \
#                                 / Array_Signal_Impulse.std() \
#                                 *  Array_Signal.std()
#         Array_Signal_Impulse = Array_Signal_Impulse + Array_Signal.mean()
#         return Array_Signal_Impulse

#     def Function_Impulse_Signal_Generation_Oscilate_Delta(self, \
#                     Array_Time_Predict, \
#                     Array_Signal_Predict, \
#                     Int_Length_Prediction):
#         Array_Signal \
#             = Array_Signal_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Signal_Predict.size - Int_Length_Prediction]
#         Array_Impulse_Signal_Predict \
#             = numpy.zeros(Array_Time_Predict.size)
#         Array_Impulse_Signal_Predict[Array_Time_Predict.size // 2] = 1
#         Array_Impulse_Signal_Predict[Array_Time_Predict.size // 2 - 1] = - 1
#         Array_Impulse_Signal_Predict \
#             = Array_Impulse_Signal_Predict / numpy.sqrt(2) \
#                 * numpy.sqrt(numpy.sum((Array_Signal - Array_Signal.mean())**2))
#         Array_Impulse_Signal_Predict \
#             = Array_Impulse_Signal_Predict + Array_Signal.mean()
#         Array_Impulse_Signal \
#             = Array_Impulse_Signal_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]       
#         return Array_Impulse_Signal, Array_Impulse_Signal_Predict

#     def Function_Remove_NAN_Value(self, Array_Time, Array_Signal):
#         Array_Index_NAN = numpy.argwhere(~numpy.isnan(Array_Signal)).reshape(-1)
#         if Array_Index_NAN[0] == 0:
#             Array_Signal[0] = 0
#         if Array_Index_NAN[-1] == Array_Signal.size:
#             Array_Signal[-1] = 0
#         Function_interpolate_1d \
#             = interpolate.interp1d(Array_Time[Array_Index_NAN], \
#                                     Array_Signal[Array_Index_NAN], \
#                                     kind = 'linear')
#         self.Array_Signal_Remove_NAN = Function_interpolate_1d(Array_Time)
#         return self.Array_Signal_Remove_NAN

#     def Function_Extract_Data_of_Delong_Zuo(self, \
#             i_Experiment_Time, i_Group, i_Col_Signal):
#         """
#         Load raw data from mat file:
#             This part of code was rewritten to avoid the "exec" commande, 
#             which caused multiple problems including the
#             the problem of hidden in the main variable space and pylint error.
#         ------------------------------------------------------------------------
#         Input:
#             i_Experiment_Time: 
#                 - Totally 10 experiments
#             i_Group:
#                 - Each experiment has 4 group of sensors
#             i_Col_Signal:
#                 - Groups 1-3 -> 30 Columns
#                 - Group 4 -> 10 Columns
#         ------------------------------------------------------------------------
#         Output:
#             Array_Time
#             Array_Signal
#         """
#         Str_Data_Path = 'Data/DataBase_Measurements/Thunderstorm_Zuo_Delong.mat'
#         RawData = io.loadmat(Str_Data_Path)
#         Struct_List_File = RawData['Struct_List_File']
#         Array_File_Name = []
#         List_Data = []
#         for i_File in range(Struct_List_File.size):
#             Str_File_Name = Struct_List_File[i_File,0][0][0]
#             if Str_File_Name[-3:] == 'csv':
#                 Array_File_Name \
#                     = numpy.append(\
#                             Array_File_Name, \
#                             numpy.squeeze(Struct_List_File[i_File,0][0][0]))
#                 List_Data.append(numpy.squeeze(Struct_List_File[i_File,0][6]))
#         # Extract specified signal from the file
#         i_File = i_Experiment_Time * 4 + i_Group
#         Value_Sampling_Interval = 0.02
#         Str_File_Name = Array_File_Name[i_File]
#         Array_Signal = List_Data[i_File][:,i_Col_Signal]
#         Array_Time = numpy.arange(0, Value_Sampling_Interval * Array_Signal.size, Value_Sampling_Interval)
#         return Array_Time, Array_Signal

#     def Function_Load_Measurements(self, Str_File_Name):
#         """
#         Input:
#             Str_File_Name
#         """
#         # Check the Str_File_Name
#         if Str_File_Name[-4:] != '.mat':
#             Str_File_Name = Str_File_Name + '.mat'
#         RawData = io.loadmat('Data/DataBase_Measurements/' + Str_File_Name)
#         self.Dict_Keys = RawData.keys()
#         if Str_File_Name == 'Haikui.mat':
#             self.Array2D_Wind = RawData['SF040100']
#             self.Value_Delta_Time = 3600 * 24 / self.Array2D_Wind.shape[0]
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#         if Str_File_Name == 'RFDdata.mat':
#             self.Array2D_Wind = RawData['matrixtot']
#             self.Value_Delta_Time = 1
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#             Int_Number_Interpolate_Times = 20
#             self.Value_Delta_Time_New = self.Value_Delta_Time / Int_Number_Interpolate_Times

#             Array_Time_New = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0],
#                                         (self.Array2D_Wind.shape[0] - 1) * Int_Number_Interpolate_Times + 1)

#             self.Array2D_Wind_New = numpy.zeros((Array_Time_New.size, self.Array2D_Wind.shape[1]))
#             for i_Signal in range(self.Array2D_Wind.shape[1]):
#                 Array_Wind_Interp = self.Array2D_Wind[:, i_Signal]
#                 Function_interpolate_1d = interpolate.interp1d(Array_Time, Array_Wind_Interp, kind='linear')
#                 self.Array2D_Wind_New[:, i_Signal] = Function_interpolate_1d(Array_Time_New)
#             Array_Time = Array_Time_New.copy()
#             self.Array2D_Wind = self.Array2D_Wind_New.copy()
#         if Str_File_Name == 'Thunderstorm_GS_SP_2_20120411.mat':
#             self.Array2D_Wind = numpy.zeros((36000,4))
#             self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
#             self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
#             self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
#             self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
#             self.Value_Delta_Time = 0.1
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#         if Str_File_Name == 'Thunderstorm_GS_SP_2_20110605_1450.mat':
#             self.Array2D_Wind = numpy.zeros((36000,4))
#             self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
#             self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
#             self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
#             self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
#             self.Value_Delta_Time = 0.1
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#         if Str_File_Name == 'Thunderstorm_GS_SP_2_20120411_0720.mat':
#             self.Array2D_Wind = numpy.zeros((36000,4))
#             self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
#             self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
#             self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
#             self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
#             self.Value_Delta_Time = 0.1
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#         if Str_File_Name == 'Thunderstorm_GS_SP_20120419_1250.mat':
#             self.Array2D_Wind = numpy.zeros((36000,4))
#             self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
#             self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
#             self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
#             self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
#             self.Value_Delta_Time = 0.1
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#         if Str_File_Name == 'Thunderstorm_GS_SP_3_20111025_1540.mat':
#             self.Array2D_Wind = numpy.zeros((36000,4))
#             self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
#             self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
#             self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
#             self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
#             self.Value_Delta_Time = 0.1
#             Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
#         for i_Col in range(self.Array2D_Wind.shape[1]):
#             self.Array2D_Wind[:,i_Col] = self.Function_Remove_NAN_Value(Array_Time, self.Array2D_Wind[:,i_Col])
#         return self.Array2D_Wind, self.Value_Delta_Time, self.Dict_Keys

#     def Function_Measurment_Description(self, Str_File_Name):
#         Array2D_Wind, Value_Delta_Time, Dict_Keys = self.Function_Load_Measurements(Str_File_Name)
#         Array_Time = numpy.linspace(Value_Delta_Time, Value_Delta_Time * Array2D_Wind.shape[0], Array2D_Wind.shape[0])
#         if Str_File_Name == 'Haikui.mat':
#             pyplot.figure(figsize = (12,8), dpi = 100, facecolor = 'white')
#             # for i_Level in range(Array2D_Signal_Empirical_Mode_Decomposition.shape[1]):
#             pyplot.subplot(221)
#             i_Col = 0
#             pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 0')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel('Wind speed (m/s)')
#             pyplot.xticks(numpy.arange(0,25,2))
#             pyplot.grid('on')
#             pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.tight_layout()
#             pyplot.subplot(222)
#             i_Col = 1
#             pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 1')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel(r'Wind direction ($^\circ$)')
#             pyplot.xticks(numpy.arange(0,25,2))
#             pyplot.grid('on')
#             pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.tight_layout()
#             pyplot.subplot(223)
#             i_Col = 2
#             pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 2')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel('Unknown Property')
#             pyplot.xticks(numpy.arange(0,25,2))
#             pyplot.grid('on')
#             pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.tight_layout()
#             pyplot.subplot(224)
#             i_Col = 3
#             pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 3')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel('Unknown Property')
#             pyplot.xticks(numpy.arange(0,25,2))
#             pyplot.grid('on')
#             pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.tight_layout()
#             pyplot.show()
#         if Str_File_Name == 'RFDdata.mat':
#             List_Keys = list(Dict_Keys)
#             pyplot.figure(figsize = ( 6 * 2, 2 * 5), dpi = 100, facecolor = 'white')
#             for i_Keys in range(Array2D_Wind.shape[1]):
#                 pyplot.subplot2grid((5,2), (i_Keys // 2, i_Keys % 2))
#                 pyplot.plot(Array_Time, Array2D_Wind[:,1], color = 'black')
#                 pyplot.grid('on')
#                 pyplot.title(List_Keys[i_Keys + 4])
#                 pyplot.xlabel('Time (s)')
#                 pyplot.ylabel('Wind speed (s)')
#                 pyplot.axis([0, 1800, 0, 45])
#                 pyplot.tight_layout()
#             pyplot.show()
#         if Str_File_Name == 'Thunderstorm_GS_SP_2_20120411.mat':
#             List_Keys = list(Dict_Keys)
#             pyplot.figure(figsize = (12,8), dpi = 100, facecolor = 'white')
#             # for i_Level in range(Array2D_Signal_Empirical_Mode_Decomposition.shape[1]):
#             pyplot.subplot(221)
#             i_Col = 0
#             pyplot.plot(Array_Time , Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 0')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel('Wind speed (m/s)')
#             pyplot.xticks(numpy.arange(0, 3800, 600))
#             pyplot.grid('on')
#             pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.title(List_Keys[i_Col + 3])
#             pyplot.tight_layout()
#             pyplot.subplot(222)
#             i_Col = 1
#             pyplot.plot(Array_Time , Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 1')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel(r'Wind direction ($^\circ$)')
#             pyplot.xticks(numpy.arange(0, 3800, 600))
#             pyplot.grid('on')
#             pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.title(List_Keys[i_Col + 3])
#             pyplot.tight_layout()
#             pyplot.subplot(223)
#             i_Col = 2
#             pyplot.plot(Array_Time , Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 2')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel('Unknown Property')
#             pyplot.xticks(numpy.arange(0, 3800, 600))
#             pyplot.grid('on')
#             pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.title(List_Keys[i_Col + 3])
#             pyplot.tight_layout()
#             pyplot.subplot(224)
#             i_Col = 3
#             pyplot.plot(Array_Time, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 3')
#             pyplot.xlabel('Time (h)')
#             pyplot.ylabel('Unknown Property')
#             pyplot.xticks(numpy.arange(0, 3800, 600))
#             pyplot.grid('on')
#             pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
#             pyplot.title(List_Keys[i_Col + 3])
#             pyplot.tight_layout()
#             pyplot.show()

#     def Function_Load_Wind_Speed(self, Str_Signal_Name):
#         """
#         This function loads the measurments and converts them with specified 
#         parameters
#         ------------------------------------------------------------------------
#         Input:
#             Str_Signal_Name
#                 1. Measured_Haikui_0
#                 2. hunderstorm_GS_SP_2_20120411
#         ------------------------------------------------------------------------
#         Output:
#             Array_Time
#             Array_Signal
#             Value_Sampling_Frequency
#             Value_Delta_Time
#         """
#         Array_Time = numpy.array([])
#         Array_Signal = numpy.array([])
#         if Str_Signal_Name == 'Measured_Haikui_0':
#             Str_File_Name = 'Haikui.mat'
#             Tuple_Function_Return \
#                 = self.Function_Load_Measurements(Str_File_Name)
#             Array2D_Wind, Value_Delta_Time \
#                 = Tuple_Function_Return[0:2]
#             Data_Matrix_Signal_Raw = Array2D_Wind[:,0]
#             Temp_Array_x5 \
#                 = Data_Matrix_Signal_Raw\
#                     [51200 - 1:Data_Matrix_Signal_Raw.shape[0]]
#             Array_Signal \
#                 = Temp_Array_x5[1001 - 1: 1001 - 1 + 1800]
#             Value_Sampling_Frequency = int(1)
#             Array_Time \
#                 = numpy.linspace\
#                     (1, Array_Signal.shape[0] * 1 / Value_Sampling_Frequency, \
#                         Array_Signal.shape[0])
#         elif Str_Signal_Name == 'Thunderstorm_GS_SP_2_20120411':
#             Str_File_Name = 'Thunderstorm_GS_SP_2_20120411.mat'
#             Tuple_Function_Return \
#                 = self\
#                     .Function_Load_Measurements(Str_File_Name)
#             Array2D_Wind, Value_Delta_Time = Tuple_Function_Return[0:2]
#             Array_Signal = Array2D_Wind[:,0]
#             Array_Time \
#                 = numpy.linspace(\
#                             0, \
#                             Value_Delta_Time * (Array_Signal.size - 1), \
#                             Array_Signal.size)
#             Value_Sampling_Frequency = int(1 / Value_Delta_Time)
#         if Array_Time.size == 0:
#             print('Error: The input signal name is not defined, returning NULL')
#         return Array_Time, Array_Signal, Value_Sampling_Frequency, Value_Delta_Time

#     def Function_Signal_Generation(self, Str_Signal_Name):
#         """
#         Output:
#             Array_Time
#             Array_Signal
#         """
#         Array_Time = numpy.array([])
#         Array_Signal = numpy.array([])
#         if Str_Signal_Name == 'Amplitude_Modulated_Signal_0':
#             Array_Time = numpy.linspace(0, 273, 2731)
#             Value_M = 1
#             Value_f_m = 0.005  # Hz
#             Value_Main_Frequency = 2  # Hz
#             Array_Signal = numpy.sin(
#                 Array_Time * Value_Main_Frequency * 2 * numpy.pi)
#             Array_Signal[1000:2000] = numpy.sin(Array_Time[1000:2000] * Value_Main_Frequency * 2 * numpy.pi) \
#                                         * (1 + 0 * Value_M * numpy.sin((Array_Time[1000:2000] - 100) * 2 * numpy.pi * Value_f_m))
#             Array_Signal = Array_Signal - Array_Signal.mean()
#         elif Str_Signal_Name == 'Composite_Signal_1':
#             Value_f11 = 1
#             Value_f12 = 2
#             Value_f2 = 3
#             Value_omega_11 = 2 * numpy.pi * Value_f11
#             Value_omega_12 = 2 * numpy.pi * Value_f12
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 30:
#                     Array_Signal[i_Time] = numpy.sin(Array_Time[i_Time] * Value_omega_11) + numpy.sin(Array_Time[i_Time] \
#                                             * Value_omega_2)
#                 elif Array_Time[i_Time] <=70:
#                     Array_Signal[i_Time] = numpy.sin(Array_Time[i_Time] * Value_omega_11 \
#                                             + (Array_Time[i_Time] - 30)**2 / 80 ) \
#                                             + numpy.sin(Array_Time[i_Time] * Value_omega_2)
#                 else:
#                     Array_Signal[i_Time] = numpy.sin(Array_Time[i_Time] * Value_omega_12) \
#                                             + numpy.sin(Array_Time[i_Time] * Value_omega_2)
#             Array_Signal = Array_Signal - Array_Signal.mean()
#         elif Str_Signal_Name == 'Composite_Signal_AM_cos_and_cos':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.cos(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.cos(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 2 - numpy.abs(Array_Time[i_Time] - 50) / 5 
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 2 - numpy.abs(Array_Time[i_Time] - 50) / 5 
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a0':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 1
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a1':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 1 + 1 - numpy.abs(Array_Time[i_Time] - 50) / 5 
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a2':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 1 + 2 * (1 - numpy.abs(Array_Time[i_Time] - 50) / 5)
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a3':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 1 + 3 * (1 - numpy.abs(Array_Time[i_Time] - 50) / 5)
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a4':
#             Value_f1 = 1
#             Value_f2 = 0.2
#             Value_omega_1 = 2 * numpy.pi * Value_f1
#             Value_omega_2 = 2 * numpy.pi * Value_f2
#             Array_Time = numpy.linspace(0, 100, 1001)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
#             Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
#             Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
#             for i_Time in range(Array_Time.size):
#                 if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
#                     Array_Amplitude_Signal_1[i_Time] \
#                         = 1 + 4 * (1 - numpy.abs(Array_Time[i_Time] - 50) / 5)
#                 else:
#                     Array_Amplitude_Signal_1[i_Time] = 1
#             Temp_Array_Sub_Signal_1 \
#                 = Array_Base_Signal_1 * Array_Amplitude_Signal_1
#             Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
#             Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
#             Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
#             Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
#             Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
#         elif Str_Signal_Name == 'Duffin_Type_Wave':
#             Value_omega = numpy.pi * 2 / 100
#             Array_Time = numpy.linspace(0,1000,1001)
#             Array_Signal = numpy.sin(Array_Time * Value_omega + 0.3 * numpy.sin(2 * Value_omega * Array_Time) )
#             Array_Signal = Array_Signal - Array_Signal.mean()
#         elif Str_Signal_Name == 'Frequency_Modulated_Signal':
#             Array_Time = numpy.linspace(0,273, 2731)
#             Value_f_Delta = 0.1
#             Value_f_m = 0.05
#             Array_Signal = numpy.sin(Array_Time * 3 * 2 * numpy.pi)
#             Array_Signal[1000:2000] = numpy.sin(Array_Time[1000:2000] * 3 * 2 * numpy.pi + Value_f_Delta / Value_f_m * numpy.sin(2 * numpy.pi * Value_f_m * (Array_Time[1000:2000]-100)))
#         elif Str_Signal_Name == 'Impulse_Signal':
#             Array_Time = numpy.linspace(0,273, 2731)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Signal[int(Array_Signal.size / 2)] = 1
#         elif Str_Signal_Name == 'Impulse_Signal_0': 
#             # The suffix 0 means that it has the same property with the original impulse signal generation function
#             Array_Time = numpy.linspace(0,273, 2731)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Signal[int(Array_Signal.size / 2)] = 1
#         elif Str_Signal_Name == 'Swept_Sine_Signal':
#             Array_Time = numpy.linspace(0,273, 273 * 10 + 1)
#             Array_Signal = numpy.sin(Array_Time * 1 * 2 * numpy.pi + 1 / 273 * Array_Time**2 * 2 * numpy.pi)
#         elif Str_Signal_Name == 'Swept_Sine_Signal_Low_Frequency':
#             Array_Time = numpy.linspace(0,273, 273 * 10 + 1)
#             Value_Frequency = 0.4
#             Array_Signal = numpy.sin(Array_Time * 1 * Value_Frequency * numpy.pi + 1 / 273 * Array_Time**2 * Value_Frequency * numpy.pi)
#         elif Str_Signal_Name == 'Swept_Sine_Signal_Multiple_Trend':
#             Array_Time = numpy.linspace(-100, 400, 500 * 10 + 1)
#             Array_Signal = numpy.zeros(Array_Time.shape)
#             Array_Signal[0:1000] = numpy.sin(Array_Time[0:1000] * 1 * 2 * numpy.pi - 1 / 273 * Array_Time[0:1000]**2 * 2 * numpy.pi + (200 + 1 / 273 * 100**2) * 2 * numpy.pi)
#             Array_Signal[1000:1000 + 273 * 10 + 1] = numpy.sin(Array_Time[1000:1000 + 273 * 10 + 1] * 1 * 2 * numpy.pi + 1 / 273 * Array_Time[1000:1000 + 273 * 10 + 1]**2 * 2 * numpy.pi + (200 + 100**2 / 273 ) * 2 * numpy.pi)
#             Array_Signal[1000 + 273 * 10 + 1:] = numpy.sin(Array_Time[1000 + 273 * 10 + 1:] * 2 * 3 * numpy.pi - 1 / 273 * (Array_Time[1000 + 273 * 10 + 1:] - 273)**2 * 2 * numpy.pi + (200 + 100**2 / 273 - 273) * 2 * numpy.pi)
#         elif Str_Signal_Name == 'Stationary_Signal_0':
#             Array_Time = numpy.linspace(0,273, 2731)
#             Array_Signal = numpy.sin(Array_Time * 2.5 * 2 * numpy.pi) #+ numpy.sin(Array_Time * 3.2 * 2 * numpy.pi) + numpy.sin(Array_Time * 4.2 * 2 * numpy.pi)
#         elif Str_Signal_Name == 'White_Noise_Signal_0': # White gaussian noise with unity std and zero mean
#             Array_Time = numpy.linspace(0, 273, 2731)
#             Array_Signal = numpy.random.randn(Array_Time.size)
#             Array_Signal = Array_Signal / Array_Signal.std()
#             Array_Signal = Array_Signal  - Array_Signal.mean()
#             Array_Signal = Array_Signal + 0
#         elif Str_Signal_Name == 'White_Noise_Signal_Cut_0': # White gaussian noise with unity std and zero mean
#             Array_Time = numpy.linspace(0, 273, 2731)
#             Array_Signal = numpy.random.randn(Array_Time.size)
#             Array_Signal = Array_Signal / Array_Signal.std()
#             Array_Signal = Array_Signal  - Array_Signal.mean()
#             Array_Signal = Array_Signal
#         # Error report
#         if Array_Time.size == 0:
#             print('Error: The input signal name is not defined, returning NULL')
#         self.Array_Time = Array_Time
#         self.Array_Signal = Array_Signal
#         if 'Array2D_Accessory_Signal' not in locals():
#             Array2D_Accessory_Signal = 0
#         return self.Array_Time, self.Array_Signal, Array2D_Accessory_Signal

#     def Function_Vertical_Profile(self, Array_Z_Coordinate):
#         Array_Vertical_Profile = numpy.zeros(Array_Z_Coordinate.shape)
#         Parameter_V_max = 10
#         Parameter_b1 = -0.22
#         Parameter_b2 = -2.75
#         Parameter_z_max = 60.35
#         for i_Height in range(Array_Z_Coordinate.size):
#             Value_Height = Array_Z_Coordinate[i_Height]
#             Array_Vertical_Profile[i_Height] \
#                 = 1.354 * Parameter_V_max \
#                     * ((numpy.exp(Value_Height * Parameter_b1 / Parameter_z_max) \
#                         - numpy.exp(Value_Height * Parameter_b2 / Parameter_z_max)))
#         Array_Vertical_Profile = Array_Vertical_Profile / Array_Vertical_Profile[0]
#         return Array_Vertical_Profile


# class Class_Signal_Processing():
#     """
#     Function List:
#         - Function_Wavlelet_Max_Level
#         - Function_Low_Pass_Filter
#         - Function_Time_Varying_Correlation_Coefficient
#         - Function_Time_Varying_Mean
#         - Function_Time_Varying_Standard_Deviation
#         - Function_WPT_HT_Decomposition
#         - Function_WPT_NHT_Decomposition
#         - Fundtion_Surrogate_Data_Generation
#         - Fundtion_Surrogate_Data_Generation_With_Padding
#         - Function_Signal_Padding
#         - Function_Mirror_Signal_Padding
#         - Function_Signal_Depedding_2D
#         - Function_SVR_Prediction
#         - Function_SVR_2D_Training_Whole_Prediction
#         - Function_Nonstationary_Ratio
#         - Function_Local_Nonstationary_Index
#         - Function_Global_Nonstationary_Index
#         - Function_Global_Nonstationary_Index_With_FRF
#         - Function_Relative_Nonstationary_Index
#         - Function_Relative_Nonstationary_Index_With_FRF
#         - Function_Relative_Nonstationary_Index_With_Distribution
#         - Function_Relative_Nonstationary_Index_With_Distribution_FRF
#         - Function_Nonstationary_Index_Pad_Depad_Remove_Prediction
#         - Function_Normalized_Hilbert_Transform
#         - Function_Moving_Average
#         - Function_Remove_Trend
#         - Function_Remove_Amplitude_Modulation
#         - Function_Make_Unit_Length
#         - Function_Zero_Padding_Both_Side
#     """
#     def __init__(self):
#         self.Class_Name = 'Class of self defined signal processing functions'
    
#     def Function_Wavlelet_Max_Level(self, \
#             Array_Signal, Value_Delta_T, Str_Wavelet_Name):
#         Int_Max_WPT_Level = pywt.dwt_max_level(Array_Signal.size, \
#                     pywt.Wavelet(Str_Wavelet_Name))
#         Value_Window_Width = Value_Delta_T * 2**Int_Max_WPT_Level
#         return Int_Max_WPT_Level, Value_Window_Width

#     def Function_Low_Pass_Filter(self, Array_Signal):
#         Array_Filter = numpy.zeros(Array_Signal.shape)
#         Int_Length_Filter = 1
#         Array_Filter[0:Int_Length_Filter] = 1 / Int_Length_Filter
#         self.Array_Filtered_Signal = numpy.convolve(Array_Signal, Array_Filter) 
#         self.Array_Filtered_Signal \
#             = self.Array_Filtered_Signal[:Array_Signal.size]
#         return self.Array_Filtered_Signal    
    
#     def Function_Time_Varying_Correlation_Coefficient(self, \
#             Array_Time, Array_Signal_1, Array_Signal_2, Value_Averaging_Time):
#         Value_Sampling_Interval = numpy.diff(Array_Time).mean()
#         Int_Averaging_Points \
#             = int(Value_Averaging_Time / Value_Sampling_Interval)
#         Array_Correlation_Coefficient = numpy.zeros(Array_Time.size)
#         i_Start_Index \
#             = int(Int_Averaging_Points / 2)
#         i_Ended_Index \
#             = int(Array_Time.size - Int_Averaging_Points / 2)
#         for i_Time in numpy.arange(i_Start_Index, i_Ended_Index, 1):
#             Temp_i_Window_Start \
#                 = int(i_Time - Int_Averaging_Points / 2)
#             Temp_i_Window_Ended \
#                 = int(i_Time + Int_Averaging_Points / 2)
#             Temp_Array_Selected_Index \
#                 = numpy.arange(Temp_i_Window_Start, Temp_i_Window_Ended, 1)
#             Temp_Array2D_Corrcoef \
#                 = numpy.corrcoef(Array_Signal_1\
#                                         [Temp_Array_Selected_Index], \
#                                 Array_Signal_2\
#                                         [Temp_Array_Selected_Index])
#             Array_Correlation_Coefficient[i_Time] \
#                 = Temp_Array2D_Corrcoef[0,1]
#         return Array_Correlation_Coefficient
    
#     def Function_Time_Varying_Mean(self, \
#             Array_Time, Array_Signal, Value_Averaging_Time):
#         self.Array_Time = Array_Time
#         self.Array_Signal = Array_Signal
#         self.Value_Averaging_Time = Value_Averaging_Time
#         self.Array_Time_Varying_Mean = numpy.zeros(self.Array_Time.size)
#         self.Value_Sampling_Interval = numpy.diff(Array_Time).mean()
#         self.Int_Averaging_Points \
#             = int(self.Value_Averaging_Time / self.Value_Sampling_Interval)
#         Temp_Index_Start \
#             = int(self.Int_Averaging_Points / 2)
#         Temp_Index_Ended \
#             = int(self.Array_Time.size - self.Int_Averaging_Points / 2)
#         for i_Time in numpy.arange(Temp_Index_Start, Temp_Index_Ended, 1):
#             Temp_Array_Selected_Index \
#                 = numpy.arange(int(i_Time - self.Int_Averaging_Points / 2), \
#                                 int(i_Time + self.Int_Averaging_Points / 2), \
#                                 1)
#             self.Array_Time_Varying_Mean[i_Time] \
#                     = numpy.mean(self.Array_Signal[Temp_Array_Selected_Index])
#         return self.Array_Time_Varying_Mean
    
#     def Function_Time_Varying_Standard_Deviation(self, \
#             Array_Time, Array_Signal_Fluctuation, Value_Averaging_Time,
#             Str_Method):
#         """
#         Description:
#             Calculate the time-vayring standard deviation by moving average
#             method
#         ------------------------------------------------------------------------
#         Input:
#             Array_Time
#             Array_Signal_Fluctuation
#             Value_Averaging_Time
#             Str_Method:
#                 1. Moving_Average
#                 2. Wavelet (TBD)
#         ------------------------------------------------------------------------
#         Output:
#             Array_Time_Varying_Standard_Deviation
#         """
#         if Str_Method == 'Moving_Average':
#             self.Array_Time = Array_Time
#             self.Array_Signal_Fluctuation = Array_Signal_Fluctuation
#             self.Value_Averaging_Time = Value_Averaging_Time
#             self.Array_Time_Varying_Standard_Deviation \
#                 = numpy.zeros(self.Array_Time.size)
#             self.Value_Sampling_Interval = numpy.diff(Array_Time).mean()
#             self.Int_Averaging_Points \
#                 = int(self.Value_Averaging_Time / self.Value_Sampling_Interval)
#             Temp_Index_Start \
#                 = int(self.Int_Averaging_Points / 2)
#             Temp_Index_Ended \
#                 = int(self.Array_Time.size - self.Int_Averaging_Points / 2)
#             for i_Time in numpy.arange(Temp_Index_Start, Temp_Index_Ended, 1):
#                 Temp_Array_Selected_Index \
#                     = numpy.arange(\
#                             int(i_Time - self.Int_Averaging_Points / 2), \
#                             int(i_Time + self.Int_Averaging_Points / 2), \
#                             1)
#                 self.Array_Time_Varying_Standard_Deviation[i_Time] \
#                     = numpy.std(self.Array_Signal_Fluctuation\
#                                         [Temp_Array_Selected_Index])
#             self.Array_Time_Varying_Standard_Deviation\
#                     [: int(self.Int_Averaging_Points / 2) + 1] \
#                 = numpy.linspace\
#                         (numpy.abs(Array_Signal_Fluctuation[0]), \
#                         self.Array_Time_Varying_Standard_Deviation\
#                         [int(self.Int_Averaging_Points / 2) + 1], \
#                         int(self.Int_Averaging_Points / 2) + 1)
#             self.Array_Time_Varying_Standard_Deviation\
#                     [-int(self.Int_Averaging_Points / 2) - 2 : ] \
#                 = numpy.linspace\
#                         (self.Array_Time_Varying_Standard_Deviation\
#                         [- int(self.Int_Averaging_Points / 2) - 2], \
#                         numpy.abs(Array_Signal_Fluctuation[-1]), \
#                         int(self.Int_Averaging_Points / 2) + 2)
#         else:
#             print("Error! Secified method not found!")
#         return self.Array_Time_Varying_Standard_Deviation
    
#     def Function_WPT_HT_Decomposition(self, \
#             Array_Time, Array_Signal, \
#             Str_Wavelet_Name, Parameter_Max_WPT_Level, \
#             Array_Index_Selected_Component):
#         """
#         Wavelet packet decompostion nested with Hilbert transform
#         ------------------------------------------------------------------------
#         Input:
#             Array_Time, Array_Signal,Str_Wavelet_Name, Parameter_Max_WPT_Level
#             Array_Index_Selected_Component:
#                 if == 'All' -> Highest decomposition level
#                 if != 'All' -> Return selected component
#         ------------------------------------------------------------------------
#         Output:
#             Array2D_Signal_Decomposed
#             Array2D_Signal_Decomposed_Amplitude
#             Array2D_Signal_Decomposed_Phase
#             Array2D_Signal_Decomposed_Frequency
#             Array_Center_Frequency[Array_Index_Selected_Component]
#         """
#         Value_Signal_Mean = Array_Signal.mean()
#         Array_Signal = Array_Signal - Value_Signal_Mean
#         WPT_Signal \
#             = pywt.WaveletPacket(Array_Signal, Str_Wavelet_Name, 'symmetric')
#         if Parameter_Max_WPT_Level == 0:
#             Value_Maximum_Level = WPT_Signal.maxlevel
#         else:
#             Value_Maximum_Level = Parameter_Max_WPT_Level # WPT_Signal.maxlevel
#         if str(Array_Index_Selected_Component) == 'All':
#             Array_Index_Selected_Component \
#                 = numpy.arange(2**Value_Maximum_Level)
#         Array2D_Signal_Decomposed \
#             = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
#         i_Current_Node = 0
#         Array_Flag_Component_Selection \
#             = numpy.zeros(2**Value_Maximum_Level, dtype= bool)
#         Array_Flag_Component_Selection[Array_Index_Selected_Component] = True
#         for node in WPT_Signal.get_level(Value_Maximum_Level, 'freq'):
#             if Array_Flag_Component_Selection[i_Current_Node] == True:
#                 WPT_Reconstruct \
#                     = pywt.WaveletPacket(numpy.zeros(Array_Signal.shape), \
#                                             Str_Wavelet_Name, 'symmetric')
#                 WPT_Reconstruct[node.path] = WPT_Signal[node.path].data
#                 Array_Signal_Decomposed = WPT_Reconstruct.reconstruct()
#                 Array2D_Signal_Decomposed[:,i_Current_Node] \
#                     = Array_Signal_Decomposed
#             i_Current_Node = i_Current_Node + 1
#         Array2D_Signal_Decomposed[:,0] += Value_Signal_Mean
#         Array2D_Signal_Decomposed \
#             = Array2D_Signal_Decomposed[:, Array_Index_Selected_Component]
#         Array2D_Signal_Decomposed_Amplitude \
#             = numpy.zeros((Array_Signal.size, \
#                             Array_Index_Selected_Component.size))
#         Array2D_Signal_Decomposed_Phase \
#             = numpy.zeros((Array_Signal.size, \
#                             Array_Index_Selected_Component.size))
#         for i_Current_Node in range(Array_Index_Selected_Component.size):
#             Array_Temp_Signal_Subcomponents \
#                 = signal.hilbert(Array2D_Signal_Decomposed[:,i_Current_Node])
#             Array_Temp_Signal_Subcomponent_Amplitude \
#                 = numpy.abs(Array_Temp_Signal_Subcomponents)
#             Array_Temp_Signal_Subcomponent_Phase \
#                 = numpy.angle(Array_Temp_Signal_Subcomponents)
#             Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node] \
#                 = Array_Temp_Signal_Subcomponent_Amplitude
#             Array2D_Signal_Decomposed_Phase[:,i_Current_Node] \
#                 = numpy.unwrap(Array_Temp_Signal_Subcomponent_Phase)
#         Array2D_Signal_Decomposed_Frequency \
#             = numpy.diff(Array2D_Signal_Decomposed_Phase, axis = 0) \
#                     / numpy.diff(Array_Time).mean() \
#                     / 2 / numpy.pi
#         Array2D_Signal_Decomposed_Frequency \
#             = numpy.append\
#                         (numpy.zeros\
#                                 ((1, \
#                                 Array2D_Signal_Decomposed_Phase.shape[1])),\
#                         Array2D_Signal_Decomposed_Frequency, \
#                         axis = 0)
#         Value_Default_Sample_Frequency = 1 / numpy.diff(Array_Time).mean()
#         Value_Default_Low_Frequency_Threshold \
#             = 1 / (Array_Time.max() - Array_Time.min())
#         Value_Frequency_Band_Width \
#             = (Value_Default_Sample_Frequency / 2 \
#                     - Value_Default_Low_Frequency_Threshold) \
#                 / 2**Value_Maximum_Level
#         Array_Center_Frequency \
#             = numpy.linspace(Value_Default_Low_Frequency_Threshold, \
#                                 Value_Default_Sample_Frequency / 2 \
#                                     - Value_Frequency_Band_Width, \
#                                 2**Value_Maximum_Level) \
#                 + Value_Frequency_Band_Width / 2
#         Temp_Int_Number_Nodes = Array2D_Signal_Decomposed_Amplitude.shape[1]
#         for i_Current_Node in range(Temp_Int_Number_Nodes):
#             Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node] \
#                 = self.Function_Low_Pass_Filter\
#                         (Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node])
#         return Array2D_Signal_Decomposed, \
#                 Array2D_Signal_Decomposed_Amplitude, \
#                 Array2D_Signal_Decomposed_Phase, \
#                 Array2D_Signal_Decomposed_Frequency, \
#                 Array_Center_Frequency[Array_Index_Selected_Component]

#     def Function_WPT_NHT_Decomposition(self, \
#             Array_Time, Array_Signal, \
#             Str_Wavelet_Name, Parameter_Max_WPT_Level = 0, \
#             Array_Index_Selected_Component = 'All'):
#         # Wavelet packet decomposition with normalized Hilbert transform
#         Array_Time = numpy.copy(Array_Time, order = 'C')
#         Array_Signal = numpy.copy(Array_Signal, order = 'C')
#         Value_Signal_Mean = Array_Signal.mean()
#         Array_Signal = Array_Signal - Value_Signal_Mean
#         Value_Default_Sample_Frequency = 1 / numpy.diff(Array_Time).mean()
#         WPT_Signal \
#             = pywt.WaveletPacket(Array_Signal, Str_Wavelet_Name, 'symmetric')
#         if Parameter_Max_WPT_Level == 0:
#             Value_Maximum_Level = WPT_Signal.maxlevel
#         else:
#             Value_Maximum_Level \
#                 = Parameter_Max_WPT_Level # WPT_Signal.maxlevel
#         Array2D_Signal_Decomposed \
#             = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
#         i_Current_Node = 0
#         for node in WPT_Signal.get_level(Value_Maximum_Level, 'freq'):
#             WPT_Reconstruct \
#                 = pywt.WaveletPacket(numpy.zeros(Array_Signal.shape), \
#                                         Str_Wavelet_Name, \
#                                         'symmetric')
#             WPT_Reconstruct[node.path] = WPT_Signal[node.path].data
#             Array_Signal_Decomposed = WPT_Reconstruct.reconstruct()
#             Array2D_Signal_Decomposed[:,i_Current_Node] \
#                 = Array_Signal_Decomposed
#             i_Current_Node = i_Current_Node + 1
#         Array2D_Signal_Decomposed[:,0] += Value_Signal_Mean 
#         Array2D_Signal_Decomposed_Amplitude \
#             = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
#         Array2D_Signal_Decomposed_Phase \
#             = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
#         for i_Current_Node in range(2**Value_Maximum_Level):
#             Tuple_Function_Return \
#                 = self.Function_Normalized_Hilbert_Transform\
#                         (Array2D_Signal_Decomposed[:,i_Current_Node])
#             Array_Normalized_Hilbert_Transform, Array_Normalize_Envelop \
#                 = Tuple_Function_Return[0:2]
#             Array_Temp_Signal_Subcomponent_Amplitude \
#                 = Array_Normalize_Envelop
#             Array_Temp_Signal_Subcomponent_Phase \
#                 = numpy.angle(Array_Normalized_Hilbert_Transform)
#             Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node] \
#                 = Array_Temp_Signal_Subcomponent_Amplitude
#             Array2D_Signal_Decomposed_Phase[:,i_Current_Node] \
#                 = numpy.unwrap(Array_Temp_Signal_Subcomponent_Phase)
#         Array2D_Signal_Decomposed_Frequency \
#             = numpy.diff(Array2D_Signal_Decomposed_Phase, axis = 0) \
#                 / numpy.diff(Array_Time).mean() / 2 / numpy.pi
#         Array2D_Signal_Decomposed_Frequency \
#             = numpy.append(\
#                     numpy.zeros((1, Array2D_Signal_Decomposed_Phase.shape[1])),\
#                     Array2D_Signal_Decomposed_Frequency, axis = 0)
#         Value_Frequency_Band_Width \
#             = Value_Default_Sample_Frequency \
#                 / Array2D_Signal_Decomposed.shape[1] / 2
#         Array_Center_Frequency \
#             = numpy.linspace(Value_Frequency_Band_Width, \
#                                 Value_Default_Sample_Frequency / 2, \
#                                 Array2D_Signal_Decomposed.shape[1])
#         Temp_Int_Number_Nodes = Array2D_Signal_Decomposed_Amplitude.shape[1]
#         for i_Current_Node in range(Temp_Int_Number_Nodes):
#             Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node] \
#                 = self.Function_Low_Pass_Filter(\
#                         Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node])  
#         return Array2D_Signal_Decomposed, \
#                 Array2D_Signal_Decomposed_Amplitude, \
#                 Array2D_Signal_Decomposed_Phase, \
#                 Array2D_Signal_Decomposed_Frequency, \
#                 Array_Center_Frequency

#     def Fundtion_Surrogate_Data_Generation(self, \
#             Array_Time, Array_Signal, Int_Number_of_Surrogates):
#         Array_Signal_FFT = numpy.fft.fft(Array_Signal)
#         Array_Signal_FFT_Amplitude = numpy.abs(Array_Signal_FFT)
#         Array_Signal_FFT_Phase = numpy.angle(Array_Signal_FFT)
#         Array2D_Signal_Surrogate \
#             = numpy.zeros((Array_Signal.size, Int_Number_of_Surrogates), \
#                             dtype = numpy.complex_)
#         for i_Surrogate in range(Int_Number_of_Surrogates):
#             Array_Signal_FFT_Phase_Generated \
#                 = numpy.random.rand(Array_Signal_FFT_Phase.size) \
#                         * 2 * numpy.pi \
#                     - numpy.pi
#             Array_Signal_FFT_Phase_Generated[0] = 0
#             if Array_Signal_FFT_Phase_Generated.size % 2 == 0:
#                 Array_Signal_FFT_Phase_Generated\
#                 [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
#                 = -numpy\
#                     .flipud(\
#                         Array_Signal_FFT_Phase_Generated\
#                         [1: int(Array_Signal_FFT_Phase_Generated.size / 2) + 0])
#             else:
#                 Array_Signal_FFT_Phase_Generated\
#                     [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
#                     = -numpy.flipud(Array_Signal_FFT_Phase_Generated\
#                         [1: int(Array_Signal_FFT_Phase_Generated.size / 2) + 1])
#             Array2D_Signal_Surrogate[:,i_Surrogate] \
#                 = numpy\
#                     .fft.ifft(Array_Signal_FFT_Amplitude \
#                             * numpy.exp(1j * Array_Signal_FFT_Phase_Generated))
#         Array2D_Signal_Surrogate = Array2D_Signal_Surrogate.real
#         return Array2D_Signal_Surrogate

#     def Fundtion_Surrogate_Data_Generation_With_Padding(self, \
#             Array_Time, Array_Signal, Int_Number_of_Surrogates, \
#             Int_Number_Prediction):
#         Array_Signal_FFT = numpy.fft.fft(Array_Signal)
#         Array_Signal_FFT_Amplitude = numpy.abs(Array_Signal_FFT)
#         Array_Signal_FFT_Phase = numpy.angle(Array_Signal_FFT)
#         Array2D_Signal_Surrogate \
#             = numpy.zeros((Array_Signal.size, Int_Number_of_Surrogates), \
#                             dtype = numpy.complex_)
#         for i_Surrogate in range(Int_Number_of_Surrogates):
#             Array_Signal_FFT_Phase_Generated \
#                 = numpy.random.rand(Array_Signal_FFT_Phase.size) \
#                     * 2 * numpy.pi - numpy.pi
#             Array_Signal_FFT_Phase_Generated[0] = 0
#             if Array_Signal_FFT_Phase_Generated.size % 2 == 0:
#                 Array_Signal_FFT_Phase_Generated\
#                     [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
#                         = -numpy.flipud(Array_Signal_FFT_Phase_Generated\
#                             [1: int(Array_Signal_FFT_Phase_Generated.size / 2) \
#                                     + 0])
#             else:
#                 Array_Signal_FFT_Phase_Generated\
#                     [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
#                         = -numpy.flipud(Array_Signal_FFT_Phase_Generated\
#                             [1: int(Array_Signal_FFT_Phase_Generated.size / 2) \
#                                     + 1])
#             Array2D_Signal_Surrogate[:,i_Surrogate] \
#                 = numpy.fft.ifft(Array_Signal_FFT_Amplitude * numpy.exp( 1j \
#                     * Array_Signal_FFT_Phase_Generated))
#         Array2D_Signal_Surrogate = Array2D_Signal_Surrogate.real
#         if Int_Number_Prediction == 0:
#             pass
#         else:
#             Array2D_Signal_Surrogate \
#                 = numpy.append(Array2D_Signal_Surrogate, \
#                                 Array2D_Signal_Surrogate\
#                                     [:Int_Number_Prediction * 2,:], \
#                                 axis = 0)
#         return Array2D_Signal_Surrogate

#     def Function_Signal_Padding(self, \
#             Array_Time, Array_Signal, Int_Number_Prediction):
#         """
#         Have minor bug
#         Should be replaced with "Function_Mirror_Signal_Padding"
#         Kept for the compatibility of old functions
#         ------------------------------------------------------------------------
#         Input:
#             Array_Time
#             Array_Signal
#             Int_Number_Prediction
#         ------------------------------------------------------------------------
#         Output:
#             Array_Time_With_Pad
#             Array_Signal_With_Pad
#         """
#         Array_Signal_With_Pad \
#             = numpy.zeros(Array_Signal.size + Int_Number_Prediction * 2)
#         Temp_Value_Mean = Array_Signal[0:Int_Number_Prediction].mean()
#         Array_Signal_With_Pad[0 :Int_Number_Prediction - 1] \
#             = - numpy.flipud(Array_Signal[0:Int_Number_Prediction - 1]) \
#                 + Temp_Value_Mean * 2
#         Array_Signal_With_Pad[Int_Number_Prediction - 1] = Temp_Value_Mean
#         Array_Signal_With_Pad\
#             [Int_Number_Prediction:Int_Number_Prediction + Array_Signal.size] \
#                 = Array_Signal
#         Temp_Value_Mean = Array_Signal[- Int_Number_Prediction:].mean()
#         Array_Signal_With_Pad\
#             [Int_Number_Prediction + Array_Signal.size + 1\
#             : Array_Signal_With_Pad.size] \
#                 = - numpy.flipud(Array_Signal[- Int_Number_Prediction + 1:]) \
#                     + Temp_Value_Mean * 2
#         Array_Signal_With_Pad[Int_Number_Prediction + Array_Signal.size] \
#             = Temp_Value_Mean
#         Array_Time_With_Pad \
#             = numpy.zeros(Array_Time.size + Int_Number_Prediction * 2)
#         Value_Delta_t = numpy.diff(Array_Time).mean()
#         Array_Time_With_Pad\
#             = numpy.linspace\
#                 (Array_Time.min() - Value_Delta_t * Int_Number_Prediction, \
#                 Array_Time.max() + Value_Delta_t * Int_Number_Prediction, \
#                 Array_Time.size + Int_Number_Prediction * 2)
#         return Array_Time_With_Pad, Array_Signal_With_Pad
    
#     def Function_Mirror_Signal_Padding(self, \
#             Array_Time, Array_Signal, Int_Number_Prediction):
#         """
#         Mirror padding of the original signal:
#             if Prediction == 0, then return original arrays
#             if Prediction >=1,  then do mirror padding
#         ------------------------------------------------------------------------
#         Input:
#             Array_Time
#             Array_Signal
#             Int_Number_Prediction
#         ------------------------------------------------------------------------
#         Output:
#             Array_Time_With_Pad
#             Array_Signal_With_Pad
#         """
#         if Int_Number_Prediction <= 0:
#             Array_Time_With_Pad, Array_Signal_With_Pad \
#                 = Array_Time, Array_Signal
#         else:
#             Array_Signal_With_Pad \
#                 = numpy.zeros(Array_Signal.size + Int_Number_Prediction * 2)
#             Temp_Value_End = Array_Signal[0]
#             Array_Signal_With_Pad[0:Int_Number_Prediction] \
#                 = - numpy.flipud(Array_Signal[1:Int_Number_Prediction + 1]) \
#                     + Temp_Value_End * 2
#             Array_Signal_With_Pad\
#                 [Int_Number_Prediction \
#                 :Int_Number_Prediction + Array_Signal.size] \
#                     = Array_Signal
#             Array_Signal_With_Pad\
#                 [Int_Number_Prediction + Array_Signal.size\
#                 : Array_Signal_With_Pad.size] \
#                     = - numpy.flipud\
#                         (Array_Signal[- Int_Number_Prediction - 1: -1]) \
#                         + Temp_Value_End * 2
#             Value_Delta_t = Array_Time[1] - Array_Time[0]
#             Array_Time_With_Pad\
#                 = numpy.linspace(\
#                     Array_Time.min() - Value_Delta_t * Int_Number_Prediction, \
#                     Array_Time.max() + Value_Delta_t * Int_Number_Prediction, \
#                     Array_Time.size + Int_Number_Prediction * 2)
#         return Array_Time_With_Pad, Array_Signal_With_Pad

#     def Function_Signal_Depedding_2D(self, Array_Time_With_Pad, \
#         Array2D_Signal_With_Pad, Int_Number_Prediction):
#         """
#         Depadding 2d Array
#         ------------------------------------------------------------------------
#         Output:
#             Array_Time
#             Array2D_Signal
#         """
#         Array_Time \
#             = Array_Time_With_Pad\
#                 [Int_Number_Prediction \
#                     : Array_Time_With_Pad.size - Int_Number_Prediction]
#         Array2D_Signal \
#             = Array2D_Signal_With_Pad\
#                 [Int_Number_Prediction \
#                     : Array_Time_With_Pad.size - Int_Number_Prediction,:]
#         return Array_Time, Array2D_Signal

#     def Function_SVR_Prediction(self,\
#             Array_Time, Array_Signal, Array_Time_Predict):
#         Array2D_X = numpy.reshape(Array_Time, (Array_Time.size, 1))
#         Array_Y = Array_Signal.copy()
#         Array2D_Time_Predict = Array_Time_Predict.reshape(-1,1)
#         Class_svr \
#             = GridSearchCV(SVR(kernel='rbf', gamma=0.1), \
#                             cv = 5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], \
#                             "gamma": numpy.logspace(2, 11, 6)})
#         Class_svr.fit(Array2D_X, Array_Y)
#         self.Array_Signal_Predict = Class_svr.predict(Array2D_Time_Predict)
#         self.Array_Index_sv_ind = Class_svr.best_estimator_.support_
#         return self.Array_Signal_Predict, self.Array_Index_sv_ind

#     def Function_SVR_2D_Training_Whole_Prediction(self, \
#             Array_Time, Array_Signal, Int_Number_Prediction = 10):
#         Array_Time = numpy.copy(Array_Time, order = 'C')
#         Array_Signal = numpy.copy(Array_Signal, order = 'C')
#         Str_CRC32_Signal = zlib.crc32(Array_Signal)
#         Str_Variable_Name_Cache = 'Cache_CRC32' + str(Str_CRC32_Signal)
#         Str_File_Name = 'Cache_SVR_Predicted_Signal.mat'
#         if not os.path.isfile('Data/Simulation_Result/' + Str_File_Name):
#             io.savemat('Data/Simulation_Result/' +Str_File_Name, {})
#         RawData = io.loadmat('Data/Simulation_Result/' + Str_File_Name)
#         List_Variable_Name = list(RawData)
#         for i_Variable_Name in numpy.arange(3, len(List_Variable_Name),1):
#             Str_Command \
#                 = List_Variable_Name[i_Variable_Name] \
#                     + ' = RawData[\'' \
#                     + List_Variable_Name[i_Variable_Name] \
#                     + '\']'
#             exec(Str_Command)
#         if Str_Variable_Name_Cache in List_Variable_Name:
#             Array_Signal_Predict \
#                 = numpy.squeeze(RawData[Str_Variable_Name_Cache][0][2])
#             Array_Time_Predict \
#                 = numpy.squeeze(RawData[Str_Variable_Name_Cache][0][3])
#             return Array_Signal_Predict, Array_Time_Predict
#         else:
#             def Function_SVR_2D_Training_Right_Prediction(\
#                     Array_Time, Array_Signal, Int_Number_Prediction = 10):
#                 Array2D_X = numpy.reshape(Array_Time, (Array_Time.size, 1))
#                 Int_Length_Section = Array_Signal.size // 2
#                 Array2D_X \
#                     = numpy.zeros((Int_Length_Section, Int_Length_Section))
#                 for i_Section in range(Int_Length_Section):
#                     Array2D_X[:,i_Section] \
#                         = Array_Signal[i_Section\
#                                         : i_Section + Int_Length_Section]
#                 Array_Y \
#                     = Array_Signal\
#                         [Int_Length_Section \
#                             : Int_Length_Section + Int_Length_Section].copy()
#                 Class_svr \
#                     = GridSearchCV\
#                         (SVR(kernel='rbf', C = 1e4, gamma=0.0001), \
#                         cv = 5, \
#                         param_grid={"C": [1e0, 10, 1e2, 1e3], \
#                                     "gamma": numpy.logspace(-8, -2, 5)})
#                 Class_svr.fit(Array2D_X, Array_Y)
#                 Temp_Array_Prediction = Array_Signal[-Int_Length_Section:]
#                 Array_Prediction = numpy.zeros(Int_Number_Prediction)
#                 for i_Predict_Time in range(Int_Number_Prediction):
#                     Temp_Value_Signal_Predict \
#                         = Class_svr.predict\
#                             (Temp_Array_Prediction\
#                                 .reshape(1,Int_Length_Section))
#                     Array_Prediction[i_Predict_Time] \
#                         = Temp_Value_Signal_Predict
#                     Temp_Array_Prediction \
#                         = numpy.append(Temp_Array_Prediction[1:], \
#                                         Temp_Value_Signal_Predict)
#                 Array_Signal_Predict = Array_Prediction.copy()
#                 Value_Delta_t = numpy.diff(Array_Time).mean()
#                 Array_Time_Predict \
#                     = numpy.linspace\
#                                 (Array_Time.max() + Value_Delta_t, \
#                                 Array_Time.max() \
#                                     + Int_Number_Prediction * Value_Delta_t, \
#                                 Int_Number_Prediction)
#                 return Array_Signal_Predict, Array_Time_Predict

#             def Function_SVR_2D_Training_Left_Prediction(\
#                     Array_Time, Array_Signal, Int_Number_Prediction = 10):
#                 Array_Signal = numpy.flip(Array_Signal, axis = 0)
#                 Array2D_X = numpy.reshape(Array_Time, (Array_Time.size, 1))
#                 Int_Length_Section = Array_Signal.size // 2
#                 Array2D_X \
#                     = numpy.zeros((Int_Length_Section, Int_Length_Section))
#                 for i_Section in range(Int_Length_Section):
#                     Array2D_X[:,i_Section] \
#                         = Array_Signal\
#                             [i_Section: i_Section + Int_Length_Section]
#                 Array_Y \
#                     = Array_Signal\
#                         [Int_Length_Section \
#                             : Int_Length_Section + Int_Length_Section].copy()
#                 Class_svr \
#                     = GridSearchCV\
#                         (SVR(kernel='rbf', C = 1e4, gamma=0.0001), \
#                         cv = 5, \
#                         param_grid={"C": [1e0, 10, 1e2, 1e3], \
#                         "gamma": numpy.logspace(-8, -2, 5)})
#                 Class_svr.fit(Array2D_X, Array_Y)
#                 Temp_Array_Prediction = Array_Signal[-Int_Length_Section:]
#                 Array_Prediction = numpy.zeros(Int_Number_Prediction)
#                 for i_Predict_Time in range(Int_Number_Prediction):
#                     Temp_Value_Signal_Predict \
#                         = Class_svr\
#                             .predict(Temp_Array_Prediction\
#                                         .reshape(1,Int_Length_Section))
#                     Array_Prediction[i_Predict_Time] \
#                         = Temp_Value_Signal_Predict
#                     Temp_Array_Prediction \
#                         = numpy.append\
#                                     (Temp_Array_Prediction[1:], \
#                                     Temp_Value_Signal_Predict)
#                 Array_Signal_Predict = numpy.flip(Array_Prediction, axis = 0)
#                 Value_Delta_t = numpy.diff(Array_Time).mean()
#                 Array_Time_Predict \
#                     = numpy.linspace(Array_Time.min() \
#                                         - Int_Number_Prediction \
#                                             * Value_Delta_t, \
#                                     Array_Time.min() - Value_Delta_t, \
#                                     Int_Number_Prediction)
#                 return Array_Signal_Predict, Array_Time_Predict

#             Array_Signal_Predict_Right, Array_Time_Predict_Right \
#                 = Function_SVR_2D_Training_Right_Prediction\
#                     (Array_Time[-Int_Number_Prediction * 2:], \
#                         Array_Signal[-Int_Number_Prediction * 2:], \
#                         Int_Number_Prediction)
#             Array_Signal_Predict_Left, Array_Time_Predict_Left \
#                 = Function_SVR_2D_Training_Left_Prediction\
#                     (Array_Time[:Int_Number_Prediction * 2], \
#                         Array_Signal[:Int_Number_Prediction * 2], \
#                         Int_Number_Prediction)
#             Array_Signal_Predict \
#                 = numpy.append(Array_Signal_Predict_Left, Array_Signal)
#             self.Array_Signal_Predict \
#                 = numpy.append(Array_Signal_Predict, Array_Signal_Predict_Right)
#             Array_Time_Predict \
#                 = numpy.append(Array_Time_Predict_Left, Array_Time)
#             self.Array_Time_Predict \
#                 = numpy.append(Array_Time_Predict, Array_Time_Predict_Right)
#             return Array_Signal_Predict, Array_Time_Predict

#     def Function_Nonstationary_Ratio(self, Array_Time, Array_Signal, \
#                                     Str_Wavelet_Name, \
#                                     Int_Number_of_Surrogates, \
#                                     Parameter_Max_WPT_Level):
#         """
#         Calculates the ratio of the local energy to that of the averaged one 
#         """
#         # Signal decompostiion
#         Tuple_Functin_Return \
#             = self.Function_WPT_HT_Decomposition\
#                     (Array_Time, Array_Signal, Str_Wavelet_Name, \
#                         Parameter_Max_WPT_Level)
#         Array2D_Signal_Decomposed, Array2D_Signal_Decomposed_Amplitude, \
#         Array2D_Signal_Decomposed_Frequency, Array_Center_Frequency \
#             = Tuple_Functin_Return[0,1,3,4]
#         # Surrogate data generation
#         Array2D_Signal_Surrogate \
#             = self.Fundtion_Surrogate_Data_Generation\
#                         (Array_Time, Array_Signal, Int_Number_of_Surrogates)
#         Array3D_Surrogate_Decomposed_Amplitude \
#             = numpy.zeros(Array_Time.size, \
#                             Array_Center_Frequency.size, \
#                             Int_Number_of_Surrogates)
#         for i_Surrogate in range(Int_Number_of_Surrogates):
#             Array_Surrogate = Array2D_Signal_Surrogate[:,i_Surrogate]
#             Tuple_Return_WPT_Decomposition \
#                 = self.Function_WPT_HT_Decomposition\
#                         (Array_Time, Array_Surrogate, Str_Wavelet_Name, \
#                         Parameter_Max_WPT_Level)
#             Array2D_Surrogate_Decomposed_Amplitude \
#                 = Tuple_Return_WPT_Decomposition[1]
#             Array_Center_Frequency \
#                 = Tuple_Return_WPT_Decomposition[4]
#             Array3D_Surrogate_Decomposed_Amplitude[:,:,i_Surrogate] \
#                 = Array2D_Surrogate_Decomposed_Amplitude
#         Array2D_Surrogate_Decomposed_Amplitude \
#             = Array3D_Surrogate_Decomposed_Amplitude.mean(axis = 2)
#         Array_Surrogate_Amplitude \
#             = Array2D_Surrogate_Decomposed_Amplitude.mean(axis = 0)
#         Array2D_Surrogate_Decomposed_Amplitude_Mean \
#             = numpy.dot(numpy.ones((Array_Time.size, 1)), \
#                         Array_Surrogate_Amplitude\
#                             .reshape(1, Array_Center_Frequency.size))
#         Array2D_Ratio \
#             = Array2D_Signal_Decomposed_Amplitude \
#                 / Array2D_Surrogate_Decomposed_Amplitude_Mean
#         return Array2D_Ratio, Array2D_Signal_Decomposed, \
#                 Array2D_Signal_Decomposed_Amplitude, \
#                 Array2D_Signal_Decomposed_Frequency, \
#                 Array_Center_Frequency, \
#                 Array3D_Surrogate_Decomposed_Amplitude, \
#                 Array2D_Signal_Surrogate

#     def Function_Local_Nonstationary_Index(self, \
#             Array_Time_Predict, Array_Signal_Predict, \
#             Int_Length_Prediction, \
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level):
#         """
#         Calculates the ratio of the local energy to that of the averaged one 
#         ------------------------------------------------------------------------
#         Input:
#             Int_Length_Prediction: Single side prediction length
#         ------------------------------------------------------------------------
#         Output:
#             Array2D_Local_Nonstationary_Index
#             Array2D_Ratio
#             Array2D_Signal_Decomposed
#             Array2D_Signal_Decomposed_Amplitude
#             Array2D_Signal_Decomposed_Frequency
#             Array_Center_Frequency
#         """
#         # Signal decompostiion
#         Tuple_Functin_Return \
#             = self.Function_WPT_HT_Decomposition(\
#                     Array_Time_Predict, Array_Signal_Predict, \
#                     Str_Wavelet_Name, Parameter_Max_WPT_Level)
#         Array2D_Signal_Predict_Decomposed, \
#         Array2D_Signal_Predict_Decomposed_Amplitude, \
#             = Tuple_Functin_Return[0:2]
#         Array2D_Signal_Predict_Decomposed_Frequency, Array_Center_Frequency \
#             = Tuple_Functin_Return[3:5]
#         Array_Time = Array_Time_Predict[Int_Length_Prediction \
#                     :Array_Time_Predict.size - Int_Length_Prediction]
#         Array2D_Signal_Decomposed \
#             = Array2D_Signal_Predict_Decomposed\
#                 [Int_Length_Prediction \
#                     :Array_Time_Predict.size - Int_Length_Prediction, :]
#         Array2D_Signal_Decomposed_Amplitude \
#             = Array2D_Signal_Predict_Decomposed_Amplitude\
#                 [Int_Length_Prediction \
#                     :Array_Time_Predict.size - Int_Length_Prediction, :]
#         Array2D_Signal_Decomposed_Frequency \
#             = Array2D_Signal_Predict_Decomposed_Frequency\
#                 [Int_Length_Prediction \
#                     :Array_Time_Predict.size - Int_Length_Prediction, :]
#         Array_Signal_Decomposed_Amplitude_Mean \
#             = Array2D_Signal_Decomposed_Amplitude.mean(axis = 0)
#         Array2D_Signal_Decomposed_Amplitude_Mean\
#             = numpy.dot(numpy.ones((Array_Time.size, 1)), \
#                         Array_Signal_Decomposed_Amplitude_Mean\
#                             .reshape(1, Array_Center_Frequency.size))
#         Array2D_Ratio \
#             = Array2D_Signal_Decomposed_Amplitude \
#                 / Array2D_Signal_Decomposed_Amplitude_Mean
#         Array2D_Local_Nonstationary_Index = numpy.abs(Array2D_Ratio - 1)
#         return Array2D_Local_Nonstationary_Index, \
#                 Array2D_Ratio, \
#                 Array2D_Signal_Decomposed, \
#                 Array2D_Signal_Decomposed_Amplitude, \
#                 Array2D_Signal_Decomposed_Frequency, \
#                 Array_Center_Frequency

#     def Function_Global_Nonstationary_Index(self, \
#             Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level):
#         """
#         Calculate the global nonstationary index
#         ------------------------------------------------------------------------
#         Output:
#             Value_Global_Nonstationary_Index
#             Array2D_Signal_Decomposed_Amplitude
#         """
#         Tuple_Function_Return \
#             = self.Function_Local_Nonstationary_Index(\
#                     Array_Time_Predict, Array_Signal_Predict, \
#                     Int_Length_Prediction, Str_Wavelet_Name, \
#                     Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
#         Array2D_Local_Nonstationary_Index = Tuple_Function_Return[0]
#         Array2D_Signal_Decomposed_Amplitude = Tuple_Function_Return[3]
#         # Calculate the global nonstationary index
#         Value_Global_Nonstationary_Index \
#             = numpy.sum(Array2D_Local_Nonstationary_Index \
#                 * Array2D_Signal_Decomposed_Amplitude) \
#                 / numpy.sum(Array2D_Signal_Decomposed_Amplitude)
#         return Value_Global_Nonstationary_Index, \
#                 Array2D_Signal_Decomposed_Amplitude

#     def Function_Global_Nonstationary_Index_With_FRF(self, \
#             Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level, \
#             Value_Damping_Ratio, Value_Mass, Value_Stiffness):
#         """
#         Calculate the global nonstationary index
#         ------------------------------------------------------------------------
#         Output:
#             Value_Global_Nonstationary_Index
#             Array2D_Signal_Decomposed_Amplitude
#         """
#         # Create structural response object
#         Object_Structural_Response = Class_Structural_Response()
#         # Calculation
#         Tuple_Function_Return \
#             = self.Function_Local_Nonstationary_Index(\
#                     Array_Time_Predict, Array_Signal_Predict, \
#                     Int_Length_Prediction, Str_Wavelet_Name, \
#                     Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
#         Array2D_Local_Nonstationary_Index = Tuple_Function_Return[0]
#         Array2D_Signal_Decomposed_Amplitude = Tuple_Function_Return[3]
#         Array_Center_Frequency = Tuple_Function_Return[5]
#         # FRF
#         Array_Frequency_Response \
#             = Object_Structural_Response\
#                 .Function_SDOF_Frequency_Response\
#                     (Array_Center_Frequency, \
#                     Value_Damping_Ratio, Value_Mass, Value_Stiffness)
#         Array2D_Frequency_Response \
#             = numpy.repeat(Array_Frequency_Response.reshape(1, -1), \
#                             Array2D_Local_Nonstationary_Index.shape[0], \
#                             axis = 0)
#         # Calculate the global nonstationary index
#         Value_Global_Nonstationary_Index \
#             = numpy.sum(Array2D_Local_Nonstationary_Index \
#                 * Array2D_Signal_Decomposed_Amplitude \
#                 * Array2D_Frequency_Response) \
#                 / numpy.sum(Array2D_Signal_Decomposed_Amplitude \
#                             * Array2D_Frequency_Response)
#         return Value_Global_Nonstationary_Index, \
#                 Array2D_Signal_Decomposed_Amplitude

#     def Function_Relative_Nonstationary_Index(self, \
#             Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level):
#         """
#         Calculate the global nonstationary index
#         """
#         # Get original data
#         Array_Time \
#             = Array_Time_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]
#         Array_Signal \
#             = Array_Signal_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]
#         # Get the corresponding impulse data
#         # Array_Impulse = numpy.zeros(Array_Time_Predict.size)
#         # Value_Energy = numpy.sum(Array_Signal**2)
#         # Int_Mid_Index = int(Array_Time.size / 2)
#         # Array_Impulse[Int_Mid_Index] \
#         #     = numpy.sqrt(Value_Energy / 2)
#         # Array_Impulse[Int_Mid_Index + 1] \
#         #     = - Array_Impulse[Int_Mid_Index]
#         Object_Signal_Generation = Class_Signal_Generation()
#         Tuple_Function_Return \
#             = Object_Signal_Generation\
#                 .Function_Impulse_Signal_Generation_Oscilate_Delta\
#                     (Array_Time_Predict, \
#                     Array_Signal_Predict, \
#                     Int_Length_Prediction)
#         Array_Impulse_Predict = Tuple_Function_Return[1]
#         # Calculate the global nonstationary index of the analyzed signal
#         Tuple_Function_Return \
#             = self.Function_Global_Nonstationary_Index(\
#                     Array_Time_Predict, Array_Signal_Predict, \
#                     Int_Length_Prediction, \
#                     Str_Wavelet_Name, Int_Number_of_Surrogates, \
#                     Parameter_Max_WPT_Level)
#         Value_Global_Nonstationary_Index_Signal = Tuple_Function_Return[0]
#         # Calculate the global nonstationary index of the impulse signal
#         Tuple_Function_Return \
#             = self.Function_Global_Nonstationary_Index(\
#                     Array_Time_Predict, Array_Impulse_Predict, \
#                     Int_Length_Prediction, \
#                     Str_Wavelet_Name, Int_Number_of_Surrogates, \
#                     Parameter_Max_WPT_Level)
#         Value_Global_Nonstationary_Index_Impulse = Tuple_Function_Return[0]
#         Value_Relative_Nonstationary_Index \
#             = Value_Global_Nonstationary_Index_Signal \
#                 / Value_Global_Nonstationary_Index_Impulse
#         return Value_Relative_Nonstationary_Index, Array_Impulse_Predict

#     def Function_Relative_Nonstationary_Index_With_FRF(self, \
#             Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level, \
#             Value_Damping_Ratio, Value_Mass, Value_Stiffness):
#         """
#         Calculate the global nonstationary index
#         """
#         # Get original data
#         # Array_Time \
#         #     = Array_Time_Predict\
#         #         [Int_Length_Prediction \
#         #             : Array_Time_Predict.size - Int_Length_Prediction]
#         # Array_Signal \
#         #     = Array_Signal_Predict\
#         #         [Int_Length_Prediction \
#         #             : Array_Time_Predict.size - Int_Length_Prediction]
#         # Get the corresponding impulse data
#         # Array_Impulse = numpy.zeros(Array_Time_Predict.size)
#         # Value_Energy = numpy.sum(Array_Signal**2)
#         # Int_Mid_Index = int(Array_Time.size / 2)
#         # Array_Impulse[Int_Mid_Index] \
#         #     = numpy.sqrt(Value_Energy / 2)
#         # Array_Impulse[Int_Mid_Index + 1] \
#         #     = - Array_Impulse[Int_Mid_Index]
#         Object_Signal_Generation = Class_Signal_Generation()
#         Tuple_Function_Return \
#             = Object_Signal_Generation\
#                 .Function_Impulse_Signal_Generation_Oscilate_Delta\
#                     (Array_Time_Predict, \
#                     Array_Signal_Predict, \
#                     Int_Length_Prediction)
#         Array_Impulse_Predict = Tuple_Function_Return[1]
#         # Calculate the global nonstationary index of the analyzed signal
#         Tuple_Function_Return \
#             = self.Function_Global_Nonstationary_Index_With_FRF(\
#                     Array_Time_Predict, Array_Signal_Predict, \
#                     Int_Length_Prediction, \
#                     Str_Wavelet_Name, Int_Number_of_Surrogates, \
#                     Parameter_Max_WPT_Level, \
#                     Value_Damping_Ratio, Value_Mass, Value_Stiffness)
#         Value_Global_Nonstationary_Index_Signal = Tuple_Function_Return[0]
#         # Calculate the global nonstationary index of the impulse signal
#         Tuple_Function_Return \
#             = self.Function_Global_Nonstationary_Index_With_FRF(\
#                     Array_Time_Predict, Array_Impulse_Predict, \
#                     Int_Length_Prediction, \
#                     Str_Wavelet_Name, Int_Number_of_Surrogates, \
#                     Parameter_Max_WPT_Level, \
#                     Value_Damping_Ratio, Value_Mass, Value_Stiffness)
#         Value_Global_Nonstationary_Index_Impulse = Tuple_Function_Return[0]
#         Value_Relative_Nonstationary_Index \
#             = Value_Global_Nonstationary_Index_Signal \
#                 / Value_Global_Nonstationary_Index_Impulse
#         return Value_Relative_Nonstationary_Index, Array_Impulse_Predict

#     def Function_Relative_Nonstationary_Index_With_Distribution(self, \
#             Array_Time_Predict, Array_Signal_Predict, \
#             Int_Length_Prediction, 
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level):
#         Array_Time \
#             = Array_Time_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]
#         Array_Signal \
#             = Array_Signal_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]
#         Tuple_Function_Return \
#             = self.Function_Relative_Nonstationary_Index(\
#                 Array_Time_Predict, Array_Signal_Predict, \
#                 Int_Length_Prediction, \
#                 Str_Wavelet_Name, Int_Number_of_Surrogates, \
#                 Parameter_Max_WPT_Level)
#         Value_Nonstationary_Index = Tuple_Function_Return[0]
#         if Int_Number_of_Surrogates == 0:
#             Array_Surrogate_Ratio_Sorted = 0
#             Array2D_Signal_Surrogate_Prediction = 0
#         else:
#             Array2D_Signal_Surrogate_Prediction \
#                 = self.Fundtion_Surrogate_Data_Generation_With_Padding(\
#                         Array_Time, Array_Signal, Int_Number_of_Surrogates, \
#                         Int_Length_Prediction)
#             Array_Relative_Nonstationary_Index_Surrogate \
#                 = numpy.zeros(Int_Number_of_Surrogates)
#             for i_Surrogate in range(Int_Number_of_Surrogates):
#                 Array_Surrogate_Prediction \
#                     = Array2D_Signal_Surrogate_Prediction[:,i_Surrogate]
#                 # Tuple_Return_WPT_Decomposition \
#                 #     = self.Function_WPT_HT_Decomposition(\
#                 #             Array_Time, Array_Surrogate, Str_Wavelet_Name, \
#                 #             Parameter_Max_WPT_Level)
#                 Tuple_Function_Return \
#                     = self.Function_Relative_Nonstationary_Index(\
#                         Array_Time_Predict, Array_Surrogate_Prediction, \
#                         Int_Length_Prediction, \
#                         Str_Wavelet_Name,  Int_Number_of_Surrogates , \
#                         Parameter_Max_WPT_Level)
#                 Array_Relative_Nonstationary_Index_Surrogate[i_Surrogate] \
#                     = Tuple_Function_Return[0]
#             Array_Surrogate_Ratio_Sorted \
#                 = numpy.sort(Array_Relative_Nonstationary_Index_Surrogate)
#         return Value_Nonstationary_Index, Array_Surrogate_Ratio_Sorted, \
#                 Array2D_Signal_Surrogate_Prediction

#     def Function_Relative_Nonstationary_Index_With_Distribution_FRF(self, \
#             Array_Time_Predict, Array_Signal_Predict, \
#             Int_Length_Prediction, 
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level, \
#             Value_Damping_Ratio, Value_Mass, Value_Stiffness):
#         Array_Time \
#             = Array_Time_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]
#         Array_Signal \
#             = Array_Signal_Predict\
#                 [Int_Length_Prediction \
#                     : Array_Time_Predict.size - Int_Length_Prediction]
#         Tuple_Function_Return \
#             = self.Function_Relative_Nonstationary_Index_With_FRF(\
#                 Array_Time_Predict, Array_Signal_Predict, \
#                 Int_Length_Prediction, \
#                 Str_Wavelet_Name, Int_Number_of_Surrogates, \
#                 Parameter_Max_WPT_Level, \
#                 Value_Damping_Ratio, Value_Mass, Value_Stiffness)
#         Value_Nonstationary_Index = Tuple_Function_Return[0]
#         if Int_Number_of_Surrogates == 0:
#             Array_Surrogate_Ratio_Sorted = 0
#             Array2D_Signal_Surrogate_Prediction = 0
#         else:
#             Array2D_Signal_Surrogate_Prediction \
#                 = self.Fundtion_Surrogate_Data_Generation_With_Padding(\
#                         Array_Time, Array_Signal, Int_Number_of_Surrogates, \
#                         Int_Length_Prediction)
#             Array_Relative_Nonstationary_Index_Surrogate \
#                 = numpy.zeros(Int_Number_of_Surrogates)
#             for i_Surrogate in range(Int_Number_of_Surrogates):
#                 Array_Surrogate_Prediction \
#                     = Array2D_Signal_Surrogate_Prediction[:,i_Surrogate]
#                 # Tuple_Return_WPT_Decomposition \
#                 #     = self.Function_WPT_HT_Decomposition(\
#                 #             Array_Time, Array_Surrogate, Str_Wavelet_Name, \
#                 #             Parameter_Max_WPT_Level)
#                 Tuple_Function_Return \
#                     = self.Function_Relative_Nonstationary_Index_With_FRF(\
#                         Array_Time_Predict, Array_Surrogate_Prediction, \
#                         Int_Length_Prediction, \
#                         Str_Wavelet_Name,  Int_Number_of_Surrogates , \
#                         Parameter_Max_WPT_Level, \
#                         Value_Damping_Ratio, Value_Mass, Value_Stiffness)
#                 Array_Relative_Nonstationary_Index_Surrogate[i_Surrogate] \
#                     = Tuple_Function_Return[0]
#             Array_Surrogate_Ratio_Sorted \
#                 = numpy.sort(Array_Relative_Nonstationary_Index_Surrogate)
#         return Value_Nonstationary_Index, Array_Surrogate_Ratio_Sorted, \
#                 Array2D_Signal_Surrogate_Prediction

#     def Function_Nonstationary_Index_Pad_Depad_Remove_Prediction(self, \
#             Array_Time_Predict, Array_Signal_Predict, \
#             Int_Number_Prediction, 
#             Str_Wavelet_Name, Int_Number_of_Surrogates, \
#             Parameter_Max_WPT_Level):
#         Object_Signal_Generation = Class_Signal_Generation()
#         class Signal_Info:
#             # Collect the information of the signal including the decomposed 
#             # signal, amplitudes, surrogates etc.
#             def __init__(self, \
#                         Array2D_Signal_Surrogate, \
#                         Array2D_Signal_Decomposed_Amplitude, \
#                         Array2D_Signal_Decomposed, \
#                         Array2D_Surrogate_Decomposed_Amplitude_Mean):
#                 self.Array2D_Surrogate = Array2D_Signal_Surrogate
#                 self.Array2D_Decomposed_Amplitude \
#                         = Array2D_Signal_Decomposed_Amplitude
#                 self.Array2D_Decomposed = Array2D_Signal_Decomposed
#                 self.Array2D_Surrogate_Decomposed_Amplitude_Mean \
#                         = Array2D_Surrogate_Decomposed_Amplitude_Mean

#         # Generate corresponding impulse signal data
#         Array_Signal_Impulse \
#             = Object_Signal_Generation\
#                 .Function_Impulse_Signal_Generation(Array_Signal_Predict)
#         Tuple_Function_Return \
#             = self.Function_Nonstationary_Ratio\
#                         (Array_Time_Predict, Array_Signal_Predict, \
#                         Str_Wavelet_Name, \
#                         Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
#         Array2D_With_Pad_Ratio = Tuple_Function_Return[0]
#         Array2D_With_Pad_Signal_Decomposed = Tuple_Function_Return[1]
#         Array2D_With_Pad_Signal_Decomposed_Amplitude = Tuple_Function_Return[3]
#         Array2D_With_Pad_Signal_Decomposed_Frequency = Tuple_Function_Return[4]
#         Array_With_Pad_Center_Frequency = Tuple_Function_Return[5]
#         Array3D_Signal_Surrogate_Decomposed_Amplitude = Tuple_Function_Return[6]
#         Array2D_Signal_Surrogate = Tuple_Function_Return[7]

#         Array2D_With_Pad_Signal_Decomposed, \
#         Array2D_With_Pad_Signal_Decomposed_Amplitude, \
#         Array2D_With_Pad_Signal_Decomposed_Phase, \
#         Array2D_With_Pad_Signal_Decomposed_Frequency, \
#         Array_With_Pad_Center_Frequency \
#             = self.Function_WPT_HT_Decomposition\
#                         (Array_Time_Predict, Array_Signal_Predict, \
#                         Str_Wavelet_Name, Parameter_Max_WPT_Level)
#         # Mirror padding impulse and obtain the Hilbert spectrum
#         Array2D_With_Pad_Impulse_Ratio, \
#         Array2D_With_Pad_Signal_Impulse_Decomposed, \
#         Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude, \
#         Array2D_With_Pad_Signal_Impulse_Decomposed_Frequency, \
#         Array_With_Pad_Impulse_Center_Frequency, \
#         Array3D_Impulse_Surrogate_Decomposed_Amplitude, \
#         Array2D_Impulse_Surrogate \
#             = self.Function_Nonstationary_Ratio\
#                         (Array_Time_Predict, Array_Signal_Impulse, \
#                         Str_Wavelet_Name, \
#                         Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
#         Array2D_With_Pad_Signal_Impulse_Decomposed, \
#         Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude, \
#         Array2D_With_Pad_Signal_Impulse_Decomposed_Phase, \
#         Array2D_With_Pad_Signal_Impulse_Decomposed_Frequency, \
#         Array_With_Pad_Impulse_Center_Frequency \
#             = self.Function_WPT_HT_Decomposition\
#                     (Array_Time_Predict, Array_Signal_Impulse, \
#                     Str_Wavelet_Name, Parameter_Max_WPT_Level)
#         # Surrogate data generation for signal and the Hilbert spectrum 
#         Array2D_Surrogate_Decomposed_Amplitude \
#             = Array3D_Signal_Surrogate_Decomposed_Amplitude.mean(axis = 2)
#         Array_Surrogate_Amplitude \
#             = Array2D_Surrogate_Decomposed_Amplitude.mean(axis = 0)
#         Array2D_Surrogate_Decomposed_Amplitude_Mean \
#             = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
#                         Array_Surrogate_Amplitude\
#                             .reshape(1, Array_With_Pad_Center_Frequency.size))
#         Array2D_With_Pad_Surrogate_Decomposed_Amplitude_Mean \
#             = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
#                         Array_Surrogate_Amplitude\
#                             .reshape(1, Array_With_Pad_Center_Frequency.size))
#         # Surrogate data generation for impulse and the Hilbert spectrum
#         Array2D_Impulse_Surrogate_Decomposed_Amplitude \
#             = Array3D_Impulse_Surrogate_Decomposed_Amplitude.mean(axis = 2)
#         Array_Impulse_Surrogate_Amplitude \
#             = Array2D_Impulse_Surrogate_Decomposed_Amplitude.mean(axis = 0)
#         Array2D_Impulse_Surrogate_Decomposed_Amplitude_Mean \
#             = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
#                         Array_Impulse_Surrogate_Amplitude\
#                             .reshape(1, Array_With_Pad_Center_Frequency.size))
#         Array2D_With_Pad_Impulse_Surrogate_Decomposed_Amplitude_Mean \
#             = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
#                         Array_Impulse_Surrogate_Amplitude\
#                             .reshape(1, Array_With_Pad_Center_Frequency.size))
#         # Get raw ratio
#         Array2D_With_Pad_Ratio \
#             = Array2D_With_Pad_Signal_Decomposed_Amplitude \
#                 / Array2D_With_Pad_Surrogate_Decomposed_Amplitude_Mean
#         Array2D_With_Pad_Impulse_Ratio \
#             = Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude \
#                 / Array2D_With_Pad_Impulse_Surrogate_Decomposed_Amplitude_Mean
#         # Depadding
#         Int_Padded_Number = Int_Number_Prediction
#         Array_With_Pad_Depadded_Time, \
#         Array2D_With_Pad_Depadded_Signal_Decomposed \
#             = self.Function_Signal_Depedding_2D\
#                     (Array_Time_Predict, \
#                     Array2D_With_Pad_Signal_Decomposed, \
#                     Int_Padded_Number)
#         Array_With_Pad_Depadded_Time, \
#         Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude \
#             = self.Function_Signal_Depedding_2D\
#                     (Array_Time_Predict, \
#                     Array2D_With_Pad_Signal_Decomposed_Amplitude, \
#                     Int_Padded_Number)
#         Array_With_Pad_Depadded_Time, Array2D_With_Pad_Depadded_Ratio \
#             = self.Function_Signal_Depedding_2D\
#                     (Array_Time_Predict, Array2D_With_Pad_Ratio, \
#                     Int_Padded_Number)
#         Array_With_Pad_Depadded_Time, \
#         Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed \
#             = self.Function_Signal_Depedding_2D\
#                     (Array_Time_Predict, \
#                     Array2D_With_Pad_Signal_Impulse_Decomposed, \
#                     Int_Padded_Number)
#         Array_With_Pad_Depadded_Time, \
#         Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude \
#             = self.Function_Signal_Depedding_2D\
#                     (Array_Time_Predict, \
#                     Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude, \
#                     Int_Padded_Number)
#         Array_With_Pad_Depadded_Time, \
#         Array2D_With_Pad_Depadded_Impulse_Ratio \
#             = self.Function_Signal_Depedding_2D\
#                     (Array_Time_Predict, \
#                     Array2D_With_Pad_Impulse_Ratio, \
#                     Int_Padded_Number)
#         # Normalization by minusing one and weighting by the spectrum
#         Array2D_With_Pad_Depadded_Normalized_Ratio \
#             = (Array2D_With_Pad_Depadded_Ratio - 1) \
#                 * Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude \
#                 / Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude\
#                     .sum().sum()

#         Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio \
#             = (Array2D_With_Pad_Depadded_Impulse_Ratio - 1) \
#             * Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude \
#             / Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude\
#                 .sum().sum()
#         # Remove negative value as they are treated as stationary part
#         Temp_Int_Number_Cols \
#             = Array2D_With_Pad_Depadded_Normalized_Ratio.shape[1]
#         Temp_Int_Number_Rows \
#             = Array2D_With_Pad_Depadded_Normalized_Ratio.shape[0]
#         for i_Col in range(Temp_Int_Number_Cols):
#             for i_Row in range(Temp_Int_Number_Rows):
#                 if Array2D_With_Pad_Depadded_Normalized_Ratio\
#                         [i_Row, i_Col] <= 0:
#                     Array2D_With_Pad_Depadded_Normalized_Ratio\
#                         [i_Row, i_Col] = 0
#                 if Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio\
#                         [i_Row, i_Col] <= 0:
#                     Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio\
#                         [i_Row, i_Col] = 0
#         # Distribution and threshold value of the surrogate data
#         Array3D_Normalized_Surrogate_Ratio \
#             = numpy.zeros((Array_With_Pad_Depadded_Time.size, \
#                             Array_With_Pad_Center_Frequency.size, \
#                             Int_Number_of_Surrogates))
#         for i_Surrogate in range(Int_Number_of_Surrogates):
#             Temp_Array2D_Surrogate_Ratio \
#                 = Array3D_Signal_Surrogate_Decomposed_Amplitude\
#                         [:,:,i_Surrogate] \
#                     / Array2D_Surrogate_Decomposed_Amplitude_Mean
#             Temp_Array2D_Surrogate_Normalized_Ratio \
#                 = (Temp_Array2D_Surrogate_Ratio - 1) \
#                     * Array3D_Signal_Surrogate_Decomposed_Amplitude\
#                         [:,:,i_Surrogate] \
#                     / Array3D_Signal_Surrogate_Decomposed_Amplitude\
#                         [:,:,i_Surrogate].sum().sum()
#             Temp_Int_Number_Cols \
#                 = Temp_Array2D_Surrogate_Normalized_Ratio.shape[1]
#             Temp_Int_Number_Rows \
#                 = Temp_Array2D_Surrogate_Normalized_Ratio.shape[0]
#             for i_Col in range(Temp_Int_Number_Cols):
#                 for i_Row in range(Temp_Int_Number_Rows):
#                     if Temp_Array2D_Surrogate_Normalized_Ratio\
#                             [i_Row, i_Col] <= 0:
#                         Temp_Array2D_Surrogate_Normalized_Ratio\
#                             [i_Row, i_Col] = 0
#             Array_With_Pad_Depadded_Time, \
#             Temp_Array2D_Surrogate_Normalized_Ratio_Depadded \
#                 = self.Function_Signal_Depedding_2D\
#                         (Array_Time_Predict, \
#                         Temp_Array2D_Surrogate_Normalized_Ratio, \
#                         Int_Padded_Number)
#             Array3D_Normalized_Surrogate_Ratio[:,:,i_Surrogate] \
#                 = Temp_Array2D_Surrogate_Normalized_Ratio_Depadded.copy()
#         Array_Surrogate_Ratio = Array3D_Normalized_Surrogate_Ratio.reshape(-1)
#         Array_Surrogate_Ratio_Sorted = numpy.sort(Array_Surrogate_Ratio)
#         Array_With_Pad_Depadded_Center_Frequency \
#             = Array_With_Pad_Center_Frequency.copy()
#         Int_Threshold_Index = int(Array_Surrogate_Ratio_Sorted.size * 0.95)
#         Value_Threshold = Array_Surrogate_Ratio_Sorted[Int_Threshold_Index]
#         Value_Mean = Array_Surrogate_Ratio_Sorted.mean()
#         # Get the final ratio    
#         Value_Raw_Nonstationary_Index_Signal \
#             = Array2D_With_Pad_Depadded_Normalized_Ratio[:,:]\
#                 .sum().sum()
#         Value_Raw_Nonstationary_Index_Impulse \
#             = Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio[:,:]\
#                 .sum().sum()
#         Value_Nonstationary_Index \
#             = Value_Raw_Nonstationary_Index_Signal \
#                 / Value_Raw_Nonstationary_Index_Impulse
#         Object_Singal \
#             = Signal_Info\
#                 (Array2D_Signal_Surrogate, \
#                 Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude, \
#                 Array2D_With_Pad_Depadded_Signal_Decomposed, \
#                 Array2D_Surrogate_Decomposed_Amplitude_Mean)
#         Object_Signal_Impulse \
#             = Signal_Info(Array2D_Signal_Surrogate, \
#                 Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude, \
#                 Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed, \
#                 Array2D_Surrogate_Decomposed_Amplitude_Mean)
#         return Value_Nonstationary_Index, \
#                 Value_Raw_Nonstationary_Index_Signal, \
#                 Value_Raw_Nonstationary_Index_Impulse, \
#                 Array2D_With_Pad_Depadded_Ratio, \
#                 Array2D_With_Pad_Depadded_Signal_Decomposed, \
#                 Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude, \
#                 Array2D_With_Pad_Depadded_Normalized_Ratio, \
#                 Array2D_With_Pad_Depadded_Impulse_Ratio, \
#                 Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed, \
#                 Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude, \
#                 Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio, \
#                 Array_With_Pad_Depadded_Center_Frequency, \
#                 Value_Threshold, \
#                 Value_Mean, \
#                 Object_Singal, \
#                 Object_Signal_Impulse, \
#                 Array_Surrogate_Ratio_Sorted, \
#                 Array3D_Signal_Surrogate_Decomposed_Amplitude, \
#                 Array3D_Normalized_Surrogate_Ratio

#     def Function_Normalized_Hilbert_Transform(self, Array_Signal):
#         """
#         Output:
#             - Array_Normalized_Hilbert_Transform
#             - Array_Normalize_Envelop
#         """
#         Array_Signal_ABS = numpy.abs(Array_Signal)
#         Array_Index_Local_Maxima_ABS, \
#             = signal.argrelextrema(Array_Signal_ABS, numpy.greater)
#         Function_interpolate_1d \
#             = interpolate.CubicSpline\
#                             (Array_Index_Local_Maxima_ABS, \
#                                 Array_Signal_ABS[Array_Index_Local_Maxima_ABS])
#         Array_Fit_Envelope \
#             = Function_interpolate_1d(numpy.arange(0, Array_Signal.size, 1))
#         Array_Normalized_Signal \
#             = Array_Signal / Array_Fit_Envelope
#         Parameter_Allowed_Error = 0.000001   
#         # This error threshold was add to avoid numerical issues, such as a 
#         # very samll value larger than one will be ignored in the cubic spline 
#         # interpolation
#         while numpy.abs(Array_Normalized_Signal\
#                             [1:Array_Normalized_Signal.size].max()) \
#                         > 1 + Parameter_Allowed_Error:
#             Array_Signal_ABS = numpy.abs(Array_Normalized_Signal)
#             Array_Index_Local_Maxima_ABS, \
#                 = signal.argrelextrema(Array_Signal_ABS, numpy.greater_equal)
#             Function_interpolate_1d \
#                 = interpolate.CubicSpline\
#                                 (Array_Index_Local_Maxima_ABS, \
#                                 Array_Signal_ABS[Array_Index_Local_Maxima_ABS])
#             Array_Fit_Envelope \
#                 = Function_interpolate_1d(numpy.arange(0, Array_Signal.size, 1))
#             Array_Normalized_Signal \
#                 = Array_Normalized_Signal / Array_Fit_Envelope
#         Array_Normalize_Envelop \
#             = Array_Signal / Array_Normalized_Signal
#         Array_Normalized_Hilbert_Transform \
#             = signal.hilbert(Array_Normalized_Signal)
#         return Array_Normalized_Hilbert_Transform, Array_Normalize_Envelop

#     def Function_Moving_Average(self, Array_Signal, Int_Averaging_Interval):
#         ''' 
#         Since the moving average should have an averaging interval, however, 
#         the averaged number of data points varies
#         with the change of the sampling interval, 
#         so the averaged number of data points is directly used in this 
#         function 
#         '''
#         Array_Signal_Moving_Average = numpy.zeros(Array_Signal.size)
#         Int_Left_Width = Int_Averaging_Interval // 2
#         Int_Right_Width = Int_Averaging_Interval - Int_Left_Width

#         for i_Time in range(Array_Signal.size):
#             if i_Time < Int_Left_Width:
#                 Array_Signal_Moving_Average[i_Time] \
#                     = numpy.mean(Array_Signal[:i_Time + Int_Right_Width])            
#             elif i_Time > Array_Signal.size - Int_Right_Width:
#                 Array_Signal_Moving_Average[i_Time] \
#                     = numpy.mean(Array_Signal[i_Time - Int_Left_Width:])
#             else:
#                 Array_Signal_Moving_Average[i_Time] \
#                     = numpy.mean(Array_Signal[i_Time - Int_Left_Width \
#                                                 : i_Time + Int_Right_Width])
#         return Array_Signal_Moving_Average

#     def Function_Remove_Trend(self, \
#             Array_Signal, Str_Wavelet_Name, Level = 0):
#         '''
#         Remove trend via discrete wavelet
#         ------------------------------------------------------------------------
#         Input:
#             Array_Signal
#             STr_Wavelet_Name
#             Level = 0
#         ------------------------------------------------------------------------
#         Output:
#             Array_Trend, Array_Fluctuation
#         '''
#         if Level == 0:
#             # print(Array_Signal.size)
#             # print(Str_Wavelet_Name)
#             # print(pywt.Wavelet(Str_Wavelet_Name))
#             # print(pywt.dwt_max_level(Array_Signal.size, \
#             #        pywt.Wavelet(Str_Wavelet_Name)))
#             Level = pywt.dwt_max_level(Array_Signal.size, \
#                                         pywt.Wavelet(Str_Wavelet_Name))
#         Tuple_DWT_Coefficients \
#             = pywt.wavedec(Array_Signal, Str_Wavelet_Name,'symmetric', Level)
#         Tuple_DWT_Coefficients[0][:] = 0
#         Array_Signal_Fluctuation \
#             = pywt.waverec(Tuple_DWT_Coefficients, \
#                             Str_Wavelet_Name, \
#                             'symmetric')[:Array_Signal.size]
#         Array_Signal_Trend = Array_Signal - Array_Signal_Fluctuation
#         return Array_Signal_Trend, Array_Signal_Fluctuation

#     def Function_Remove_Amplitude_Modulation(self, \
#             Array_Signal, Str_Wavelet_Name, Level = 0):
#         '''
#         Remove amplitude modulation
#         ------------------------------------------------------------------------
#         Input:
#             Array_Signal
#             STr_Wavelet_Name
#             Level = 0
#         ------------------------------------------------------------------------
#         Output:
#             Array_Signal_Trend
#             Array_Fit_Envelope
#             Array_Normalized_Fluctuation
#         '''
#         if Level == 0:
#             Level = pywt.dwt_max_level(Array_Signal.size, \
#                                         pywt.Wavelet(Str_Wavelet_Name))
#         Tuple_DWT_Coefficients \
#             = pywt.wavedec(Array_Signal, Str_Wavelet_Name,'symmetric', Level)
#         Tuple_DWT_Coefficients[0][:] = 0
#         Array_Signal_Fluctuation \
#             = pywt.waverec(Tuple_DWT_Coefficients, Str_Wavelet_Name, \
#                             'symmetric')[:Array_Signal.size]
#         Array_Signal_Trend = Array_Signal - Array_Signal_Fluctuation
#         Array_Signal_Fluctuation_ABS = numpy.abs(Array_Signal_Fluctuation)
#         Array_Index_Local_Maxima_ABS, \
#             = signal.argrelextrema(Array_Signal_Fluctuation_ABS, numpy.greater)
#         Array_Moving_Average_Local_Maxima \
#             = self.Function_Moving_Average\
#                     (Array_Signal_Fluctuation_ABS\
#                         [Array_Index_Local_Maxima_ABS], \
#                     2**Level)
#         Function_interpolate_1d \
#             = interpolate.CubicSpline(Array_Index_Local_Maxima_ABS, \
#                                         Array_Moving_Average_Local_Maxima)
#         Array_Fit_Envelope \
#             = Function_interpolate_1d(numpy.arange(0, Array_Signal.size, 1))
#         Array_Normalized_Fluctuation \
#             = Array_Signal_Fluctuation / Array_Fit_Envelope
#         return Array_Signal_Trend, Array_Fit_Envelope, \
#                 Array_Normalized_Fluctuation

#     def Function_Make_Unit_Length(self, \
#             Array_Time, Array_Signal, Int_Unit_Length):
#         """
#         Description: Pad arbitray length signal with zero to unit lengths
#         ------------------------------------------------------------------------
#         Input: 
#             Array_Time, Array_Signal, Int_Unitlength
#         ------------------------------------------------------------------------
#         Output: 
#             Array_Time_Make_Int_Length, Array_Signal_Make_Int_Length
#         """
#         Value_Delta_T = Array_Time[1] - Array_Time[0]
#         Array_Signal_Make_Int_Length \
#             = numpy.append\
#                     (Array_Signal, \
#                         numpy.zeros\
#                                 ((int(Array_Signal.size / Int_Unit_Length) \
#                                         + 1) \
#                                     * Int_Unit_Length \
#                                 - Array_Signal.size))
#         Array_Time_Make_Int_Length \
#             = numpy.linspace\
#                         (0, \
#                         Value_Delta_T * Array_Signal_Make_Int_Length.size, \
#                         Array_Signal_Make_Int_Length.size)
#         return Array_Time_Make_Int_Length, Array_Signal_Make_Int_Length

#     def Function_Zero_Padding_Both_Side(self, \
#             Array_Time, Array_Signal, Int_Left_Padding, Int_Right_Padding):
#         """
#         Description: Zero-padding signal with specific lengths
#         ------------------------------------------------------------------------
#         Input: 
#             Array_Time, Array_Signal, Int_Left_Padding, Int_Right_Padding
#         ------------------------------------------------------------------------
#         Output: 
#             Array_Time_Both_Expanded 
#             Array_Signal_Both_Expanded
#         """
#         Value_Delta_T = Array_Time[1] - Array_Time[0]
#         Array_Signal_Right_Expanded \
#             = numpy.append(Array_Signal, \
#                             numpy.zeros(Int_Right_Padding), axis = 0)
#         Array_Signal_Both_Expanded \
#             = numpy.append(numpy.zeros(Int_Left_Padding), \
#                             Array_Signal_Right_Expanded, axis = 0)
#         Array_Time_Both_Expanded \
#             = numpy.linspace\
#                         (- Value_Delta_T * Int_Left_Padding, \
#                         Value_Delta_T * Array_Signal_Right_Expanded.size, \
#                         Array_Signal_Both_Expanded.size)
#         return Array_Time_Both_Expanded, Array_Signal_Both_Expanded


class Class_Structural_Response():
    '''
    Class defined for structural response calculation
    '''

    def __init__(self):
        self.Class_Name = 'Class of structural response calculation'

    def Function_SDOF_Frequency_Response(self, \
            Array_Frequency, Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        """
        Calculate the SDOF frequency response function
        ------------------------------------------------------------------------
        Output:
            Array_Frequency_Response
        """
        Value_omega_n = numpy.sqrt(Value_Stiffness / Value_Mass)
        Value_omega_D = Value_omega_n * numpy.sqrt(1 - Value_Damping_Ratio**2)
        Array_Frequency_omega = Array_Frequency * 2 * numpy.pi
        Array_Frequency_Response_Complex \
            = 1 / Value_Mass \
                / (- Array_Frequency_omega**2 \
                    + 2 * numpy.complex(1j) * Value_Damping_Ratio \
                        * Value_omega_D * Array_Frequency_omega \
                    + Value_omega_D**2 )
        Array_Frequency_Response = numpy.abs(Array_Frequency_Response_Complex)
        return Array_Frequency_Response
    
    def Funciton_SDOF_Impulse_Response(self, \
            Array_Time, Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        """
        Calculate the SDOF impulse response function
        ------------------------------------------------------------------------
        Output:
            Array_Impulse_Response
        """
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Value_omega_n = numpy.sqrt(Value_Stiffness / Value_Mass)
        Value_omega_D = Value_omega_n * numpy.sqrt(1 - Value_Damping_Ratio**2)
        Temp_Array_Time_Reset_To_Zero = Array_Time - Array_Time.min()
        Array_Impulse_Response \
            = 1 / Value_Mass \
                / Value_omega_D \
                * numpy.exp(- Value_Damping_Ratio * Value_omega_n \
                                * Temp_Array_Time_Reset_To_Zero) \
                * numpy.sin(Temp_Array_Time_Reset_To_Zero * Value_omega_D)
        return Array_Impulse_Response

    def Function_SDOF_Response(self, \
            Array_Time, Array_Force, \
            Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        """
        Calculate the SDOF response 
        ------------------------------------------------------------------------
        Output:
            Array_SDOF_Response
        """
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Array_Impulse_Response \
            = self.Funciton_SDOF_Impulse_Response\
                    (Array_Time, Value_Damping_Ratio, Value_Mass, \
                        Value_Stiffness)
        Temp_Array_Response \
            = numpy.convolve(Array_Force, Array_Impulse_Response) \
                * Value_Delta_T
        Array_SDOF_Response = Temp_Array_Response[:Array_Time.size]
        return Array_SDOF_Response

    def Function_SDOF_Response_Runge_Kutta(self, \
            Array_Time, Array_Force, Parameter_m, Parameter_ksee, Parameter_k):
        """
        Input:
            Parameter_m:    Mass
            Parameter_ksee: Damping ratio
            Parameter_k:    Stiffness
        ------------------------------------------------------------------------
        Return:
            Array_SDOF_Response
            Array_SDOF_Response_1st_Derivative
        """
        Parameter_c = Parameter_ksee * 2 * numpy.sqrt(Parameter_k * Parameter_m)
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Array_X = numpy.zeros(Array_Time.size)
        Array_Y = numpy.zeros(Array_Time.size)
        Array_F = numpy.zeros(Array_Time.size)
        for i_Time in numpy.arange(0, Array_Time.size - 1):
            Temp_Array_T = numpy.zeros(4)
            Temp_Array_X = numpy.zeros(4)
            Temp_Array_Y = numpy.zeros(4)
            Temp_Array_F = numpy.zeros(4)
            # 1st calculation
            Temp_Array_T[0] = Array_Time[i_Time]
            Temp_Array_X[0] = Array_X[i_Time]
            Temp_Array_Y[0] = Array_Y[i_Time]
            Temp_Array_F[0] \
                = 1 / Parameter_m \
                    * (Array_Force[i_Time] \
                        - Parameter_k * Temp_Array_X[0] \
                        - Parameter_c * Temp_Array_Y[0])
            #  2nd calculation
            Temp_Array_T[1] = Array_Time[i_Time] + Value_Delta_T / 2
            Temp_Array_X[1] \
                = Array_X[i_Time] + Temp_Array_Y[0] * Value_Delta_T / 2
            Temp_Array_Y[1] \
                = Array_Y[i_Time] + Temp_Array_F[0] * Value_Delta_T / 2
            Temp_Array_F[1] \
                = 1 / Parameter_m \
                    * ((Array_Force[i_Time] + Array_Force[i_Time + 1]) / 2 \
                        - Parameter_k * Temp_Array_X[1] \
                        - Parameter_c * Temp_Array_Y[1])
            # 3rd calculation
            Temp_Array_T[2] = Array_Time[i_Time] + Value_Delta_T / 2
            Temp_Array_X[2] \
                = Array_X[i_Time] + Temp_Array_Y[1] * Value_Delta_T / 2
            Temp_Array_Y[2] \
                = Array_Y[i_Time] + Temp_Array_F[1] * Value_Delta_T / 2
            Temp_Array_F[2] \
                = 1 / Parameter_m \
                    * ((Array_Force[i_Time] + Array_Force[i_Time + 1]) / 2 \
                        - Parameter_k * Temp_Array_X[2] \
                        - Parameter_c * Temp_Array_Y[2])
            # 4th calculation
            Temp_Array_T[3] = Array_Time[i_Time] + Value_Delta_T
            Temp_Array_X[3] = Array_X[i_Time] + Temp_Array_Y[2] * Value_Delta_T
            Temp_Array_Y[3] = Array_Y[i_Time] + Temp_Array_F[2] * Value_Delta_T
            Temp_Array_F[3] \
                = 1 / Parameter_m \
                    * (Array_Force[i_Time + 1] \
                        - Parameter_k * Temp_Array_X[3] \
                        - Parameter_c * Temp_Array_Y[3])                                                                                         

            Array_X[i_Time + 1] \
                = Array_X[i_Time] \
                    + Value_Delta_T / 6 \
                        * (Temp_Array_Y[0] + 2 * Temp_Array_Y[1] + \
                            2 * Temp_Array_Y[2] + Temp_Array_Y[3])
            Array_Y[i_Time + 1] \
                = Array_Y[i_Time] \
                    + Value_Delta_T / 6 \
                        * (Temp_Array_F[0] + 2 * Temp_Array_F[1] + \
                            2 * Temp_Array_F[2] + Temp_Array_F[3])

        Array_SDOF_Response = Array_X.copy()
        Array_SDOF_Response_1st_Derivative = Array_Y.copy()
        return Array_SDOF_Response, Array_SDOF_Response_1st_Derivative

    def Function_MDOF_Response_Runge_Kutta(self, \
            Array_Time, Array2D_Force, Array2D_M, Array2D_C, Array2D_K):
        """
        Input:
            Array2D_M: Mass matrix
            Array2D_C: Damping coefficient matrix
            Array2D_K: Stiffness matrix
        ------------------------------------------------------------------------
        Return:
            Array2D_MDOF_Response
            Array2D_MDOF_Response_1st_Derivative 
        """
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Int_Number_of_Node = Array2D_Force.shape[1]
        Array2D_Z = numpy.zeros([Array_Time.size, Int_Number_of_Node * 2])
        Array2D_M_inv = numpy.linalg.inv(Array2D_M)
        for i_Time in numpy.arange(0, Array_Time.size - 1):
            Temp_Array2D_h = numpy.zeros([Int_Number_of_Node * 2,4])
            # 1st calculation
            Temp_Array_Force = Array2D_Force[i_Time,:]
            Temp_Array_z = Array2D_Z[i_Time, :]
            Temp_Array2D_h[:Int_Number_of_Node,0] \
                = Temp_Array_z[Int_Number_of_Node:]
            Temp_Array2D_h[Int_Number_of_Node:,0] \
                = Array2D_M_inv\
                    .dot(Temp_Array_Force \
                            - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node])\
                            - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node:]))
            # 2nd calculation
            Temp_Array_Force \
                = (Array2D_Force[i_Time,:] + Array2D_Force[i_Time + 1,:]) / 2
            Temp_Array_z \
                = Array2D_Z[i_Time, :] \
                + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,0]
            Temp_Array2D_h[:Int_Number_of_Node,1] \
                = Temp_Array_z[Int_Number_of_Node:]
            Temp_Array2D_h[Int_Number_of_Node:,1] \
                = Array2D_M_inv\
                    .dot(Temp_Array_Force \
                            - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node])\
                            - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node:]))
            # 3rd calculation
            Temp_Array_Force \
                = (Array2D_Force[i_Time,:] + Array2D_Force[i_Time + 1,:]) / 2
            Temp_Array_z \
                = Array2D_Z[i_Time, :] \
                + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,1]
            Temp_Array2D_h[:Int_Number_of_Node,2] \
                = Temp_Array_z[Int_Number_of_Node:]
            Temp_Array2D_h[Int_Number_of_Node:,2] \
                = Array2D_M_inv\
                    .dot(Temp_Array_Force \
                            - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node])\
                            - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node:]))
            # 4th calculation
            Temp_Array_Force = Array2D_Force[i_Time + 1,:]
            Temp_Array_z \
                = Array2D_Z[i_Time, :] + Value_Delta_T * Temp_Array2D_h[:,2]
            Temp_Array2D_h[:Int_Number_of_Node,3] \
                = Temp_Array_z[Int_Number_of_Node:]
            Temp_Array2D_h[Int_Number_of_Node:,3] \
                = Array2D_M_inv\
                    .dot(Temp_Array_Force \
                            - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node])\
                            - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node:]))                                                                                     

            Array2D_Z[i_Time + 1,:] \
                = Array2D_Z[i_Time,:] \
                + Value_Delta_T / 6 \
                    * (Temp_Array2D_h[:,0] + 2 * Temp_Array2D_h[:,1] \
                        + 2 * Temp_Array2D_h[:,2] + Temp_Array2D_h[:,3])

        Array2D_MDOF_Response = Array2D_Z[:,:Int_Number_of_Node]
        Array2D_MDOF_Response_1st_Derivative = Array2D_Z[:,Int_Number_of_Node:]
        return Array2D_MDOF_Response, Array2D_MDOF_Response_1st_Derivative      

    def Function_MDOF_Next_Step_Response_Runge_Kutta(self, Value_Delta_T, \
            Array2D_Force_Two_Step, Array2D_M, Array2D_C, Array2D_K,\
            Array_Response_Current_Step, \
            Array_Response_1st_Derivative_Current_Step):
        """
        ------------------------------------------------------------------------
        Input:
            Value_Delta_T: Time interval
            Array_Force: Force array at current time instant
            Array2D_M: Mass matrix
            Array2D_C: Damping coefficient matrix
            Array2D_K: Stiffness matrix
        ------------------------------------------------------------------------
        Output:
            Array_Response_
        """
        Int_Number_of_Node_Free = Array_Response_Current_Step.size
        Temp_Array_Z_Initial \
            = numpy.append(Array_Response_Current_Step, \
                            Array_Response_1st_Derivative_Current_Step)
        Array2D_M_inv = numpy.linalg.inv(Array2D_M)

        Temp_Array2D_h = numpy.zeros([Int_Number_of_Node_Free * 2,4])
        # 1st calculation
        Temp_Array_Force = Array2D_Force_Two_Step[0,:]
        Temp_Array_z = Temp_Array_Z_Initial.copy()
        Temp_Array2D_h[:Int_Number_of_Node_Free,0] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_h[Int_Number_of_Node_Free:,0] \
            = Array2D_M_inv\
                .dot(Temp_Array_Force \
                        - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node_Free])\
                        - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node_Free:]))
        # 2nd calculation
        Temp_Array_Force \
            = (Array2D_Force_Two_Step[0,:] + Array2D_Force_Two_Step[1,:]) / 2
        Temp_Array_z \
            = Temp_Array_Z_Initial + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,0]
        Temp_Array2D_h[:Int_Number_of_Node_Free,1] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_h[Int_Number_of_Node_Free:,1] \
            = Array2D_M_inv\
                .dot(Temp_Array_Force \
                        - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node_Free])\
                        - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node_Free:]))
        # 3rd calculation
        Temp_Array_Force \
            = (Array2D_Force_Two_Step[0,:] + Array2D_Force_Two_Step[1,:]) / 2
        Temp_Array_z \
            = Temp_Array_Z_Initial + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,1]
        Temp_Array2D_h[:Int_Number_of_Node_Free,2] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_h[Int_Number_of_Node_Free:,2] \
            = Array2D_M_inv\
                .dot(Temp_Array_Force \
                        - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node_Free])\
                        - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node_Free:]))
        # 4th calculation
        Temp_Array_Force = Array2D_Force_Two_Step[1,:]
        Temp_Array_z \
            = Temp_Array_Z_Initial + Value_Delta_T * Temp_Array2D_h[:,2]
        Temp_Array2D_h[:Int_Number_of_Node_Free,3] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_h[Int_Number_of_Node_Free:,3] \
            = Array2D_M_inv\
                .dot(Temp_Array_Force \
                        - Array2D_K.dot(Temp_Array_z[:Int_Number_of_Node_Free])\
                        - Array2D_C.dot(Temp_Array_z[Int_Number_of_Node_Free:]))                                                                                     

        Temp_Array_Z_End \
            = Temp_Array_Z_Initial\
            + Value_Delta_T / 6 \
                * (Temp_Array2D_h[:,0] + 2 * Temp_Array2D_h[:,1] \
                    + 2 * Temp_Array2D_h[:,2] + Temp_Array2D_h[:,3])

        Array_Response_Next_Step \
            = Temp_Array_Z_End[:Int_Number_of_Node_Free]
        Array_Response_1st_Derivative_Next_Step \
            = Temp_Array_Z_End[Int_Number_of_Node_Free:]
        return Array_Response_Next_Step, Array_Response_1st_Derivative_Next_Step    

    def Function_Stiffness_Matrix_Tall_Building(self,\
            Array_State_Free, Array_State_Fixed, Array2D_K_Element_Basic):
        Temp_Array2D_K_Element = Array2D_K_Element_Basic
        Int_Number_of_Node_Free = int(Array_State_Free.size / 2)
        Int_Number_of_Node_Total \
            = int((Array_State_Free.size + Array_State_Fixed.size) / 2)
        Array2D_K_Full \
            = numpy.zeros([Int_Number_of_Node_Total,Int_Number_of_Node_Total])
        for i_Element in range(Int_Number_of_Node_Total - 1):
            Array2D_K_Full[i_Element:i_Element + 2, i_Element:i_Element+2] \
                += Temp_Array2D_K_Element * numpy.sqrt((i_Element + 1))
        return Array2D_K_Full

    def Function_Stiffness_Matrix_Tall_Building_With_BRB(self,\
            Array_State_Free, Array_State_Fixed, Array2D_K_Element_Basic, \
            Value_Threshold_Buckling):
        Temp_Array2D_K_Element \
            = Array2D_K_Element_Basic
        Value_Threshold = Value_Threshold_Buckling
        Value_Remain_Stiffness_Ratio \
            = 0.5 # Residual stiffness after buckling
        Int_Number_of_Node_Free \
            = int(Array_State_Free.size / 2)
        Int_Number_of_Node_Total \
            = int((Array_State_Free.size + Array_State_Fixed.size) / 2)
        Array2D_K_Full \
            = numpy.zeros([Int_Number_of_Node_Total,Int_Number_of_Node_Total])
        Temp_Array3D_K_Element \
            = numpy.zeros([2,2, Int_Number_of_Node_Total - 1])
        # for i_Element in range(Int_Number_of_Node_Total - 1):
        #     Temp_Array_Displacement_Element \
        #         = Array_State_Free[i_Element: i_Element + 2]
        #     Temp_Value_Relative_Displacement \
        #         = numpy.abs(Temp_Array_Displacement_Element[0] \
        #                       - Temp_Array_Displacement_Element[1])
        #     if Temp_Value_Relative_Displacement < Value_Threshold:    
        #         Temp_Array3D_K_Element[:,:,i_Element] \
        #             = Temp_Array2D_K_Element
        #     else:
        #         Temp_Ratio_XT_X \
        #             = Value_Threshold / Temp_Value_Relative_Displacement
        #         Temp_Array3D_K_Element[:,:,i_Element] \
        #             = Temp_Array2D_K_Element \
        #                   * ( Temp_Ratio_XT_X + Value_Remain_Stiffness_Ratio \
        #                                           * (1 - Temp_Ratio_XT_X))
        Temp_Array_Drift = Array_State_Free[:-1] - Array_State_Free[1:]
        for i_Element in range(Int_Number_of_Node_Total - 1):
            Temp_Value_Relative_Displacement \
                = numpy.abs(Temp_Array_Drift[i_Element]) 
            Temp_Array3D_K_Element[:,:,i_Element] \
                = Temp_Array2D_K_Element
            if Temp_Value_Relative_Displacement \
                    > Value_Threshold and i_Element == 0:
                Temp_Ratio_XT_X \
                    = Value_Threshold / Temp_Value_Relative_Displacement
                Temp_Array3D_K_Element[:,:,i_Element] \
                    = Temp_Array2D_K_Element \
                        * ( Temp_Ratio_XT_X \
                            + Value_Remain_Stiffness_Ratio \
                                * (1 - Temp_Ratio_XT_X))
        for i_Element in range(Int_Number_of_Node_Total - 1):
            Array2D_K_Full[i_Element:i_Element + 2, i_Element:i_Element+2] \
                += Temp_Array3D_K_Element[:,:,i_Element] \
                    * numpy.sqrt((i_Element + 1))
        return Array2D_K_Full

    def Function_Element_Restoring_Force_LR_Bearing(self,\
            Array2D_Element_State_Current, \
            Array2D_Element_Restoring_Force_Current, \
            Array_K, Value_Threshold):
        """
        LR bearing: Led rubber bearing
        ------------------------------------------------------------------------
        Input:
            Array2D_State: History of the displacement and velocity of the nodes
            Array2D_Element_Restoring_Force_Current: History of the restoring force
            Array_k: Array of the two sitffness
        """
        Temp_Array_Relative_Disp \
            = Array2D_Element_State_Current[:,0] \
                - Array2D_Element_State_Current[:,1]
        Temp_Array_Relative_Velo \
            = Array2D_Element_State_Current[:,2] \
                - Array2D_Element_State_Current[:,3]
        Temp_Array_Force = Array2D_Element_Restoring_Force_Current[:,0]
        Temp_Array_Index_Local_Max \
            = scipy.signal.argrelextrema\
                            (Temp_Array_Relative_Disp, numpy.greater_equal)[0]
        Temp_Array_Index_Local_Min \
            = scipy.signal.argrelextrema\
                            (Temp_Array_Relative_Disp, numpy.less_equal)[0]
        Temp_Int_Index_Last_Local_Max = Temp_Array_Index_Local_Max[-1]
        Temp_Int_Index_Last_Local_Min = Temp_Array_Index_Local_Min[-1]
        Temp_Bool_Conditon_1 = Temp_Array_Index_Local_Max.size == 1
        Temp_Bool_Conditon_2 = Temp_Array_Index_Local_Min.size == 1
        if  Temp_Bool_Conditon_1 and Temp_Bool_Conditon_2:
            if numpy.abs(Temp_Array_Relative_Disp[-1]) < Value_Threshold:
                Temp_Value_Current_Force \
                    = Temp_Array_Relative_Disp[-1] * Array_K[0]
            else:
                Temp_Value_Current_Elastic_Deformation \
                    = Temp_Array_Relative_Disp[-1] - Value_Threshold
                Temp_Value_Current_Force \
                    = Value_Threshold * Array_K[0] \
                    + numpy.sign(Temp_Array_Relative_Disp[-1]) \
                        * (numpy.abs(Temp_Value_Current_Elastic_Deformation)) \
                        * Array_K[1]
        else:
            Temp_Int_Last_Extreme \
                = numpy.min([Temp_Int_Index_Last_Local_Max, \
                                Temp_Int_Index_Last_Local_Min])
            Temp_Value_Last_Extreme_Disp \
                = Temp_Array_Relative_Disp[Temp_Int_Last_Extreme]
            Temp_Value_Last_Extreme_Force \
                = Temp_Array_Force[Temp_Int_Last_Extreme]

            Temp_Value_Current_Disp = Temp_Array_Relative_Disp[-1]
            Temp_Value_Compare_1 \
                = numpy.abs(Temp_Value_Current_Disp \
                                - Temp_Value_Last_Extreme_Disp)
            Temp_Value_Compare_2 \
                = 2 * Value_Threshold
            if Temp_Value_Compare_1 <= Temp_Value_Compare_2:
                Temp_Value_Current_Force \
                    = Temp_Value_Last_Extreme_Force \
                    + (Temp_Value_Current_Disp - Temp_Value_Last_Extreme_Disp) \
                        * Array_K[0]
            else:
                Temp_Value_Absolute_Disp_Diff \
                    = numpy.abs(Temp_Value_Current_Disp \
                                    - Temp_Value_Last_Extreme_Disp)
                Temp_Value_Sign \
                    = numpy.sign(Temp_Value_Current_Disp \
                                    - Temp_Value_Last_Extreme_Disp)
                Temp_Value_Current_Force \
                    = Temp_Value_Last_Extreme_Force \
                    + Temp_Value_Sign * 2 * Value_Threshold * Array_K[0]\
                    + Temp_Value_Sign \
                        * (Temp_Value_Absolute_Disp_Diff \
                                - 2 * Value_Threshold) \
                        * Array_K[1]
        Array2D_Element_Restoring_Force_Next_Step \
            = numpy.append\
                        (Array2D_Element_Restoring_Force_Current, \
                        numpy.array([[Temp_Value_Current_Force, \
                                        -Temp_Value_Current_Force]]), \
                        axis = 0)
        return Array2D_Element_Restoring_Force_Next_Step

    def Function_Stiffness_Matrix_Tall_Building_Conventional_Part(self,\
            Array_Current_State_Upper_Part, Array3D_K_Element_Basic):
        Int_Number_of_Node_Total = int(Array_Current_State_Upper_Part.size / 2)
        Array2D_K_Full \
            = numpy.zeros([Int_Number_of_Node_Total,Int_Number_of_Node_Total])
        for i_Element in range(Array3D_K_Element_Basic.shape[2]):
            Array2D_K_Full[i_Element:i_Element + 2, i_Element:i_Element+2] \
                += Array3D_K_Element_Basic[:,:,i_Element]
        return Array2D_K_Full

    def Function_Total_Restoring_Force_To_Element_Restoring_Force(self,\
            Array2D_Restoring_Force):
        Array3D_Element_Restoring_Force \
            = numpy.zeros([Array2D_Restoring_Force.shape[0], \
                            2, \
                            Array2D_Restoring_Force.shape[1] - 1])
        Array3D_Element_Restoring_Force[:,0,i_Element] \
            = Array2D_Restoring_Force[:,0]
        Array3D_Element_Restoring_Force[:,1,i_Element] \
            = - Array3D_Element_Restoring_Force[:,0,i_Element]
        Temp_Index_Ended = Array3D_Element_Restoring_Force.shape[2]
        for i_Element in numpy.arange(1, Temp_Index_Ended):
            Array3D_Element_Restoring_Force[:,0,i_Element] \
                = Array2D_Restoring_Force[:,i_Element] -\
                     Array3D_Element_Restoring_Force[:,1,i_Element - 1]
            Array3D_Element_Restoring_Force[:,1,i_Element] \
                =  - Array3D_Element_Restoring_Force[:,0,i_Element]
        return Array3D_Element_Restoring_Force

    def Function_Damping_Force_Tall_Building_With_Base_Isolation(self,\
            Array2D_State, Array_Index_Free_Node, Array_Index_Fixed_Node, \
            Array3D_Element_Damping_Matrix):
        Int_Nomber_of_Node_Total \
            = Array_Index_Free_Node.size + Array_Index_Fixed_Node.size
        Temp_Array2D_Velocity = Array2D_State[:,Int_Nomber_of_Node_Total:]
        Array2D_Element_Damping_Force \
            = numpy.zeros([Array3D_Element_Damping_Matrix.shape[0], \
                            Array3D_Element_Damping_Matrix.shape[2]])
        for i_Element in range(Array3D_Element_Damping_Matrix.shape[2]):
            Array2D_Element_Damping_Force[:,i_Element] \
                = Array3D_Element_Damping_Matrix[:,:,i_Element]\
                    .dot(Temp_Array2D_Velocity[-1,i_Element: i_Element + 2])
        Array_Damping_Force = numpy.zeros(Array2D_State.shape[2])
        for i_Element in range(Array3D_Element_Damping_Matrix.shape[2]):
            Array_Damping_Force[i_Element: i_Element + 2] \
                += Array2D_Element_Damping_Force[:,i_Element]
        return Array_Damping_Force

    def Function_Restoring_Force_Tall_Building_With_Base_Isolation(self,\
            Array2D_Restoring_Force_Current, \
            Array2D_State, Array_Index_Free_Node, Array_Index_Fixed_Node, \
            Array3D_K_Element_Basic, Array_K, Value_Threshold):
        Int_Number_of_Node_Upper_Part = Array_Index_Free_Node.size
        Int_Number_of_Node_Total \
            = Array_Index_Free_Node.size + Array_Index_Fixed_Node.size
        Temp_Array_Index_Selection \
            = numpy.zeros(Array_Index_Free_Node.size * 2, dtype = numpy.int)
        Temp_Array_Index_Selection[:Array_Index_Free_Node.size] \
            = Array_Index_Free_Node
        Temp_Array_Index_Selection[Array_Index_Free_Node.size:] \
            = Array_Index_Free_Node + Int_Number_of_Node_Total
        Array_Current_State = Array2D_State[-1,:]
        Array2D_K_Full \
            = self.Function_Stiffness_Matrix_Tall_Building_Conventional_Part(\
                        Array_Current_State[Temp_Array_Index_Selection], \
                        Array3D_K_Element_Basic)
        Array_Restoring_Force_Conventional_Part \
            = Array2D_K_Full\
                .dot(Array_Current_State[:Int_Number_of_Node_Upper_Part])

        Array2D_Element_State_Current \
            = numpy.zeros([Array2D_State.shape[0], 4])
        Array2D_Element_State_Current[:,:2] \
            = Array2D_State\
                [:,Int_Number_of_Node_Total - 2 : Int_Number_of_Node_Total]
        Array2D_Element_State_Current[:,2:] \
            = Array2D_State\
                [:,Int_Number_of_Node_Total*2 - 2 : Int_Number_of_Node_Total*2]
        Array2D_Element_Restoring_Force_Current \
            = numpy.zeros([Array2D_Restoring_Force_Current.shape[0], 2])
        Array2D_Element_Restoring_Force_Current[:,1] \
            = Array2D_Restoring_Force_Current[:,-1]
        Array2D_Element_Restoring_Force_Current[:,0] \
            = - Array2D_Element_Restoring_Force_Current[:,1] 
        Array2D_Element_Restoring_Force_Next_Step \
            = self.Function_Element_Restoring_Force_LR_Bearing(\
                Array2D_Element_State_Current, \
                Array2D_Element_Restoring_Force_Current, \
                Array_K, Value_Threshold)
        Array_Restoring_Force_Next_Step \
            = numpy.append(Array_Restoring_Force_Conventional_Part, \
                            Array2D_Element_Restoring_Force_Next_Step[-1,1])
        Array_Restoring_Force_Next_Step[-1,-2] \
            += Array2D_Element_Restoring_Force_Next_Step[-1,0]
        Array2D_Restoring_Force_Next_Step \
            = numpy\
                .append(\
                    Array2D_Restoring_Force_Current, \
                    Array_Restoring_Force_Next_Step.reshape(1,-1), \
                    axis = 0)
        return Array2D_Restoring_Force_Next_Step

    def Function_MDOF_Next_Step_Response_Runge_Kutta_LR(self, \
            Value_Delta_T, \
            Array2D_External_Force_Current, \
            Array2D_M, \
            Array3D_Element_Damping_Matrix,\
            Array2D_State_Current, Array_Index_Free_Node, \
            Array_Index_Fixed_Node, \
            Array2D_Restoring_Force_Current, \
            Array3D_K_Element_Basic, \
            Array_K, \
            Value_Threshold):
        """
        ------------------------------------------------------------------------
        Input:
            Array2D_State_Current: 
                State histroy until current time
            Value_Delta_T: 
                Time interval
            Array2D_External_Force_Current:
                External force history until current time
            Array2D_M: 
                Mass matrix
            Array3D_Element_Damping_Matrix: 
                Damping coefficient matrix of each element
            Array2D_Restoring_Force_Current:
                Restoring force history until current time
        ------------------------------------------------------------------------
        Output:
            Array_Response_
        """
        Int_Number_of_Node_Free = Array_Index_Free_Node.size
        Int_Number_of_Node_Total \
            = Int_Number_of_Node_Free + Array_Index_Fixed_Node.size

        Temp_Array_Z_Initial = Array2D_State_Current[-1,:]
        Array2D_M_inv = numpy.linalg.inv(Array2D_M)

        Temp_Array2D_h = numpy.zeros([Int_Number_of_Node_Free * 2,4])
        # 1st calculation
        Temp_Array_Force = Array2D_Force_Two_Step[0,:]
        Temp_Array_z = Temp_Array_Z_Initial.copy()
        Temp_Array2D_h[:Int_Number_of_Node_Free,0] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_Restoring_Force_Current \
            = self.Function_Restoring_Force_Tall_Building_With_Base_Isolation(\
                    Array2D_Restoring_Force_Current, \
                    Array2D_State_Current, \
                    Array_Index_Free_Node, \
                    Array_Index_Fixed_Node, \
                    Array3D_K_Element_Basic, \
                    Array_K, \
                    Value_Threshold)
        Temp_Array_Damping_Force_Current \
            = self.Function_Damping_Force_Tall_Building_With_Base_Isolation(\
                    Array2D_State_Current, \
                    Array_Index_Free_Node, \
                    Array_Index_Fixed_Node, \
                    Array3D_Element_Damping_Matrix)
        Temp_Array2D_h[Int_Number_of_Node_Free:,0] \
            = Array2D_M_inv.dot(Temp_Array_Force \
                                - Temp_Array2D_Restoring_Force_Current[-1,:]\
                                - Temp_Array_Damping_Force_Current)
        # 2nd calculation
        Temp_Array_Force \
            = (Array2D_Force_Two_Step[0,:] + Array2D_Force_Two_Step[1,:]) / 2
        Temp_Array_z \
            = Temp_Array_Z_Initial + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,0]
        Temp_Array2D_h[:Int_Number_of_Node_Free,1] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_Restoring_Force_Current \
            = self.Function_Restoring_Force_Tall_Building_With_Base_Isolation(\
                    Temp_Array2D_Restoring_Force_Current, \
                    numpy.append(Array2D_State_Current, \
                                    Temp_Array_z.reshape(1,-1), \
                                    axie = 0), \
                    Array_Index_Free_Node, \
                    Array_Index_Fixed_Node, \
                    Array3D_K_Element_Basic, \
                    Array_K, \
                    Value_Threshold)
        Temp_Array_Damping_Force_Current \
            = self.Function_Damping_Force_Tall_Building_With_Base_Isolation(\
                    numpy.append(Array2D_State_Current, \
                                    Temp_Array_z.reshape(1,-1), \
                                    axie = 0), \
                    Array_Index_Free_Node, Array_Index_Fixed_Node, \
                    Array3D_Element_Damping_Matrix)
        Temp_Array2D_h[Int_Number_of_Node_Free:,0] \
            = Array2D_M_inv.dot(Temp_Array_Force \
                                - Temp_Array2D_Restoring_Force_Current[-1,:]\
                                - Temp_Array_Damping_Force_Current)
        # 3rd calculation
        Temp_Array_Force \
            = (Array2D_Force_Two_Step[0,:] + Array2D_Force_Two_Step[1,:]) / 2
        Temp_Array_z \
            = Temp_Array_Z_Initial + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,1]
        Temp_Array2D_h[:Int_Number_of_Node_Free,2] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_Restoring_Force_Current \
            = self.Function_Restoring_Force_Tall_Building_With_Base_Isolation(\
                    Temp_Array2D_Restoring_Force_Current, \
                    numpy.append(Array2D_State_Current, \
                                    Temp_Array_z.reshape(1,-1), axie = 0), \
                    Array_Index_Free_Node, Array_Index_Fixed_Node, \
                    Array3D_K_Element_Basic, Array_K, Value_Threshold)
        Temp_Array_Damping_Force_Current \
            = self.Function_Damping_Force_Tall_Building_With_Base_Isolation(\
                    numpy.append(Array2D_State_Current, \
                                    Temp_Array_z.reshape(1,-1), axie = 0), \
                    Array_Index_Free_Node, Array_Index_Fixed_Node, \
                    Array3D_Element_Damping_Matrix)
        Temp_Array2D_h[Int_Number_of_Node_Free:,0] \
            = Array2D_M_inv.dot(Temp_Array_Force \
                                - Temp_Array2D_Restoring_Force_Current[-1,:]\
                                - Temp_Array_Damping_Force_Current)
        # 4th calculation
        Temp_Array_Force = Array2D_Force_Two_Step[1,:]
        Temp_Array_z \
            = Temp_Array_Z_Initial + Value_Delta_T * Temp_Array2D_h[:,2]
        Temp_Array2D_h[:Int_Number_of_Node_Free,3] \
            = Temp_Array_z[Int_Number_of_Node_Free:]
        Temp_Array2D_Restoring_Force_Current \
            = self.Function_Restoring_Force_Tall_Building_With_Base_Isolation(\
                    Temp_Array2D_Restoring_Force_Current, \
                    numpy.append(Array2D_State_Current, \
                        Temp_Array_z.reshape(1,-1), axie = 0), \
                    Array_Index_Free_Node, Array_Index_Fixed_Node, \
                    Array3D_K_Element_Basic, Array_K, Value_Threshold)
        Temp_Array_Damping_Force_Current \
            = self.Function_Damping_Force_Tall_Building_With_Base_Isolation(\
                    numpy.append(Array2D_State_Current, \
                        Temp_Array_z.reshape(1,-1), axie = 0), \
                    Array_Index_Free_Node, Array_Index_Fixed_Node, \
                    Array3D_Element_Damping_Matrix)
        Temp_Array2D_h[Int_Number_of_Node_Free:,0] \
            = Array2D_M_inv.dot(Temp_Array_Force \
                                - Temp_Array2D_Restoring_Force_Current[-1,:]\
                                - Temp_Array_Damping_Force_Current)                                                                          
        Temp_Array_Z_End \
            = Temp_Array_Z_Initial\
            + Value_Delta_T / 6 \
                * (Temp_Array2D_h[:,0] \
                    + 2 * Temp_Array2D_h[:,1] \
                    + 2 * Temp_Array2D_h[:,2] \
                    + Temp_Array2D_h[:,3])

        Array_Response_Next_Step \
            = Temp_Array_Z_End[:Int_Number_of_Node_Free]
        Array_Response_1st_Derivative_Next_Step \
            = Temp_Array_Z_End[Int_Number_of_Node_Free:]
        return Array_Response_Next_Step, Array_Response_1st_Derivative_Next_Step    

    def Function_SDOF_Response_Unit_Initial_Velocity(self, \
            Array_Time, Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        Value_omega_n = numpy.sqrt(Value_Stiffness / Value_Mass)
        Value_omega_D = Value_omega_n * numpy.sqrt(1 - Value_Damping_Ratio**2)
        Temp_Array_Time_Reset_To_Zero = Array_Time - Array_Time.min()
        Array_Impulse_Response \
            = 1 / 1 / Value_omega_D \
                * numpy.exp(- Value_Damping_Ratio * Value_omega_n \
                * Temp_Array_Time_Reset_To_Zero) \
                * numpy.sin(Temp_Array_Time_Reset_To_Zero * Value_omega_D)
        return Array_Impulse_Response

    def Function_SDOF_Response_Unit_Initial_Displacement(self, \
            Array_Time, Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        Value_omega_n = numpy.sqrt(Value_Stiffness / Value_Mass)
        Value_omega_D = Value_omega_n * numpy.sqrt(1 - Value_Damping_Ratio**2)
        Temp_Array_Time_Reset_To_Zero = Array_Time - Array_Time.min()
        Array_Impulse_Response \
            = 1 / 1 * numpy.exp(- Value_Damping_Ratio * Value_omega_n \
                                * Temp_Array_Time_Reset_To_Zero) \
                    * numpy.cos(Temp_Array_Time_Reset_To_Zero * Value_omega_D)
        return Array_Impulse_Response

    def Funciton_SDOF_Impulse_Response_With_Velocity_Acceleration(self, \
            Array_Time, Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Value_omega_n = numpy.sqrt(Value_Stiffness / Value_Mass)
        Value_omega_D = Value_omega_n * numpy.sqrt(1 - Value_Damping_Ratio**2)
        Temp_Array_Time_Reset_To_Zero = Array_Time - Array_Time.min()
        Array2D_Impulse_Response = numpy.zeros([Array_Time.size, 3])
        Array2D_Impulse_Response[:,0] \
            = 1 / Value_Mass / Value_omega_D \
            * numpy.exp(- Value_Damping_Ratio \
                            * Value_omega_n \
                            * Temp_Array_Time_Reset_To_Zero) \
            * numpy.sin(Temp_Array_Time_Reset_To_Zero * Value_omega_D)
        Array2D_Impulse_Response[:,1] \
            = 1 / Value_Mass / Value_omega_D \
            * numpy.exp(- Value_Damping_Ratio \
                            * Value_omega_n \
                            * Temp_Array_Time_Reset_To_Zero) \
            * (- Value_Damping_Ratio * Value_omega_n \
                    * numpy.sin(Temp_Array_Time_Reset_To_Zero * Value_omega_D) \
                + Value_omega_D \
                    * numpy.cos(Temp_Array_Time_Reset_To_Zero * Value_omega_D))
        Array2D_Impulse_Response[:,2] \
            = 1 / Value_Mass / Value_omega_D \
            * numpy.exp(- Value_Damping_Ratio \
                            * Value_omega_n \
                            * Temp_Array_Time_Reset_To_Zero) \
            * ((Value_Damping_Ratio**2 * Value_omega_n**2 - Value_omega_D**2) \
                * numpy.sin(Temp_Array_Time_Reset_To_Zero * Value_omega_D)
                - 2 * Value_Damping_Ratio * Value_omega_n * Value_omega_D \
                    * numpy.cos(Temp_Array_Time_Reset_To_Zero * Value_omega_D))
        return Array2D_Impulse_Response

    def Function_SDOF_Response_With_Velocity_Acceleration(self, \
            Array_Time, Array_Force, \
            Value_Damping_Ratio, Value_Mass, Value_Stiffness):
        """
        Input:
            Array_Time
            Array_Force
            Value_Damping_Ratio
            Value_Maxx
            Value_Stiffness
        ------------------------------------------------------------------------
        Output:
            Array2D_SDOF_Response
        """
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Array2D_Impulse_Response \
            = self.Funciton_SDOF_Impulse_Response_With_Velocity_Acceleration(\
                    Array_Time, \
                    Value_Damping_Ratio, \
                    Value_Mass, \
                    Value_Stiffness)
        Array2D_SDOF_Response = numpy.zeros([Array_Time.size, 3])    
        for i_Derivative in range(3):
            Temp_Array_Response \
                = numpy.convolve(Array_Force, \
                                    Array2D_Impulse_Response[:,i_Derivative]) \
                    * Value_Delta_T
            Array2D_SDOF_Response[:,i_Derivative] \
                = Temp_Array_Response[:Array_Time.size]
        Array2D_SDOF_Response[:,2] \
            = Array2D_SDOF_Response[:,2] + Array_Force / Value_Mass
        return Array2D_SDOF_Response, Array_Time

class Class_Structural_Response_MDOF_Finite_Element():
    
    def __init__(self):
        self.Class_Name = 'MDOF_FEM_Method_CoCo'
        self.Int_Number_of_Node_Total = int(7)
        self.Int_Number_of_Node_Free = int(self.Int_Number_of_Node_Total - 1)
        self.Int_Number_of_Element = self.Int_Number_of_Node_Total - 1
        self.Value_Node_Mass_Basic = 1.92 * 10**6 # Base node mass
        self.Value_K =  1.274 * 10**8 # Base Stiffness
        # # Base Stiffness
        # self.Array_K = numpy.array([self.Value_K, self.Value_K / 10])
        # self.Value_Threshold = 0.2
        self.Value_C \
                = Parameter_Damping_Ratio_Building \
                    * 2 * numpy.sqrt(self.Value_K \
                                        * self.Value_Node_Mass_Basic \
                                        * self.Int_Number_of_Node_Free)
        self.Value_Height = 120 # Building height

    class Sub_Class_Element_Normal(object):
        '''
        Class defined for normal structure elements
        '''
        def __init__(self, Value_C, Value_K):
            self.Class_Name = 'Sub_Class_Element_Normal'
            Temp_Array2D_Element_Stiffness_Matrix = numpy.zeros([2,2])
            Temp_Array2D_Element_Stiffness_Matrix[0,0] = Value_K
            Temp_Array2D_Element_Stiffness_Matrix[0,1] = - Value_K
            Temp_Array2D_Element_Stiffness_Matrix[1,0] = - Value_K
            Temp_Array2D_Element_Stiffness_Matrix[1,1] = Value_K
            self.Array2D_Stiffness_Matrix \
                = Temp_Array2D_Element_Stiffness_Matrix
            Temp_Array2D_Element_Damping_Matrix = numpy.zeros([2,2])
            Temp_Array2D_Element_Damping_Matrix[0,0] = Value_C
            Temp_Array2D_Element_Damping_Matrix[0,1] = - Value_C
            Temp_Array2D_Element_Damping_Matrix[1,0] = - Value_C
            Temp_Array2D_Element_Damping_Matrix[1,1] = Value_C
            self.Array2D_Damping_Matrix = Temp_Array2D_Element_Damping_Matrix
            self.Array2D_Element_State = numpy.zeros(4).reshape(1,-1)
            self.Array2D_Element_Damping_Force = numpy.zeros(2).reshape(1,-1)
            self.Array2D_Element_Restoring_Force = numpy.zeros(2).reshape(1,-1)   

        def Function_Update_State(self, Array_State):
            self.Array2D_Element_State \
                = numpy.append(self.Array2D_Element_State, \
                                Array_State.reshape(1,-1), axis = 0)

        def Function_Estimate_Restoring_Force(self, \
                Temp_Array_State_Node_Current):
            Array_Restoring_Force \
                = self.Array2D_Stiffness_Matrix\
                        .dot(Temp_Array_State_Node_Current[:2])
            return Array_Restoring_Force
            
        def Function_Update_Restoring_Force(self, Array_Restoring_Force):
            self.Array2D_Element_Restoring_Force \
                = numpy.append(self.Array2D_Element_Restoring_Force, \
                                Array_Restoring_Force.reshape(1,-1), axis = 0)

        def Function_Estimate_Damping_Force(self, Temp_Array_State_Node_Current):
            Array_Damping_Force \
                = self.Array2D_Damping_Matrix\
                        .dot(Temp_Array_State_Node_Current[2:])
            return Array_Damping_Force

        def Function_Update_Damping_Force(self, Array_Damping_Force):
            self.Array2D_Element_Damping_Force \
                = numpy.append(self.Array2D_Element_Damping_Force, \
                                Array_Damping_Force.reshape(1,-1), axis = 0)

    class Sub_Class_Element_Lead_Rubber_Bearing(object):
        def __init__(self, \
                        Array_K_LR_Element, \
                        Value_Threshold_LR_Element, Value_C, \
                        Array_Element_Initial_Restoring_Force, \
                        Array_Element_Initial_State):
            self.Bool_Debug = False
            self.Class_Name = 'Sub_Class_Element_Lead_Rubber_Bearing'
            self.Array2D_Element_Damping_Force = numpy.zeros(2).reshape(1,-1)
            self.Array2D_Element_Restoring_Force \
                = Array_Element_Initial_Restoring_Force.reshape(1,-1)
            self.Array2D_Element_State \
                = Array_Element_Initial_State.reshape(1,-1)
            self.Value_C = Value_C
            self.Array_K_LR_Element = Array_K_LR_Element
            self.Value_Threshold_LR_Element = Value_Threshold_LR_Element
            Temp_Array2D_Element_Damping_Matrix = numpy.zeros([2,2])
            Temp_Array2D_Element_Damping_Matrix[0,0] \
                = self.Value_C
            Temp_Array2D_Element_Damping_Matrix[0,1] \
                = - self.Value_C
            Temp_Array2D_Element_Damping_Matrix[1,0] \
                = - self.Value_C
            Temp_Array2D_Element_Damping_Matrix[1,1] \
                = self.Value_C
            self.Array2D_Damping_Matrix \
                = Temp_Array2D_Element_Damping_Matrix
            Temp_Array2D_Element_Stiffness_Matrix \
                = numpy.zeros([2,2])
            Temp_Array2D_Element_Stiffness_Matrix[0,0] \
                = self.Array_K_LR_Element[0]
            Temp_Array2D_Element_Stiffness_Matrix[0,1] \
                = - self.Array_K_LR_Element[0]
            Temp_Array2D_Element_Stiffness_Matrix[1,0] \
                = - self.Array_K_LR_Element[0]
            Temp_Array2D_Element_Stiffness_Matrix[1,1] \
                = self.Array_K_LR_Element[0]
            self.Array2D_Stiffness_Matrix \
                = Temp_Array2D_Element_Stiffness_Matrix
            self.Array_Current_Time_Step_Stage_Displacement \
                = numpy.zeros(Array_K_LR_Element.size)

        def Function_Estimate_Stage_Displacement(self, \
                Temp_Array_Element_State_Current):
            Temp_Array_Last_Step_Stage_Displacement \
                = self.Array_Current_Time_Step_Stage_Displacement
            Temp_Value_Last_Relative_Displacement \
                = self.Array2D_Element_State[-1,0] \
                    - self.Array2D_Element_State[-1,1]
            Temp_Value_New_Relative_Displacement \
                = Temp_Array_Element_State_Current[0] \
                    - Temp_Array_Element_State_Current[1]
            Temp_Array_New_Step_Delta_Stage_Displacement \
                = numpy.zeros(self.Array_K_LR_Element.size)
            if Temp_Array_Last_Step_Stage_Displacement[0] \
                    == self.Value_Threshold_LR_Element:
                if Temp_Value_New_Relative_Displacement \
                        >= Temp_Value_Last_Relative_Displacement:
                    Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 1-1')
                else:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 1-2')
            elif Temp_Array_Last_Step_Stage_Displacement[0] \
                    ==  - self.Value_Threshold_LR_Element:
                if Temp_Value_New_Relative_Displacement \
                        <= Temp_Value_Last_Relative_Displacement:
                    Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 2-1')
                else:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 2-2')
            else:
                Temp_Delta_Value \
                    = Temp_Value_New_Relative_Displacement \
                        - Temp_Value_Last_Relative_Displacement
                Temp_Value_New_Disp \
                    = Temp_Array_Last_Step_Stage_Displacement[0] \
                        + Temp_Delta_Value
                if  numpy.abs(Temp_Value_New_Disp) \
                        <= self.Value_Threshold_LR_Element:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = Temp_Delta_Value
                    # print('Condition 3-1', Temp_Delta_Value)
                else:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = numpy.sign(Temp_Delta_Value) \
                                * self.Value_Threshold_LR_Element \
                            - Temp_Array_Last_Step_Stage_Displacement[0]
                    if Temp_Value_New_Disp > self.Value_Threshold_LR_Element:
                        Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                            = Temp_Value_New_Disp \
                                - self.Value_Threshold_LR_Element
                    else:
                        Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                            = Temp_Value_New_Disp \
                                + self.Value_Threshold_LR_Element
                        # print('Condition 3-2-2')
            # print('Before Return:', \
            #           Temp_Array_New_Step_Delta_Stage_Displacement)
            return Temp_Array_New_Step_Delta_Stage_Displacement

        def Function_Estimate_Restoring_Force(self, \
                Temp_Array_Element_State_Current):
            Temp_Array_New_Step_Delta_Stage_Displacement \
                = self.Function_Estimate_Stage_Displacement(\
                        Temp_Array_Element_State_Current)
            Temp_Value_Current_Force \
                = self.Array2D_Element_Restoring_Force[-1,0] \
                    + numpy.sum(self.Array_K_LR_Element \
                                * Temp_Array_New_Step_Delta_Stage_Displacement)
            Temp_Array_Element_Restoring_Force_Current \
                = numpy.array([Temp_Value_Current_Force, \
                                - Temp_Value_Current_Force])
            return Temp_Array_Element_Restoring_Force_Current

        def Function_Update_Restoring_Force(self, \
                Array_Element_Restoring_Force_Current):
            self.Array2D_Element_Restoring_Force \
                = numpy.append(\
                        self.Array2D_Element_Restoring_Force, \
                        Array_Element_Restoring_Force_Current.reshape(1,-1), \
                        axis = 0)

        def Function_Estimate_Damping_Force(self, \
                Temp_Array_State_Node_Current):
            # print(Temp_Array_State_Node_Current)
            Array_Damping_Force \
                = self.Array2D_Damping_Matrix\
                        .dot(Temp_Array_State_Node_Current[2:])
            return Array_Damping_Force

        def Function_Update_Damping_Force(self, Array_Damping_Force):
            self.Array2D_Element_Damping_Force \
                = numpy.append(self.Array2D_Element_Damping_Force, \
                                Array_Damping_Force.reshape(1,-1), axis = 0)

        def Function_Update_State(self, Array_State):
            # print('\nNew_Iteration:')
            # print('Last state', self.Array2D_Element_State[-1,:])
            Temp_Array_New_Step_Delta_Stage_Displacement \
                = self.Function_Estimate_Stage_Displacement(Array_State)
            self.Array_Current_Time_Step_Stage_Displacement \
                = self.Array_Current_Time_Step_Stage_Displacement \
                    + Temp_Array_New_Step_Delta_Stage_Displacement
            self.Array2D_Element_State \
                = numpy.append(self.Array2D_Element_State, \
                                Array_State.reshape(1,-1), axis = 0)
            
    class Sub_Class_Element_BRB(object):
        def __init__(self, \
                    Array_K_BRB_Element, \
                    Value_Threshold_BRB_Element, Value_C, \
                    Array_Element_Initial_Restoring_Force, \
                    Array_Element_Initial_State):
            self.Bool_Debug = False
            self.Class_Name = 'Sub_Class_Element_BRB'
            self.Array_K_BRB_Element = Array_K_BRB_Element
            self.Value_Threshold_BRB_Element = Value_Threshold_BRB_Element
            self.Array2D_Element_Damping_Force = numpy.zeros(2).reshape(1,-1)
            self.Array2D_Element_Restoring_Force \
                = Array_Element_Initial_Restoring_Force.reshape(1,-1)
            self.Array2D_Element_State \
                = Array_Element_Initial_State.reshape(1,-1)
            Temp_Array2D_Element_Damping_Matrix = numpy.zeros([2,2])
            Temp_Array2D_Element_Damping_Matrix[0,0] = Value_C
            Temp_Array2D_Element_Damping_Matrix[0,1] = - Value_C
            Temp_Array2D_Element_Damping_Matrix[1,0] = - Value_C
            Temp_Array2D_Element_Damping_Matrix[1,1] = Value_C
            self.Array2D_Damping_Matrix = Temp_Array2D_Element_Damping_Matrix
            Temp_Array2D_Element_Stiffness_Matrix = numpy.zeros([2,2])
            Temp_Array2D_Element_Stiffness_Matrix[0,0] \
                = self.Array_K_BRB_Element[0]
            Temp_Array2D_Element_Stiffness_Matrix[0,1] \
                = - self.Array_K_BRB_Element[0]
            Temp_Array2D_Element_Stiffness_Matrix[1,0] \
                = - self.Array_K_BRB_Element[0]
            Temp_Array2D_Element_Stiffness_Matrix[1,1] \
                = self.Array_K_BRB_Element[0]
            self.Array2D_Stiffness_Matrix \
                = Temp_Array2D_Element_Stiffness_Matrix
            self.Array_Current_Time_Step_Stage_Displacement \
                = numpy.zeros(Array_K_BRB_Element.size)

        def Function_Estimate_Stage_Displacement(self, \
                Temp_Array_Element_State_Current):
            Temp_Array_Last_Step_Stage_Displacement \
                = self.Array_Current_Time_Step_Stage_Displacement
            Temp_Value_Last_Relative_Displacement \
                = self.Array2D_Element_State[-1,0] \
                    - self.Array2D_Element_State[-1,1]
            Temp_Value_New_Relative_Displacement \
                = Temp_Array_Element_State_Current[0] \
                    - Temp_Array_Element_State_Current[1]
            Temp_Array_New_Step_Delta_Stage_Displacement \
                = numpy.zeros(self.Array_K_BRB_Element.size)
            if Temp_Array_Last_Step_Stage_Displacement[0] \
                    == self.Value_Threshold_BRB_Element:
                if Temp_Value_New_Relative_Displacement \
                        >= Temp_Value_Last_Relative_Displacement:
                    Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 1-1')
                else:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 1-2')
            elif Temp_Array_Last_Step_Stage_Displacement[0] \
                    ==  - self.Value_Threshold_BRB_Element:
                if Temp_Value_New_Relative_Displacement \
                        <= Temp_Value_Last_Relative_Displacement:
                    Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 2-1')
                else:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = Temp_Value_New_Relative_Displacement \
                            - Temp_Value_Last_Relative_Displacement
                    # print('Condition 2-2')
            else:
                Temp_Delta_Value \
                    = Temp_Value_New_Relative_Displacement \
                        - Temp_Value_Last_Relative_Displacement
                Temp_Value_New_Disp \
                    = Temp_Array_Last_Step_Stage_Displacement[0] \
                        + Temp_Delta_Value
                if  numpy.abs(Temp_Value_New_Disp) \
                        <= self.Value_Threshold_BRB_Element:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = Temp_Delta_Value
                    # print('Condition 3-1', Temp_Delta_Value)
                else:
                    Temp_Array_New_Step_Delta_Stage_Displacement[0] \
                        = numpy.sign(Temp_Delta_Value) \
                                * self.Value_Threshold_BRB_Element \
                            - Temp_Array_Last_Step_Stage_Displacement[0]
                    if Temp_Value_New_Disp > self.Value_Threshold_BRB_Element:
                        Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                            = Temp_Value_New_Disp \
                                - self.Value_Threshold_BRB_Element
                    else:
                        Temp_Array_New_Step_Delta_Stage_Displacement[1] \
                            = Temp_Value_New_Disp \
                                + self.Value_Threshold_BRB_Element
            return Temp_Array_New_Step_Delta_Stage_Displacement

        def Function_Estimate_Restoring_Force(self, \
                Temp_Array_Element_State_Current):
            Temp_Array_New_Step_Delta_Stage_Displacement \
                = self.Function_Estimate_Stage_Displacement(\
                        Temp_Array_Element_State_Current)
            Temp_Value_Current_Force \
                = self.Array2D_Element_Restoring_Force[-1,0] \
                    + numpy.sum(self.Array_K_BRB_Element \
                                * Temp_Array_New_Step_Delta_Stage_Displacement)
            Temp_Array_Element_Restoring_Force_Current \
                = numpy.array([Temp_Value_Current_Force, \
                                - Temp_Value_Current_Force])
            return Temp_Array_Element_Restoring_Force_Current

        def Function_Update_Restoring_Force(self, \
                Array_Element_Restoring_Force_Current):
            self.Array2D_Element_Restoring_Force \
                = numpy\
                    .append(\
                        self.Array2D_Element_Restoring_Force, \
                        Array_Element_Restoring_Force_Current.reshape(1,-1), \
                        axis = 0)

        def Function_Estimate_Damping_Force(self, \
                Temp_Array_State_Node_Current):
            # print(Temp_Array_State_Node_Current)
            Array_Damping_Force \
                = self.Array2D_Damping_Matrix\
                        .dot(Temp_Array_State_Node_Current[2:])
            return Array_Damping_Force

        def Function_Update_Damping_Force(self, Array_Damping_Force):
            self.Array2D_Element_Damping_Force \
                = numpy.append(self.Array2D_Element_Damping_Force, \
                                Array_Damping_Force.reshape(1,-1), axis = 0)

        def Function_Update_State(self, Array_State):
            Temp_Array_New_Step_Delta_Stage_Displacement \
                = self.Function_Estimate_Stage_Displacement(Array_State)
            self.Array_Current_Time_Step_Stage_Displacement \
                = self.Array_Current_Time_Step_Stage_Displacement \
                        + Temp_Array_New_Step_Delta_Stage_Displacement
            self.Array2D_Element_State \
                = numpy.append(self.Array2D_Element_State, \
                                Array_State.reshape(1,-1), axis = 0)

    def Function_Relative_Stiffness(self, Value_z):
        Value_Stiffness_Ratio \
            = 1 - (1 - 0.55) * (Value_z / self.Value_Height)**2
        return Value_Stiffness_Ratio

    def Function_Relative_Mass(self, Value_z):
        Value_Mass_Ratio = 1 - 0.5 * (Value_z / self.Value_Height)
        return Value_Mass_Ratio

    def Function_Model_Creation_Normal(self):
        self.Array_Node_Height \
            = numpy.linspace(self.Value_Height, \
                                0, \
                                self.Int_Number_of_Node_Total)
        self.Array2D_Mapping_Element_to_Node \
            = numpy.zeros([self.Int_Number_of_Element, 2], dtype = numpy.int)
        List_Element = []
        for i_Element in range(self.Int_Number_of_Element):
            Temp_Value_Element_Mid_Height \
                = (self.Array_Node_Height[i_Element] \
                    + self.Array_Node_Height[i_Element + 1]) / 2
            Temp_Value_Stiffness_Ratio \
                = self.Function_Relative_Stiffness(\
                        Temp_Value_Element_Mid_Height)
            Temp_Object_Element_Normal \
                = self.Sub_Class_Element_Normal(\
                        self.Value_C, \
                        self.Value_K * Temp_Value_Stiffness_Ratio)
            List_Element.append(Temp_Object_Element_Normal)
            self.Array2D_Mapping_Element_to_Node[i_Element,:] \
                = numpy.array([i_Element, i_Element + 1], dtype = numpy.int)
        self.List_Element = List_Element
        self.Array2D_State \
            = numpy.zeros(self.Int_Number_of_Node_Total * 2).reshape(1,-1)
        self.Array2D_Node_Force \
            = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array_Node_Weight \
            = numpy.zeros(self.Int_Number_of_Node_Total)
        for i_Node in range(self.Int_Number_of_Node_Total):
            Temp_Value_Node_Height = self.Array_Node_Height[i_Node]
            Temp_Value_Mass_Ratio \
                = self.Function_Relative_Mass(Temp_Value_Node_Height)
            self.Array_Node_Weight[i_Node] \
                = self.Value_Node_Mass_Basic * Temp_Value_Mass_Ratio
        self.Array2D_Mass_Matrix = numpy.diag(self.Array_Node_Weight)
        self.Array2D_Restoring_Force \
            = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array2D_Damping_Force \
            = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        # print(self.Array2D_Node_Force.shape)

    def Function_Model_Creation_Base_Isolation(self, \
            Array_K_LR_Element, Value_Threshold_LR_Element):
        self.Int_Number_of_Node_Total = self.Int_Number_of_Node_Total + 1 
        self.Int_Number_of_Node_Free = self.Int_Number_of_Node_Free + 1 
        self.Int_Number_of_Element = self.Int_Number_of_Element + 1 
        self.Array_K_LR_Element = Array_K_LR_Element
        self.Value_Threshold_LR_Element = Value_Threshold_LR_Element
        self.Array_Node_Height \
            = numpy.append(\
                    numpy.linspace(self.Value_Height, 0, \
                                    self.Int_Number_of_Node_Total - 1), \
                    0)
        self.Array2D_Mapping_Element_to_Node \
            = numpy.zeros([self.Int_Number_of_Element, 2], dtype = numpy.int)
        List_Element = []
        for i_Element in range(self.Int_Number_of_Element - 1):
            Temp_Value_Element_Mid_Height \
                = (self.Array_Node_Height[i_Element] \
                    + self.Array_Node_Height[i_Element + 1]) / 2
            Temp_Value_Stiffness_Ratio \
                = self.Function_Relative_Stiffness(\
                        Temp_Value_Element_Mid_Height)
            Temp_Object_Element_Normal \
                = self.Sub_Class_Element_Normal(\
                        self.Value_C, \
                        self.Value_K * Temp_Value_Stiffness_Ratio)
            List_Element.append(Temp_Object_Element_Normal)
            self.Array2D_Mapping_Element_to_Node[i_Element,:] \
                = numpy.array([i_Element, i_Element + 1], dtype = numpy.int)
        i_Element = self.Int_Number_of_Element - 1
        Temp_Object_Element_LR \
            = self.Sub_Class_Element_Lead_Rubber_Bearing(\
                self.Array_K_LR_Element, \
                self.Value_Threshold_LR_Element, \
                self.Value_C,\
                Array_Element_Initial_Restoring_Force = numpy.zeros(2), \
                Array_Element_Initial_State = numpy.zeros(4))
        List_Element.append(Temp_Object_Element_LR)
        self.Array2D_Mapping_Element_to_Node[i_Element,:] \
            = numpy.array([i_Element, i_Element + 1], dtype = numpy.int)
        self.List_Element = List_Element
        self.Array2D_State = numpy.zeros(self.Int_Number_of_Node_Total * 2).reshape(1,-1)
        self.Array2D_Node_Force = numpy.append(numpy.zeros(self.Int_Number_of_Node_Free), 0).reshape(1,-1)
        self.Array_Node_Weight = numpy.zeros(self.Int_Number_of_Node_Total)
        for i_Node in range(self.Int_Number_of_Node_Total):
            Temp_Value_Node_Height = self.Array_Node_Height[i_Node]
            Temp_Value_Mass_Ratio = self.Function_Relative_Mass(Temp_Value_Node_Height)
            self.Array_Node_Weight[i_Node] = self.Value_Node_Mass_Basic * Temp_Value_Mass_Ratio
        # self.Array_Node_Weight[-2] = self.Array_Node_Weight[-2] / 10
        self.Array2D_Mass_Matrix = numpy.diag(self.Array_Node_Weight)
        self.Array2D_Restoring_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array2D_Damping_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)

    def Function_Model_Creation_with_BRB(self, Value_Threshold_BRB_Element):
        self.Array_Node_Height = numpy.linspace(self.Value_Height, 0, self.Int_Number_of_Node_Total)
        self.Value_K = self.Value_K / 2
        self.Array_K_BRB_Element = numpy.array([self.Value_K, self.Value_K / 20])
        self.Value_Threshold_BRB_Element = Value_Threshold_BRB_Element
        self.Int_Number_of_Element_Normal = self.Int_Number_of_Element
        self.Int_Number_of_Element_BRB = int(self.Int_Number_of_Element_Normal)
        self.Int_Number_of_Element_Total = self.Int_Number_of_Element_Normal + self.Int_Number_of_Element_BRB
        self.Int_Number_of_Element = self.Int_Number_of_Element_Total
        self.Array2D_Mapping_Element_to_Node = numpy.zeros([self.Int_Number_of_Element_Total, 2], dtype = numpy.int)
        List_Element = []
        for i_Element in range(self.Int_Number_of_Element_Normal):
            Temp_Value_Element_Mid_Height \
                = (self.Array_Node_Height[i_Element] + self.Array_Node_Height[i_Element + 1]) / 2
            Temp_Value_Stiffness_Ratio = self.Function_Relative_Stiffness(Temp_Value_Element_Mid_Height)
            Temp_Object_Element_Normal \
                = self.Sub_Class_Element_Normal(\
                        self.Value_C, \
                        self.Value_K * Temp_Value_Stiffness_Ratio)
            List_Element.append(Temp_Object_Element_Normal)
            self.Array2D_Mapping_Element_to_Node[i_Element,:] \
                = numpy.array([i_Element, i_Element + 1], dtype = numpy.int)
        for i_Element in numpy.arange(self.Int_Number_of_Element_Normal, self.Int_Number_of_Element_Total):
            self.Array2D_Mapping_Element_to_Node[i_Element,:] \
                = self.Array2D_Mapping_Element_to_Node[2 * self.Int_Number_of_Element_Normal - 1 - i_Element,:]
            Temp_Value_Element_Mid_Height \
                = (self.Array_Node_Height[self.Array2D_Mapping_Element_to_Node[i_Element,0]] \
                    + self.Array_Node_Height[self.Array2D_Mapping_Element_to_Node[i_Element,1]]) / 2  
            Temp_Value_Stiffness_Ratio = self.Function_Relative_Stiffness(Temp_Value_Element_Mid_Height)   
            Temp_Value_Scale_C = 0.0
            Temp_Object_Element_BRB \
                = self.Sub_Class_Element_BRB(\
                        self.Array_K_BRB_Element * Temp_Value_Stiffness_Ratio, \
                        self.Value_Threshold_BRB_Element, \
                        self.Value_C * Temp_Value_Scale_C, \
                        numpy.zeros(2), \
                        numpy.zeros(2 * 2))
            List_Element.append(Temp_Object_Element_BRB)

        self.List_Element = List_Element
        self.Array2D_State = numpy.zeros(self.Int_Number_of_Node_Total * 2).reshape(1,-1)
        self.Array2D_Node_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array_Node_Weight = numpy.zeros(self.Int_Number_of_Node_Total)
        for i_Node in range(self.Int_Number_of_Node_Total):
            Temp_Value_Node_Height = self.Array_Node_Height[i_Node]
            Temp_Value_Mass_Ratio = self.Function_Relative_Mass(Temp_Value_Node_Height)
            self.Array_Node_Weight[i_Node] = self.Value_Node_Mass_Basic * Temp_Value_Mass_Ratio
        self.Array2D_Mass_Matrix = numpy.diag(self.Array_Node_Weight)
        self.Array2D_Restoring_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array2D_Damping_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        # print(self.Array2D_Node_Force.shape)

    def Function_Model_Creation_with_BRB_2(self, Value_Threshold_BRB_Element):
        self.Array_Node_Height = numpy.linspace(self.Value_Height, 0, self.Int_Number_of_Node_Total)
        self.Value_K = self.Value_K
        self.Array_K_BRB_Element = numpy.array([self.Value_K, self.Value_K / 20])
        self.Value_Threshold_BRB_Element = Value_Threshold_BRB_Element
        self.Int_Number_of_Element_Normal = self.Int_Number_of_Element
        self.Int_Number_of_Element_BRB = int(self.Int_Number_of_Element_Normal)
        self.Int_Number_of_Element_Total = self.Int_Number_of_Element_Normal + self.Int_Number_of_Element_BRB
        self.Int_Number_of_Element = self.Int_Number_of_Element_Total
        self.Array2D_Mapping_Element_to_Node = numpy.zeros([self.Int_Number_of_Element_Total, 2], dtype = numpy.int)
        List_Element = []
        for i_Element in range(self.Int_Number_of_Element_Normal):
            Temp_Value_Element_Mid_Height \
                = (self.Array_Node_Height[i_Element] + self.Array_Node_Height[i_Element + 1]) / 2
            Temp_Value_Stiffness_Ratio = self.Function_Relative_Stiffness(Temp_Value_Element_Mid_Height)
            Temp_Object_Element_Normal \
                = self.Sub_Class_Element_Normal(\
                        self.Value_C, \
                        self.Value_K * Temp_Value_Stiffness_Ratio)
            List_Element.append(Temp_Object_Element_Normal)
            self.Array2D_Mapping_Element_to_Node[i_Element,:] \
                = numpy.array([i_Element, i_Element + 1], dtype = numpy.int)
        for i_Element in numpy.arange(self.Int_Number_of_Element_Normal, self.Int_Number_of_Element_Total):
            self.Array2D_Mapping_Element_to_Node[i_Element,:] \
                = self.Array2D_Mapping_Element_to_Node[2 * self.Int_Number_of_Element_Normal - 1 - i_Element,:]
            Temp_Value_Element_Mid_Height \
                = (self.Array_Node_Height[self.Array2D_Mapping_Element_to_Node[i_Element,0]] \
                    + self.Array_Node_Height[self.Array2D_Mapping_Element_to_Node[i_Element,1]]) / 2  
            Temp_Value_Stiffness_Ratio = self.Function_Relative_Stiffness(Temp_Value_Element_Mid_Height)
            Temp_Value_Stiffness_Ratio = Temp_Value_Stiffness_Ratio * 2
            Temp_Value_Scale_C = 0.0
            Temp_Object_Element_BRB \
                = self.Sub_Class_Element_BRB(\
                        self.Array_K_BRB_Element * Temp_Value_Stiffness_Ratio, \
                        self.Value_Threshold_BRB_Element, \
                        self.Value_C * Temp_Value_Scale_C, \
                        numpy.zeros(2), \
                        numpy.zeros(2 * 2))
            List_Element.append(Temp_Object_Element_BRB)

        self.List_Element = List_Element
        self.Array2D_State = numpy.zeros(self.Int_Number_of_Node_Total * 2).reshape(1,-1)
        self.Array2D_Node_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array_Node_Weight = numpy.zeros(self.Int_Number_of_Node_Total)
        for i_Node in range(self.Int_Number_of_Node_Total):
            Temp_Value_Node_Height = self.Array_Node_Height[i_Node]
            Temp_Value_Mass_Ratio = self.Function_Relative_Mass(Temp_Value_Node_Height)
            self.Array_Node_Weight[i_Node] = self.Value_Node_Mass_Basic * Temp_Value_Mass_Ratio
        self.Array2D_Mass_Matrix = numpy.diag(self.Array_Node_Weight)
        self.Array2D_Restoring_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        self.Array2D_Damping_Force = numpy.zeros(self.Int_Number_of_Node_Total).reshape(1,-1)
        # print(self.Array2D_Node_Force.shape)


    def Function_Estimate_Model_Restoring_Force(self, Temp_Array_Model_State_Current):
        # print('Function_RForce:\n',Temp_Array_Model_State_Current)
        Temp_Array_Model_Restoring_Force_Current = numpy.zeros(self.Int_Number_of_Node_Total)
        for i_Element in range(self.Int_Number_of_Element):
            Temp_Array_Element_State_Current = numpy.zeros(4)
            Temp_Array_Int_Index_of_Node_Disp = self.Array2D_Mapping_Element_to_Node[i_Element,:]
            # print(Temp_Array_Int_Index_of_Node)
            # print(Temp_Array_Int_Index_of_Node)
            Temp_Array_Element_State_Current[:2] \
                = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node_Disp]
            Temp_Array_Int_Index_of_Node_Velo \
                = self.Array2D_Mapping_Element_to_Node[i_Element,:] + self.Int_Number_of_Node_Total
            # print(Temp_Array_Int_Index_of_Node)
            Temp_Array_Element_State_Current[2:] \
                = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node_Velo]
            Temp_Array_Element_Restoring_Force_Current \
                = self.List_Element[i_Element].Function_Estimate_Restoring_Force(Temp_Array_Element_State_Current)
            Temp_Array_Model_Restoring_Force_Current[Temp_Array_Int_Index_of_Node_Disp] \
                = Temp_Array_Model_Restoring_Force_Current[Temp_Array_Int_Index_of_Node_Disp] \
                    + Temp_Array_Element_Restoring_Force_Current
            # print('\t \t Element', i_Element ,'Function_ERF', Temp_Array_Element_Restoring_Force_Current)
            # print('\t \t Element status', Temp_Array_Element_State_Current)
        # i_Element = self.Int_Number_of_Element - 1 # The LR Element
        # Temp_Array_Element_State_Current = numpy.zeros(4)
        # Temp_Array_Int_Index_of_Node = self.Array2D_Mapping_Element_to_Node[i_Element,:]
        # # print(Temp_Array_Int_Index_of_Node)
        # Temp_Array_Element_State_Current[:2] \
        #     = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node]
        # Temp_Array_Int_Index_of_Node \
        #     = self.Array2D_Mapping_Element_to_Node[i_Element,:] + self.Int_Number_of_Node_Total
        # # print(Temp_Array_Int_Index_of_Node)
        # Temp_Array_Element_State_Current[2:] \
        #     = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node]
        # Temp_Array_Element_Restoring_Force_Current \
        #     = self.List_Element[i_Element].Function_Estimate_Restoring_Force(Temp_Array_Element_State_Current)
        # # print(Temp_Array_Model_State_Current)
        # # # print(Temp_Array_Element_State_Current)
        # # print(Temp_Array_Element_Restoring_Force_Current)
        # # print(Temp_Array_Element_Restoring_Force_Current)
        # # print('Model_Function_Test_Restoring_Force', Temp_Array_Model_Restoring_Force_Current.shape)
        # # print(Temp_Array_Element_Restoring_Force_Current.shape)
        # # print(Temp_Array_Element_Restoring_Force_Current)
        # Temp_Array_Model_Restoring_Force_Current[i_Element: i_Element + 2] \
        #     =  Temp_Array_Model_Restoring_Force_Current[i_Element: i_Element + 2] \
        #         + Temp_Array_Element_Restoring_Force_Current
        # print(Temp_Array_Model_Restoring_Force_Current)
        # print('\tSM:', self.List_Element[0].Array2D_Stiffness_Matrix)
        # print('\tSM:', self.List_Element[1].Array2D_Stiffness_Matrix)
        # print('\tRS:', Temp_Array_Model_State_Current)
        # print('\tRF:', Temp_Array_Model_Restoring_Force_Current)
        return Temp_Array_Model_Restoring_Force_Current

    def Function_Estimate_Model_Damping_Force(self, Temp_Array_Model_State_Current):
        Temp_Array_Model_Damping_Force_Current = numpy.zeros(self.Int_Number_of_Node_Total)
        for i_Element in range(self.Int_Number_of_Element):
            Temp_Array_Element_State_Current = numpy.zeros(4)
            Temp_Array_Int_Index_of_Node_Disp = self.Array2D_Mapping_Element_to_Node[i_Element,:]
            # print(Temp_Array_Int_Index_of_Node)
            # print(Temp_Array_Int_Index_of_Node)
            Temp_Array_Element_State_Current[:2] \
                = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node_Disp]
            Temp_Array_Int_Index_of_Node_Velo \
                = self.Array2D_Mapping_Element_to_Node[i_Element,:] + self.Int_Number_of_Node_Total
            # print(Temp_Array_Int_Index_of_Node)
            Temp_Array_Element_State_Current[2:] \
                = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node_Velo]
            Temp_Array_Model_Damping_Force_Current[Temp_Array_Int_Index_of_Node_Disp] \
                = Temp_Array_Model_Damping_Force_Current[Temp_Array_Int_Index_of_Node_Disp] \
                    + self.List_Element[i_Element].Function_Estimate_Damping_Force(Temp_Array_Element_State_Current)
        # i_Element = self.Int_Number_of_Element - 1 # The LR Element
        # Temp_Array_Element_State_Current = numpy.zeros(4)
        # Temp_Array_Int_Index_of_Node = self.Array2D_Mapping_Element_to_Node[i_Element,:]
        # # print(Temp_Array_Int_Index_of_Node)
        # # print(Temp_Array_Int_Index_of_Node)
        # Temp_Array_Element_State_Current[:2] \
        #     = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node]
        # Temp_Array_Int_Index_of_Node \
        #     = self.Array2D_Mapping_Element_to_Node[i_Element,:] + self.Int_Number_of_Node_Total
        # # print(Temp_Array_Int_Index_of_Node)
        # Temp_Array_Element_State_Current[2:] \
        #     = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node]
        # Temp_Array_Model_Damping_Force_Current[i_Element: i_Element + 2] \
        #         = Temp_Array_Model_Damping_Force_Current[i_Element: i_Element + 2] \
        #             + self.List_Element[i_Element].Function_Estimate_Damping_Force(Temp_Array_Element_State_Current)
        # print('\tDS:', Temp_Array_Model_State_Current)
        # print('\tDF:', Temp_Array_Model_Damping_Force_Current)
        return Temp_Array_Model_Damping_Force_Current

    def Function_Update_Model_Runge_Kutta(self, Array_External_Force, Value_Delta_T):
        # Use status until last time step
        # Use force until current time step
        # Source: [http://www.wind.civil.aau.dk/lecture/7sem/notes/Lecture6.pdf, Page 44]
        Temp_Array_Z_Initial = self.Array2D_State[-1,:]
        Array2D_M_inv = numpy.linalg.inv(self.Array2D_Mass_Matrix)
        Temp_Array_Node_Force = numpy.append(Array_External_Force, 0)
        # print(self.Array2D_Node_Force.shape)
        # print(Temp_Array_Node_Force.shape)
        self.Array2D_Node_Force = numpy.append(self.Array2D_Node_Force, Temp_Array_Node_Force.reshape(1,-1), axis = 0)
        Temp_Array2D_External_Force_Last_Two_Step = self.Array2D_Node_Force[-2:,:]
        Temp_Int_Number_of_Node_Free = self.Int_Number_of_Node_Free
        Temp_Int_Number_of_Node_Total = self.Int_Number_of_Node_Total
        Temp_Array2D_h = numpy.zeros([Temp_Int_Number_of_Node_Total * 2,4])
        # 1st calculation
        Temp_Array_Force = Temp_Array2D_External_Force_Last_Two_Step[0,:]
        Temp_Array_z = Temp_Array_Z_Initial.copy()
        Temp_Array2D_h[:Temp_Int_Number_of_Node_Free,0] \
            = Temp_Array_z[Temp_Int_Number_of_Node_Total:-1]
        # print('1st \t \t Z Array', Temp_Array_z)
        Temp_Array_Model_Restoring_Force_Current \
            = self.Function_Estimate_Model_Restoring_Force(Temp_Array_z)
        # print('1st \t \t RF', Temp_Array_Model_Restoring_Force_Current)
        Temp_Array_Model_Damping_Force_Current \
            = self.Function_Estimate_Model_Damping_Force(Temp_Array_z)
        Temp_Array2D_h[Temp_Int_Number_of_Node_Total:-1,0] \
            = Array2D_M_inv[:-1,:-1].dot(Temp_Array_Force[:-1] \
                                - Temp_Array_Model_Restoring_Force_Current[:-1]\
                                - Temp_Array_Model_Damping_Force_Current[:-1])
        # print('1st', '\nArray_Force:\n', Temp_Array_Force , '\nArray_Z:\n',  Temp_Array_z, '\nArray_h:\n', Temp_Array2D_h[:,0])
        # print('1st \t Mass', Array2D_M_inv, Temp_Array_Force)
        # print('1st \t Z Array', Temp_Array_z)
        # print('1st \t Restorng Force', Temp_Array_Model_Restoring_Force_Current)
        # print('1st \t Damping Force', Temp_Array_Model_Damping_Force_Current)
        # 2nd calculation
        Temp_Array_Force \
            = (Temp_Array2D_External_Force_Last_Two_Step[0,:] + Temp_Array2D_External_Force_Last_Two_Step[1,:]) / 2
        Temp_Array_z \
            = Temp_Array_Z_Initial + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,0]
        Temp_Array2D_h[:Temp_Int_Number_of_Node_Free,1] \
            = Temp_Array_z[Temp_Int_Number_of_Node_Total:-1]
        Temp_Array_Model_Restoring_Force_Current \
            = self.Function_Estimate_Model_Restoring_Force(Temp_Array_z)
        Temp_Array_Model_Damping_Force_Current \
            = self.Function_Estimate_Model_Damping_Force(Temp_Array_z)
        Temp_Array2D_h[Temp_Int_Number_of_Node_Total:-1,1] \
            = Array2D_M_inv[:-1,:-1].dot(Temp_Array_Force[:-1] \
                                - Temp_Array_Model_Restoring_Force_Current[:-1]\
                                - Temp_Array_Model_Damping_Force_Current[:-1])
        # print('2nd', '\nArray_Force:\n', Temp_Array_Force, '\nArray_Z:\n', Temp_Array_z, '\nArray_h:\n', Temp_Array2D_h[:,1])
        # print('2nd \t Mass', Array2D_M_inv, Temp_Array_Force)
        # print('2nd \t Restorng Force', Temp_Array_Model_Restoring_Force_Current)
        # print('2nd \t Damping Force', Temp_Array_Model_Damping_Force_Current)
        # 3rd calculation
        Temp_Array_Force \
            = (Temp_Array2D_External_Force_Last_Two_Step[0,:] + Temp_Array2D_External_Force_Last_Two_Step[1,:]) / 2
        Temp_Array_z \
            = Temp_Array_Z_Initial + 1 / 2 * Value_Delta_T * Temp_Array2D_h[:,1]
        Temp_Array2D_h[:Temp_Int_Number_of_Node_Free,2] \
            = Temp_Array_z[Temp_Int_Number_of_Node_Total:-1]
        Temp_Array_Model_Restoring_Force_Current \
            = self.Function_Estimate_Model_Restoring_Force(Temp_Array_z)
        Temp_Array_Model_Damping_Force_Current \
            = self.Function_Estimate_Model_Damping_Force(Temp_Array_z)
        Temp_Array2D_h[Temp_Int_Number_of_Node_Total:-1,2] \
            = Array2D_M_inv[:-1,:-1].dot(Temp_Array_Force[:-1] \
                                - Temp_Array_Model_Restoring_Force_Current[:-1]\
                                - Temp_Array_Model_Damping_Force_Current[:-1])
        # print('3rd', '\nArray_Force:\n', Temp_Array_Force , '\nArray_Z:\n',  Temp_Array_z, '\nArray_h:\n', Temp_Array2D_h[:,2])
        # 4th calculation
        Temp_Array_Force = Temp_Array2D_External_Force_Last_Two_Step[1,:]
        Temp_Array_z \
            = Temp_Array_Z_Initial + Value_Delta_T * Temp_Array2D_h[:,2]
        Temp_Array2D_h[:Temp_Int_Number_of_Node_Free,3] \
            = Temp_Array_z[Temp_Int_Number_of_Node_Total:-1]
        Temp_Array_Model_Restoring_Force_Current \
            = self.Function_Estimate_Model_Restoring_Force(Temp_Array_z)
        Temp_Array_Model_Damping_Force_Current \
            = self.Function_Estimate_Model_Damping_Force(Temp_Array_z)
        Temp_Array2D_h[Temp_Int_Number_of_Node_Total:-1,3] \
            = Array2D_M_inv[:-1,:-1].dot(Temp_Array_Force[:-1] \
                                - Temp_Array_Model_Restoring_Force_Current[:-1]\
                                - Temp_Array_Model_Damping_Force_Current[:-1])                                                                      
        # print('4th', '\nArray_Force:\n', Temp_Array_Force , '\nArray_Z:\n',  Temp_Array_z, '\nArray_h:\n', Temp_Array2D_h[:,3])
        # print('Detail:\n')
        # print('\t:', 'ZA', Temp_Array_z)
        # print('\t:', 'RF', Temp_Array_Model_Restoring_Force_Current)
        # print('\t:', 'DF', Temp_Array_Model_Damping_Force_Current)

        # print(Temp_Array2D_h)
        # print(self.Array2D_State.shape)
        Temp_Array_Model_State_Current \
            = Temp_Array_Z_Initial\
            + Value_Delta_T / 6 \
                * (Temp_Array2D_h[:,0] + 2 * Temp_Array2D_h[:,1] + 2 * Temp_Array2D_h[:,2] + Temp_Array2D_h[:,3])
        self.Array2D_State = numpy.append(self.Array2D_State, Temp_Array_Model_State_Current.reshape(1,-1), axis = 0) 

        Temp_Array_Model_Restoring_Force_Current \
            = self.Function_Estimate_Model_Restoring_Force(Temp_Array_Model_State_Current)
        Temp_Array_Model_Damping_Force_Current \
            = self.Function_Estimate_Model_Damping_Force(Temp_Array_Model_State_Current)
        # print(Temp_Array_Model_State_Current)
        # print(Temp_Array_Model_Restoring_Force_Current)
        self.Array2D_Node_Force[-1,-1] \
            = Temp_Array_Model_Restoring_Force_Current[-1] + Temp_Array_Model_Damping_Force_Current[-1]
        self.Array2D_Restoring_Force \
            = numpy.append(self.Array2D_Restoring_Force, \
                            Temp_Array_Model_Restoring_Force_Current.reshape(1,-1), axis = 0)
        self.Array2D_Damping_Force \
            = numpy.append(self.Array2D_Damping_Force, \
                            Temp_Array_Model_Damping_Force_Current.reshape(1,-1), axis = 0)
        for i_Element in range(self.Int_Number_of_Element):
            Temp_Array_Element_State_Current = numpy.zeros(4)
            Temp_Array_Int_Index_of_Node_Disp = self.Array2D_Mapping_Element_to_Node[i_Element,:]
            Temp_Array_Element_State_Current[:2] \
                = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node_Disp]
            Temp_Array_Int_Index_of_Node_Velo \
                = self.Array2D_Mapping_Element_to_Node[i_Element,:] + self.Int_Number_of_Node_Total
            Temp_Array_Element_State_Current[2:] \
                = Temp_Array_Model_State_Current[Temp_Array_Int_Index_of_Node_Velo]
            # Temp_Array_Element_Restoring_Force_Current = numpy.zeros(2)
            # print(Temp_Array_Element_State_Current.shape)
            Temp_Array_Element_Restoring_Force_Current \
                = self.List_Element[i_Element].Function_Estimate_Restoring_Force(Temp_Array_Element_State_Current)
            self.List_Element[i_Element].Function_Update_Restoring_Force(Temp_Array_Element_Restoring_Force_Current)
            # Temp_Array_Element_Damping_Force_Current = numpy.zeros(2)
            Temp_Array_Element_Damping_Force_Current \
                = self.List_Element[i_Element].Function_Estimate_Damping_Force(Temp_Array_Element_State_Current)
            self.List_Element[i_Element].Function_Update_Damping_Force(Temp_Array_Element_Damping_Force_Current)
            self.List_Element[i_Element].Function_Update_State(Temp_Array_Element_State_Current)

    def Function_Mode_Decomposition(self):
        self.Array2D_Model_Stiffness_Matrix \
            = numpy.zeros([self.Int_Number_of_Node_Total, self.Int_Number_of_Node_Total])
        for i_Element in range(self.Int_Number_of_Element):
            Temp_Array_Int_Index_of_Node_Disp = self.Array2D_Mapping_Element_to_Node[i_Element,:]
            Temp_Array2D_Element_Stiffness_Matrix = self.List_Element[i_Element].Array2D_Stiffness_Matrix
            for i_Row in range(2):
                for i_Col in range(2):
                    self.Array2D_Model_Stiffness_Matrix\
                        [Temp_Array_Int_Index_of_Node_Disp[i_Row], Temp_Array_Int_Index_of_Node_Disp[i_Col]] \
                            += Temp_Array2D_Element_Stiffness_Matrix[i_Row, i_Col]
        Temp_Array2D = numpy.linalg.inv(self.Array2D_Mass_Matrix).dot(self.Array2D_Model_Stiffness_Matrix)
        Tuple_Function_Return = numpy.linalg.eig(Temp_Array2D[:-1,:-1])
        self.Array_Model_Eigen_Value = Tuple_Function_Return[0]
        Temp_Array2D_Eigen_Vector = Tuple_Function_Return[1]
        # Temp_Array2D_Mass_Normalized_Eigen_Vector = numpy.sqrt().dot(Temp_Array2D_Eigen_Vector)
        self.Array_Model_Natural_Frequency = numpy.sqrt(self.Array_Model_Eigen_Value) / 2 / numpy.pi
        self.Array2D_Model_Mode_Shape = Temp_Array2D_Eigen_Vector
        Temp_Array_Sorted_Index = numpy.argsort(self.Array_Model_Natural_Frequency)
        self.Array_Model_Natural_Frequency = self.Array_Model_Natural_Frequency[Temp_Array_Sorted_Index]
        self.Array2D_Model_Mode_Shape = self.Array2D_Model_Mode_Shape[:,Temp_Array_Sorted_Index]


class Class_Structural_Response_Wavelet_Galerkin():
    '''
    Class defined for structural response calculation by wavelt-Galerkin method
    '''

    def __init__(self):
        self.Class_Name = 'Class of structural response calculation with wavelet-Galerkin method'

    def Function_Signal_Generation_Wavelet_Galerkin(self, Parameter_Signal_Name, Flag_Draw = False):
        """
        Generate input signal for wawelet-Galerkin code

        ----------
        Input:
            Parameter_Signal_Name
        ----------
        Output:
            Array_Time, Array_Signal
        ----------
        Notes:
            Since the structures usually have a frequency of 0.05-10 Hz,  the sampling frequency should reach 200 Hz 
        (20 X 10 Hz) to ensure the fundamental mode can be captured. As a result, the sampling interval is set to be
        0.005s.
        """
        if Parameter_Signal_Name == 'Stationary_0':
            Parameter_Frequency = 0.5 # Hz
            Parameter_Total_Time = 100 # s
            Value_Delta_T = 0.005
            Int_Length_Signal = int(Parameter_Total_Time / Value_Delta_T)
            Array_Time = numpy.linspace(0, Int_Length_Signal * Value_Delta_T, Int_Length_Signal)
            Array_Signal = numpy.sin(Array_Time * 2 * numpy.pi * Parameter_Frequency)
        if Parameter_Signal_Name == 'Stationary_1':
            Parameter_Frequency = 0.5 # Hz
            Parameter_Total_Time = 50 # s
            Value_Delta_T = 0.005
            Int_Length_Signal = int(Parameter_Total_Time / Value_Delta_T)
            Array_Time = numpy.linspace(0, Int_Length_Signal * Value_Delta_T, Int_Length_Signal)
            Array_Signal = numpy.sin(Array_Time * 2 * numpy.pi * Parameter_Frequency)
        if Parameter_Signal_Name == 'Amplitude_Modulation_0':
            Parameter_Frequency = 0.5 # Hz
            Parameter_Total_Time = 100 # s
            Value_Delta_T = 0.005
            Int_Length_Signal = int(Parameter_Total_Time / Value_Delta_T)
            Array_Time = numpy.linspace(0, Int_Length_Signal * Value_Delta_T, Int_Length_Signal)
            Array_Signal = numpy.sin(Array_Time * 2 * numpy.pi * Parameter_Frequency)
            Array_Amplitude = numpy.linspace(0, 2, Array_Signal.size)
            Array_Amplitude[-Array_Signal.size //2 :] = 1
            Array_Amplitude[-Array_Signal.size //4 :] = 0
            Array_Signal = Array_Amplitude * Array_Signal
        if Parameter_Signal_Name == 'Amplitude_Modulation_1':
            Parameter_Frequency = 0.5 # Hz
            Parameter_Total_Time = 100 # s
            Value_Delta_T = 0.005
            Int_Length_Signal = int(Parameter_Total_Time / Value_Delta_T)
            Array_Time = numpy.linspace(0, Int_Length_Signal * Value_Delta_T, Int_Length_Signal)
            Array_Signal = numpy.sin(Array_Time * 2 * numpy.pi * Parameter_Frequency)
            Array_Amplitude = numpy.linspace(0, 2, Array_Signal.size)
            Array_Amplitude[: Array_Signal.size // 12] = 0
            Array_Amplitude[-Array_Signal.size //2 :] = 1
            Array_Amplitude[-Array_Signal.size //4 :] = 0
            Array_Signal = Array_Amplitude * Array_Signal
        if Parameter_Signal_Name == 'Amplitude_Modulation_2':
            Parameter_Frequency = 0.5 # Hz
            Parameter_Total_Time = 100 # s
            Value_Delta_T = 0.005
            Int_Length_Signal = int(Parameter_Total_Time / Value_Delta_T)
            Array_Time = numpy.linspace(0, Int_Length_Signal * Value_Delta_T, Int_Length_Signal)
            Array_Signal = numpy.sin(Array_Time * 2 * numpy.pi * Parameter_Frequency)
            Array_Amplitude = numpy.linspace(0, 2, Array_Signal.size)
            Array_Amplitude[: Array_Signal.size // 6] = 0
            Array_Amplitude[Array_Signal.size // 6  :Array_Signal.size // 5] = 1
            Array_Amplitude[Array_Signal.size //5 :] = 0
            Array_Signal = Array_Amplitude * Array_Signal
        if Flag_Draw:
            pyplot.figure(figsize = (6,4), dpi = 200, facecolor = 'white')
            pyplot.plot(Array_Time,Array_Signal)
            pyplot.grid(True)
            pyplot.xlim(Array_Time.min(), Array_Time.max())
            pyplot.show()
        return Array_Time, Array_Signal

    def Function_Load_Basic_Parameters_Wavelet_Galerkin(self):
        Parameter_ksee = 0.05 # Damping rato
        Parameter_f = 0.2 #Hz
        Parameter_k = 1
        Parameter_Int_Refinement_Level = 6
        Parameter_Order_of_db = 5
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        return Parameter_ksee, Parameter_f, Parameter_k, \
                Parameter_Int_Refinement_Level, Parameter_Order_of_db, Str_Name_Wavelet

    def Function_Structure_Properties_SDOF(self, Parameter_ksee, Parameter_f, Parameter_k):
        Parameter_m = Parameter_k / (2 * numpy.pi * Parameter_f)**2
        Parameter_c = Parameter_ksee * 2 * numpy.sqrt(Parameter_m * Parameter_k)
        return Parameter_m, Parameter_c

    def Function_Wavelet_Function_Scaled(self, \
                            Parameter_Order_of_db, Parameter_Int_Refinement_Level, Int_Scale_Time:int):
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Int_Refinement_Level)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Array_Wavelet_Function = Tuple_Function_Return[1]
        Array_Time_Wavelet = Array_Time_Wavelet[::Int_Scale_Time]
        Array_Scaling_Function = Array_Scaling_Function[::Int_Scale_Time]
        Array_Wavelet_Function = Array_Wavelet_Function[::Int_Scale_Time]
        Array_Time_Wavelet = Array_Time_Wavelet / Int_Scale_Time
        return Array_Scaling_Function, Array_Wavelet_Function, Array_Time_Wavelet

    def Function_Wavelet_Function_Derivative_Direct(self, \
            Parameter_Order_of_db, Parameter_Int_Refinement_Level, Parameter_Int_Order_of_Derivative):
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Int_Refinement_Level)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()

        Array_Signal = Array_Scaling_Function.copy()
        Parameter_Scaling_of_Time = 1
        Array_Time = Array_Time_Wavelet * Parameter_Scaling_of_Time
        Value_Delta_T = Value_Delta_T_Wavelet * Parameter_Scaling_of_Time
        
        Array2D_Signal_Derivative_Order = numpy.zeros([Array_Signal.size, Parameter_Int_Order_of_Derivative + 1])
        Array2D_Signal_Derivative_Order[:,0] = Array_Signal 
        for i_Order in numpy.arange(1, Parameter_Int_Order_of_Derivative + 1, 1):
            Array2D_Signal_Derivative_Order[1:-1,i_Order] \
                = (Array2D_Signal_Derivative_Order[2:,i_Order - 1]- Array2D_Signal_Derivative_Order[:-2,i_Order - 1])\
                     / 2 / Value_Delta_T

        Array2D_Wavelet_Function_Derivative = Array2D_Signal_Derivative_Order
        Array_Time_Wavelet = Array_Time
        return Array2D_Wavelet_Function_Derivative, Array_Time_Wavelet

    def Function_Wavelet_Function_Derivative_Connection_Coefficient_Method(self, \
            Parameter_Order_of_db, Parameter_Int_Refinement_Level, Parameter_Int_Order_of_Derivative):
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Int_Refinement_Level)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Array_Signal = Array_Scaling_Function.copy()
        if Parameter_Int_Order_of_Derivative == 0:
            Array2D_Wavelet_Function_Derivative = Array_Signal.reshape(-1,1)
            Array_Time_Wavelet = Array_Time_Wavelet
        else:
            Parameter_Scaling_of_Time = 1
            Array_Time = Array_Time_Wavelet * Parameter_Scaling_of_Time
            Value_Delta_T = Value_Delta_T_Wavelet * Parameter_Scaling_of_Time
            Parameter_Int_Refinement_Level = 6
            Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Int_Refinement_Level)
            Array_Time_Wavelet = Tuple_Function_Return[2]
            Array_Scaling_Function = Tuple_Function_Return[0]
            Array_Wavelet_Function = Tuple_Function_Return[1]
            Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
            Array_Scaling_Function = Array_Scaling_Function * numpy.sqrt(Value_Delta_T_Wavelet / Value_Delta_T)
            Array_Wavelet_Function = Array_Wavelet_Function * numpy.sqrt(Value_Delta_T_Wavelet / Value_Delta_T)
            Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet== 1)[0][0]
            Int_Number_of_Unitlength = numpy.int(Array_Time_Wavelet.max())
            Array_Time_Make_Int_Length, Array_Signal_Make_Int_Length \
                = Class_Signal_Processing.Function_Make_Unit_Length(self, Array_Time, Array_Signal, Int_Unit_Length)
            Int_Number_of_Coefficient \
                = int(Array_Time_Make_Int_Length.size / Int_Unit_Length + Int_Number_of_Unitlength - 1)
            Temp_Array_Signal_Coefficient_Direct = numpy.zeros(Int_Number_of_Coefficient)
            for i_Coefficient in range(Int_Number_of_Coefficient):
                # Coefficients of the signal
                Temp_Array_1 = numpy.zeros((Int_Number_of_Coefficient + Int_Number_of_Unitlength - 1) * Int_Unit_Length + 1)
                Temp_Int_Index_Start = (Int_Number_of_Unitlength - 1) * Int_Unit_Length
                Temp_Int_Index_Ended = Temp_Int_Index_Start + Array_Signal_Make_Int_Length.size
                Temp_Array_1[Temp_Int_Index_Start:Temp_Int_Index_Ended] = Array_Signal_Make_Int_Length
                Temp_Array_2 = numpy.zeros((Int_Number_of_Coefficient + Int_Number_of_Unitlength - 1) * Int_Unit_Length + 1)
                Temp_Int_Index_Start = i_Coefficient * Int_Unit_Length
                Temp_Int_Index_Ended = numpy.min([i_Coefficient * Int_Unit_Length + Array_Scaling_Function.size, Temp_Array_1.size])
                Temp_Array_2[Temp_Int_Index_Start:Temp_Int_Index_Ended] \
                    = Array_Scaling_Function[:Temp_Int_Index_Ended - Temp_Int_Index_Start]
                Temp_Array_Signal_Coefficient_Direct[i_Coefficient] = numpy.sum(Temp_Array_1 * Temp_Array_2) * Value_Delta_T
            Array2D_Connection_Coefficient_Essential_Proper \
                = self.Function_Essential_Connection_Coefficient_Matrix_Improper(Parameter_Order_of_db, 1)
            Int_Half = Int_Number_of_Unitlength - 1
            Array3D_Omega = numpy.zeros([Int_Number_of_Coefficient, \
                                            Int_Number_of_Coefficient, \
                                            Parameter_Int_Order_of_Derivative + 1])
            Array2D_Omega_00 = numpy.zeros([Int_Number_of_Coefficient, Int_Number_of_Coefficient])
            Array2D_Omega_01 = numpy.zeros([Int_Number_of_Coefficient, Int_Number_of_Coefficient])
            for i_Coefficient in range(Int_Number_of_Coefficient):
                for j_Coefficient in range(Int_Number_of_Coefficient):
                    Int_Delta_ij = i_Coefficient - j_Coefficient
                    if numpy.abs(Int_Delta_ij) <= Int_Half:
                        Array2D_Omega_00[i_Coefficient, j_Coefficient] \
                            = Array2D_Connection_Coefficient_Essential_Proper[0,Int_Half + Int_Delta_ij]
                        Array2D_Omega_01[i_Coefficient, j_Coefficient] \
                            = Array2D_Connection_Coefficient_Essential_Proper[1,Int_Half + Int_Delta_ij] \
                                * (Value_Delta_T_Wavelet / Value_Delta_T)
            Array3D_Omega[:,:,0] = Array2D_Omega_00
            Array3D_Omega[:,:,1] = Array2D_Omega_01
            for i_Derivative in numpy.arange(2, Parameter_Int_Order_of_Derivative + 1):
                Array3D_Omega[:,:,i_Derivative] = numpy.dot(Array3D_Omega[:,:,i_Derivative-1], Array2D_Omega_01)
            
            Array2D_Signal_Coefficient_of_Derivatives \
                = numpy.zeros([Int_Number_of_Coefficient, Parameter_Int_Order_of_Derivative + 1])
            Array2D_Signal_Coefficient_of_Derivatives[:,0] = Temp_Array_Signal_Coefficient_Direct
            for i_Derivative in numpy.arange(1, Parameter_Int_Order_of_Derivative + 1):
                Array2D_Signal_Coefficient_of_Derivatives[:,i_Derivative] \
                    = Array3D_Omega[:,:,i_Derivative].dot(Temp_Array_Signal_Coefficient_Direct)

            Temp_Array2D_Signal_Derivative_Reconstruct = \
                numpy.zeros([Array_Time_Make_Int_Length.size + Array_Scaling_Function.size - Int_Unit_Length, \
                                Parameter_Int_Order_of_Derivative + 1])

            for i_Derivative in numpy.arange(0, Parameter_Int_Order_of_Derivative + 1):
                for i_Coefficient in range(Int_Number_of_Coefficient):
                    Temp_Int_Index_Start = i_Coefficient * Int_Unit_Length
                    Temp_Int_Index_Ended = Temp_Int_Index_Start + Array_Scaling_Function.size
                    if Temp_Int_Index_Ended <= Temp_Array2D_Signal_Derivative_Reconstruct.shape[0]:
                        Temp_Array2D_Signal_Derivative_Reconstruct[Temp_Int_Index_Start:Temp_Int_Index_Ended,i_Derivative] \
                            +=Array_Scaling_Function*Array2D_Signal_Coefficient_of_Derivatives[i_Coefficient,i_Derivative]     

            Array2D_Signal_Derivative_Reconstruct \
                = Temp_Array2D_Signal_Derivative_Reconstruct[Int_Unit_Length * (Int_Number_of_Unitlength - 1): \
                        Int_Unit_Length * (Int_Number_of_Unitlength - 1) + Array_Time.size, :]
            Array2D_Wavelet_Function_Derivative = Array2D_Signal_Derivative_Reconstruct
            Array_Time_Wavelet = Array_Time
        return Array2D_Wavelet_Function_Derivative, Array_Time_Wavelet

    def Function_Essential_Connection_Coefficient_Array_Imroper_Direct(self, \
            Parameter_Order_of_db, Parameter_Int_Refinement_Level, \
            Parameter_Int_Order_of_Derivative_1, Parameter_Int_Order_of_Derivative_2):
        Parameter_Int_Order_of_Derivative = numpy.max([Parameter_Int_Order_of_Derivative_1, Parameter_Int_Order_of_Derivative_2])
        Tuple_Function_Return \
            = self\
                .Function_Wavelet_Function_Derivative_Connection_Coefficient_Method(\
                    Parameter_Order_of_db, Parameter_Int_Refinement_Level, Parameter_Int_Order_of_Derivative)
        Array2D_Wavelet_Function_Derivative, Array_Time_Wavelet = Tuple_Function_Return

        Int_Length_Unit_Time = numpy.argwhere(Array_Time_Wavelet == 1)
        Int_Number_of_Unitlength = numpy.int(Parameter_Order_of_db * 2 - 1)
        Int_Length_Scaling_Function = Array_Time_Wavelet.size
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Array_Connection_Coefficient_Improper_Direct \
            = numpy.zeros(Int_Number_of_Unitlength * 2 - 1)
        Array_Mapping_Index_to_Translation \
            = numpy.linspace(-Int_Number_of_Unitlength + 1, Int_Number_of_Unitlength - 1, 2 * Int_Number_of_Unitlength - 1,\
                                dtype = numpy.int)
        for i_Row in range(Array_Connection_Coefficient_Improper_Direct.shape[0]):
            Int_Translation_Row = Array_Mapping_Index_to_Translation[i_Row]
            Temp_Array_Translated_Function_1 = numpy.zeros((Int_Length_Scaling_Function - 1) * 3 + 1)
            Temp_Array_Translated_Function_2 = numpy.zeros((Int_Length_Scaling_Function - 1) * 3 + 1)
            Int_Index_Start = int(Int_Length_Scaling_Function - 1 + Int_Translation_Row * Int_Length_Unit_Time)
            Int_Index_Ended = Int_Index_Start + Int_Length_Scaling_Function
            Temp_Array_Translated_Function_1[Int_Index_Start:Int_Index_Ended] \
                = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_1]
            Int_Index_Start = int(Int_Length_Scaling_Function - 1)
            Int_Index_Ended = Int_Index_Start + Int_Length_Scaling_Function
            Temp_Array_Translated_Function_2[Int_Index_Start:Int_Index_Ended] \
                = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_2]
            Array_Connection_Coefficient_Improper_Direct[i_Row] \
                = numpy.sum(Temp_Array_Translated_Function_1[Int_Index_Start:Int_Index_Ended] \
                            * Temp_Array_Translated_Function_2[Int_Index_Start:Int_Index_Ended]) \
                            * Value_Delta_T_Wavelet
        Array_Scaling_Function_Derivative_Order_1 \
            = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_1]
        Array_Scaling_Function_Derivative_Order_2 \
            = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_2]
        return Array_Connection_Coefficient_Improper_Direct, \
                Array_Scaling_Function_Derivative_Order_1, Array_Scaling_Function_Derivative_Order_2, Array_Time_Wavelet

    def Function_Essential_Connection_Coefficient_Matrix_Proper_Direct_Order_NN(self, \
            Parameter_Order_of_db, Parameter_Int_Refinement_Level, \
            Parameter_Int_Order_of_Derivative_1, Parameter_Int_Order_of_Derivative_2):
        Parameter_Int_Order_of_Derivative = numpy.max([Parameter_Int_Order_of_Derivative_1, Parameter_Int_Order_of_Derivative_2])
        Tuple_Function_Return \
            = self\
                .Function_Wavelet_Function_Derivative_Connection_Coefficient_Method(\
                    Parameter_Order_of_db, Parameter_Int_Refinement_Level, Parameter_Int_Order_of_Derivative)
        Array2D_Wavelet_Function_Derivative, Array_Time_Wavelet = Tuple_Function_Return

        Int_Length_Unit_Time = numpy.argwhere(Array_Time_Wavelet == 1)
        Int_Number_of_Unitlength = numpy.int(Parameter_Order_of_db * 2 - 1)
        Int_Length_Scaling_Function = Array_Time_Wavelet.size
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Array2D_Connection_Coefficient_Proper_Direct \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1])
        Array_Mapping_Index_to_Translation \
            = numpy.linspace(-Int_Number_of_Unitlength + 1, Int_Number_of_Unitlength - 1, 2 * Int_Number_of_Unitlength - 1,\
                                dtype = numpy.int)
        for i_Row in range(Array2D_Connection_Coefficient_Proper_Direct.shape[0]):
            for i_Col in range(Array2D_Connection_Coefficient_Proper_Direct.shape[1]):
                Int_Translation_Row = Array_Mapping_Index_to_Translation[i_Row]
                Int_Translation_Col = Array_Mapping_Index_to_Translation[i_Col]
                Temp_Array_Translated_Function_1 = numpy.zeros((Int_Length_Scaling_Function - 1) * 3 + 1)
                Temp_Array_Translated_Function_2 = numpy.zeros((Int_Length_Scaling_Function - 1) * 3 + 1)
                Int_Index_Start = int(Int_Length_Scaling_Function - 1 + Int_Translation_Row * Int_Length_Unit_Time)
                Int_Index_Ended = Int_Index_Start + Int_Length_Scaling_Function
                Temp_Array_Translated_Function_1[Int_Index_Start:Int_Index_Ended] \
                    = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_1]
                Int_Index_Start = int(Int_Length_Scaling_Function - 1 + Int_Translation_Col * Int_Length_Unit_Time)
                Int_Index_Ended = Int_Index_Start + Int_Length_Scaling_Function
                Temp_Array_Translated_Function_2[Int_Index_Start:Int_Index_Ended] \
                    = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_2]
                Int_Index_Start = int(Int_Length_Scaling_Function - 1)
                Int_Index_Ended = Int_Index_Start + Int_Length_Scaling_Function
                Array2D_Connection_Coefficient_Proper_Direct[i_Row, i_Col] \
                    = numpy.sum(Temp_Array_Translated_Function_1[Int_Index_Start:Int_Index_Ended] \
                                * Temp_Array_Translated_Function_2[Int_Index_Start:Int_Index_Ended]) \
                                * Value_Delta_T_Wavelet
        Array_Scaling_Function_Derivative_Order_1 \
            = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_1]
        Array_Scaling_Function_Derivative_Order_2 \
            = Array2D_Wavelet_Function_Derivative[:, Parameter_Int_Order_of_Derivative_2]
        return Array2D_Connection_Coefficient_Proper_Direct, \
                Array_Scaling_Function_Derivative_Order_1, Array_Scaling_Function_Derivative_Order_2, Array_Time_Wavelet

    def Function_Full_Connection_Coefficient_Proper_0_d2_Direct(self,\
            Array_Signal, Array_Time, \
            Parameter_Int_Refinement_Level, Parameter_Order_of_db,\
            Int_d2):
        
        Object_Signal_Processing = Class_Signal_Processing()
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Int_Refinement_Level)
        Array_Time_Wavelet = Tuple_Function_Return[2]

        Int_Length_Unit_Time = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        Int_Number_of_Unitlength = numpy.int(Parameter_Order_of_db * 2 - 1)

        Tuple_Function_Return \
            = Object_Signal_Processing.Function_Make_Unit_Length(Array_Time, Array_Signal, Int_Length_Unit_Time)
        Array_Time_Make_Int_Length = Tuple_Function_Return[0]

        Int_Maximum_d2 = int(Int_d2)
        Array3D_Essential_Proper_Connection_Coefficient_0_d2_Direct \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1, Int_Maximum_d2 + 1])
        for i_d2 in range(Int_Maximum_d2 + 1):
            Tuple_Fuction_Return = self\
                    .Function_Essential_Connection_Coefficient_Matrix_Proper_Direct(\
                        Parameter_Order_of_db, 14, 0,i_d2)
            Array3D_Essential_Proper_Connection_Coefficient_0_d2_Direct[:,:,i_d2] \
                = Tuple_Fuction_Return[0]
        Array2D_Essential_Connection_Coefficint \
            = Array3D_Essential_Proper_Connection_Coefficient_0_d2_Direct[:,:, Int_d2]
        Array_Essential_Connection_Coefficint_Improper \
            = Array2D_Essential_Connection_Coefficint[Int_Number_of_Unitlength - 1,:]
        Int_Number_of_Coefficient \
            = int(Array_Time_Make_Int_Length.size / Int_Length_Unit_Time + Int_Number_of_Unitlength - 1)
        Array2D_Full_Connection_Coefficient = numpy.zeros([Int_Number_of_Coefficient, Int_Number_of_Coefficient])
        Array2D_Full_Connection_Coefficient[: Int_Number_of_Unitlength * 2 -1, : Int_Number_of_Unitlength * 2 -1] \
            = Array2D_Essential_Connection_Coefficint
        Array2D_Full_Connection_Coefficient[- Int_Number_of_Unitlength * 2 + 1:, -Int_Number_of_Unitlength * 2 + 1:] \
            = Array2D_Essential_Connection_Coefficint
        for i_Row in numpy.arange(Int_Number_of_Unitlength, Int_Number_of_Coefficient - Int_Number_of_Unitlength):
            Int_Index_Col_Start = i_Row - (Int_Number_of_Unitlength - 1)
            Int_Index_Col_Ended = Int_Index_Col_Start + Array_Essential_Connection_Coefficint_Improper.size
            Array2D_Full_Connection_Coefficient[i_Row, Int_Index_Col_Start : Int_Index_Col_Ended] \
                = Array_Essential_Connection_Coefficint_Improper
        return Array2D_Full_Connection_Coefficient

    def Function_Essential_Connection_Coefficient_Array(self, Parameter_Order_of_db, Parameter_n):
        """
        Calculate the improper connection coefficients of wavelet-Galerkin method.
        
        Based on the method described by Beylkin-1992-Wavelet
        
        Parameters
        ----------
        Parameter_Order_of_db
        Parameter_n

        Returns
        ----------
        Array_r, Array_a
        Array_r is the list of connection coefficient
        Array_a is a mid product of the calculation, it is listed as an output for validation
        """
        # Parameter_n = 1

        Int_M = Parameter_Order_of_db
        Int_Length_Wavelet = 2 * Int_M
        Value_CM = (scipy.special.factorial(2 * Int_M - 1) \
                    / scipy.special.factorial(Int_M - 1)  / 4**(Int_M - 1))**2
        Array_a = numpy.zeros(Int_M * 2)
        Array_a[0] = 2
        for i_a in numpy.arange(1, Array_a.size, 2): # Even indeices correspond to zero
            Temp_Int_m = int((i_a + 1) / 2)
            Array_a[i_a] = (-1)**(Temp_Int_m - 1) * Value_CM \
                            / scipy.special.factorial(Int_M - Temp_Int_m) \
                            / scipy.special.factorial(Int_M + Temp_Int_m - 1) \
                            / (2 * Temp_Int_m - 1)

        Array_a_expanded = numpy.zeros(Int_Length_Wavelet * 2 - 1)
        Array_a_expanded[Int_Length_Wavelet:] = Array_a[1:]
        Array_a_expanded[:Int_Length_Wavelet] = numpy.flipud(Array_a[:])
        Array2D_a_matrix = numpy.zeros([Int_Length_Wavelet * 2 - 1, Int_Length_Wavelet * 2 - 1])
        Array2D_a_matrix[Int_Length_Wavelet - 1, :] = Array_a_expanded * 2**(Parameter_n - 1) 
        for i_Row_a_Matrix_Middle in numpy.arange(1, Int_Length_Wavelet, 1):
            Array2D_a_matrix[Int_Length_Wavelet - 1 + i_Row_a_Matrix_Middle, i_Row_a_Matrix_Middle * 2:] \
                = Array_a_expanded[:-i_Row_a_Matrix_Middle * 2] * 2**(Parameter_n - 1) 
            Array2D_a_matrix[Int_Length_Wavelet - 1 - i_Row_a_Matrix_Middle, :-i_Row_a_Matrix_Middle * 2] \
                = Array_a_expanded[i_Row_a_Matrix_Middle * 2:] * 2**(Parameter_n - 1) 
        Array_Sum_Part = (numpy.arange(-(Array_a.size - 1), Array_a.size , 1))**Parameter_n

        Array2D_Left_Matrix = Array2D_a_matrix - numpy.eye(Array_a_expanded.size)
        Array2D_Left_Matrix[Int_Length_Wavelet,:] = Array_Sum_Part
        Array_Right_Side = numpy.zeros(Array_a_expanded.size)
        Array_Right_Side[Int_Length_Wavelet] = (-1)**Parameter_n * scipy.special.factorial(Parameter_n)
        Array_r = numpy.dot(numpy.linalg.inv(Array2D_Left_Matrix), Array_Right_Side.reshape(-1,1)).reshape(-1)
        Array_r = Array_r 
        return Array_r, Array_a

    def Function_Essential_Connection_Coefficient_Matrix_Improper(self, Parameter_Order_of_db, Parameter_Int_Order_of_Derivative):
        """
        Returns
        ----------
        Array2D_Essential_Connection_Coefficient_Improper
        """
        Parameter_n = Parameter_Int_Order_of_Derivative
        Int_Length_Wavelet = Parameter_Order_of_db * 2
        Array2D_Essential_Connection_Coefficient_Improper = numpy.zeros([Parameter_n + 1, Int_Length_Wavelet * 2 - 1])
        for i_Row in range(Parameter_n + 1):
            if i_Row == 0:
                Array2D_Essential_Connection_Coefficient_Improper[0, Int_Length_Wavelet - 1] = 1
            else:
                Tuple_Function_Return \
                    = self.Function_Essential_Connection_Coefficient_Array(Parameter_Order_of_db, i_Row)
                Array2D_Essential_Connection_Coefficient_Improper[i_Row,:] = Tuple_Function_Return[0].reshape(-1)
        return Array2D_Essential_Connection_Coefficient_Improper[:,1:-1]

    def Function_Full_Connection_Coefficient_Matrix_Improper(self, Array2D_Essential_Connection_Coefficient_Improper, \
                            Int_Length_Wavelet_Coefficient, \
                            Parameter_k, Parameter_c, Parameter_m,\
                            Value_Delta_T, Value_Delta_T_Wavelet):
        Int_Half = int((Array2D_Essential_Connection_Coefficient_Improper.shape[1] - 1) / 2)
        Int_Number_of_Order = Array2D_Essential_Connection_Coefficient_Improper.shape[0]
        Array3D_Omega = numpy.zeros((Int_Length_Wavelet_Coefficient, Int_Length_Wavelet_Coefficient, Int_Number_of_Order))
        for i_Coefficient in range(Int_Length_Wavelet_Coefficient):
            for j_Coefficient in range(Int_Length_Wavelet_Coefficient):
                Int_Delta_ij = i_Coefficient - j_Coefficient
                if numpy.abs(Int_Delta_ij) <= Int_Half:
                    for i_Order in range(Array2D_Essential_Connection_Coefficient_Improper.shape[0]):
                        Array3D_Omega[i_Coefficient, j_Coefficient, i_Order] \
                            = Array2D_Essential_Connection_Coefficient_Improper[i_Order,Int_Half + Int_Delta_ij]
                                # * (Value_Delta_T_Wavelet / Value_Delta_T)**i_Order
        Array2D_A_k = Parameter_k * Array3D_Omega[:,:,0] * (Value_Delta_T_Wavelet / Value_Delta_T)**0
        Array2D_A_c = Parameter_c * Array3D_Omega[:,:,1] * (Value_Delta_T_Wavelet / Value_Delta_T)**1
        Array2D_A_m = Parameter_m * Array3D_Omega[:,:,2] * (Value_Delta_T_Wavelet / Value_Delta_T)**2
        #/ Value_Delta_T is not used in the above two lines since it has been considered in coefficients
        Array2D_A = Array2D_A_m + Array2D_A_c + Array2D_A_k
        return Array2D_A, Array3D_Omega

    def Function_Full_Connection_Coefficient_Matrix_Improper_Validation(self, Array2D_Essential_Connection_Coefficient_Improper, \
                            Int_Length_Wavelet_Coefficient, \
                            Parameter_k, Parameter_c, Parameter_m,\
                            Value_Delta_T, Value_Delta_T_Wavelet):
        Int_Half = int((Array2D_Essential_Connection_Coefficient_Improper.shape[1] - 1) / 2)
        Array2D_Omega_00 = numpy.zeros((Int_Length_Wavelet_Coefficient, Int_Length_Wavelet_Coefficient))
        Array2D_Omega_01 = numpy.zeros((Int_Length_Wavelet_Coefficient, Int_Length_Wavelet_Coefficient))
        Array2D_Omega_02 = numpy.zeros((Int_Length_Wavelet_Coefficient, Int_Length_Wavelet_Coefficient))
        for i_Coefficient in range(Int_Length_Wavelet_Coefficient):
            for j_Coefficient in range(Int_Length_Wavelet_Coefficient):
                Int_Delta_ij = i_Coefficient - j_Coefficient
                if numpy.abs(Int_Delta_ij) <= Int_Half:
                    Array2D_Omega_00[i_Coefficient, j_Coefficient] \
                        = Array2D_Essential_Connection_Coefficient_Improper[0,Int_Half + Int_Delta_ij]
                    Array2D_Omega_01[i_Coefficient, j_Coefficient] \
                        = Array2D_Essential_Connection_Coefficient_Improper[1,Int_Half + Int_Delta_ij]
        # Array2D_Omega_01 = Array2D_Omega_01 * (Value_Delta_T_Wavelet / Value_Delta_T)
        Array2D_Omega_02 = numpy.dot(Array2D_Omega_01, Array2D_Omega_01)

        Array2D_A_k = Parameter_k * Array2D_Omega_00
        Array2D_A_c = Parameter_c * Array2D_Omega_01 * (Value_Delta_T_Wavelet / Value_Delta_T)
        Array2D_A_m = Parameter_m * Array2D_Omega_02 * (Value_Delta_T_Wavelet / Value_Delta_T)**2
        #/ Value_Delta_T is not used in the above two lines since it has been considered in coefficients
        Array2D_A = Array2D_A_m + Array2D_A_c + Array2D_A_k
        return Array2D_A, Array2D_Omega_00, Array2D_Omega_01, Array2D_Omega_02


    def Function_Array_Coefficient_of_Initial_Unit_Displacement_Unit_Velocity(self, \
            Value_Delta_T,\
            Parameter_ksee, Parameter_m, Parameter_k,\
            Array_Time_Wavelet, Array_Scaling_Function):
        Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max())
        Int_Length_Scaling_Function = Array_Scaling_Function.size
        Int_Scale_Prediction = 2
        Object_Structural_Response = Class_Structural_Response()
        Temp_Array_Time_Predict \
            = numpy.linspace(0, \
                            Value_Delta_T * Int_Length_Scaling_Function * Int_Scale_Prediction, \
                            Int_Length_Scaling_Function * Int_Scale_Prediction)
        Temp_Array_Response_Unit_Initial_Velocity \
            = Object_Structural_Response.Function_SDOF_Response_Unit_Initial_Velocity(\
                                            Temp_Array_Time_Predict, Parameter_ksee, Parameter_m, Parameter_k)
        Temp_Array_Response_Unit_Initial_Displacement \
            = Object_Structural_Response.Function_SDOF_Response_Unit_Initial_Displacement(\
                                            Temp_Array_Time_Predict, Parameter_ksee, Parameter_m, Parameter_k)

        Temp_Array2D_Phi_Predict \
            = numpy.zeros([Int_Number_of_Unitlength * Int_Scale_Prediction, Temp_Array_Time_Predict.size])
        for i_Coefficient in range(Temp_Array2D_Phi_Predict.shape[0]):
            Temp_Index_Start = i_Coefficient * Int_Unit_Length
            Temp_Index_Ended = i_Coefficient * Int_Unit_Length + Int_Length_Scaling_Function
            if Temp_Index_Ended <= Temp_Array2D_Phi_Predict.shape[1]:
                Temp_Array2D_Phi_Predict[i_Coefficient, Temp_Index_Start: Temp_Index_Ended] = Array_Scaling_Function
            else:
                Temp_Array2D_Phi_Predict[i_Coefficient,Temp_Index_Start:] \
                    = Array_Scaling_Function[:Temp_Array2D_Phi_Predict.shape[1] - Temp_Index_Ended] 
                
        Temp_Array2D_Coef_Initial_Unit_Disp \
            = numpy.dot(Temp_Array2D_Phi_Predict, \
                        Temp_Array_Response_Unit_Initial_Displacement.reshape(-1,1)) \
                        * Value_Delta_T
        Temp_Array2D_Coef_Initial_Unit_Velo \
            = numpy.dot(Temp_Array2D_Phi_Predict, \
                        Temp_Array_Response_Unit_Initial_Velocity.reshape(-1,1)) \
                        * Value_Delta_T
        Temp_Array_Coef_Initial_Unit_Disp \
            = Temp_Array2D_Coef_Initial_Unit_Disp[: Int_Number_of_Unitlength -1 ].reshape(-1)
        Temp_Array_Coef_Initial_Unit_Velo \
            = Temp_Array2D_Coef_Initial_Unit_Velo[: Int_Number_of_Unitlength -1 ].reshape(-1)
        return Temp_Array_Coef_Initial_Unit_Disp, Temp_Array_Coef_Initial_Unit_Velo

    def Function_Array_Coefficient_of_Initial_Unit_Displacement_Unit_Velocity_for_Right_Boundary(self, \
            Value_Delta_T,\
            Parameter_ksee, Parameter_m, Parameter_k,\
            Array_Time_Wavelet, Array_Scaling_Function):
        Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0] # Length of unit in wavelet function
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max()) # Number of unit lengths in a wavelet function
        Int_Length_Scaling_Function = Array_Scaling_Function.size
        Int_Scale_Prediction = 2
        Object_Structural_Response = Class_Structural_Response()
        Temp_Array_Time_Predict \
            = numpy.linspace(0, \
                            Value_Delta_T * Int_Length_Scaling_Function * Int_Scale_Prediction, \
                            Int_Length_Scaling_Function * Int_Scale_Prediction)
        Temp_Array_Response_Unit_Initial_Velocity \
            = Object_Structural_Response.Function_SDOF_Response_Unit_Initial_Velocity(\
                                            Temp_Array_Time_Predict, Parameter_ksee, Parameter_m, Parameter_k)
        Temp_Array_Response_Unit_Initial_Displacement \
            = Object_Structural_Response.Function_SDOF_Response_Unit_Initial_Displacement(\
                                            Temp_Array_Time_Predict, Parameter_ksee, Parameter_m, Parameter_k)

        Temp_Array2D_Phi_Predict \
            = numpy.zeros([Int_Number_of_Unitlength * (Int_Scale_Prediction + 1) - 1, Temp_Array_Time_Predict.size])
        for i_Coefficient in range(Temp_Array2D_Phi_Predict.shape[0]):
            if i_Coefficient < (Int_Number_of_Unitlength - 1):
                Temp_Int_Length_Used \
                    = i_Coefficient * Int_Unit_Length + Int_Unit_Length
                Temp_Array2D_Phi_Predict[i_Coefficient, :Temp_Int_Length_Used] \
                    = Array_Scaling_Function[-Temp_Int_Length_Used:]
            else:
                Temp_Index_Start \
                    = (i_Coefficient - (Int_Number_of_Unitlength - 1)) * Int_Unit_Length
                Temp_Index_Ended \
                    = (i_Coefficient - (Int_Number_of_Unitlength - 1)) * Int_Unit_Length + Int_Length_Scaling_Function
                if Temp_Index_Ended <= Temp_Array2D_Phi_Predict.shape[1]:
                    Temp_Array2D_Phi_Predict[i_Coefficient, Temp_Index_Start: Temp_Index_Ended] = Array_Scaling_Function
                else:
                    Temp_Array2D_Phi_Predict[i_Coefficient,Temp_Index_Start:] \
                        = Array_Scaling_Function[:Temp_Array2D_Phi_Predict.shape[1] - Temp_Index_Ended] 
                
        Temp_Array2D_Coef_Initial_Unit_Disp \
            = numpy.dot(Temp_Array2D_Phi_Predict, Temp_Array_Response_Unit_Initial_Displacement.reshape(-1,1)) \
                * Value_Delta_T
        Temp_Array2D_Coef_Initial_Unit_Velo \
            = numpy.dot(Temp_Array2D_Phi_Predict, Temp_Array_Response_Unit_Initial_Velocity.reshape(-1,1)) \
                * Value_Delta_T
        Temp_Array_Coef_Initial_Unit_Disp_for_Right_Boundary \
            = Temp_Array2D_Coef_Initial_Unit_Disp[: 2 * (Int_Number_of_Unitlength -1)].reshape(-1)
        Temp_Array_Coef_Initial_Unit_Velo_for_Right_Boundary \
            = Temp_Array2D_Coef_Initial_Unit_Velo[: 2 * (Int_Number_of_Unitlength -1)].reshape(-1)
        return Temp_Array_Coef_Initial_Unit_Disp_for_Right_Boundary, \
                Temp_Array_Coef_Initial_Unit_Velo_for_Right_Boundary

    def Function_Levenberg_Marquardt_Method_Linear(self, Array2D_A_LM, Array_Y_LM):
        
        def Sub_Function_Jacobian_Matrix(Array_X, Array2D_A):
            Array2D_Jacobian_Matrix = Array2D_A
            return Array2D_Jacobian_Matrix

        def Sub_Function_Function_Value(Array_X, Array2D_A):
            Array_Y_Hat = Array2D_A.dot(Array_X)
            return Array_Y_Hat

        def Sub_Function_Array_Delta_X(Value_lambda, Array_X, Array2D_A, Array_Y, Array2D_Weight_Matrix):
            Array2D_Jacobian_Matrix = Sub_Function_Jacobian_Matrix(Array_X, Array2D_A)
            Array_Y_hat = Sub_Function_Function_Value(Array_X, Array2D_A)
            Array2D_Left = Array2D_Jacobian_Matrix.T.dot(Array2D_Weight_Matrix).dot(Array2D_Jacobian_Matrix)\
                            + Value_lambda * numpy.eye(Array_X.size)
            Array2D_Right = Array2D_Jacobian_Matrix.T.dot(Array2D_Weight_Matrix).dot(Array_Y - Array_Y_hat)
            Array_Delta_X = numpy.linalg.inv(Array2D_Left).dot(Array2D_Right)
            return Array_Delta_X

        def Sub_Function_rho_Value(Array_X_0, Array_Delta_X, Array_Y, Array2D_A):
            Array_X_1 = Array_X_0 + Array_Delta_X
            Array_Y_hat_0 = Sub_Function_Function_Value(Array_X_0, Array2D_A)
            Array_Y_hat_1 = Sub_Function_Function_Value(Array_X_1, Array2D_A)
            Temp_Value_kai_square_0 = \
                Array_Y.reshape(1,-1).dot(Array2D_Weight_Matrix).dot(Array_Y.reshape(-1,1))\
                - 2 * Array_Y.reshape(1,-1).dot(Array2D_Weight_Matrix).dot(Array_Y_hat_0.reshape(-1,1))\
                + Array_Y_hat_0.reshape(1,-1).dot(Array2D_Weight_Matrix).dot(Array_Y_hat_0.reshape(-1,1))

            Temp_Value_kai_square_1 = \
                Array_Y.reshape(1,-1).dot(Array2D_Weight_Matrix).dot(Array_Y.reshape(-1,1))\
                - 2 * Array_Y.reshape(1,-1).dot(Array2D_Weight_Matrix).dot(Array_Y_hat_1.reshape(-1,1))\
                + Array_Y_hat_1.reshape(1,-1).dot(Array2D_Weight_Matrix).dot(Array_Y_hat_1.reshape(-1,1))

            Temp_Value_numerator = Temp_Value_kai_square_0 - Temp_Value_kai_square_1

            Temp_Array_Part_1 = Array_Y - Array_Y_hat_0
            Temp_Value_Denominator_Part_1 = numpy.sum(Temp_Array_Part_1**2)

            Array2D_Jacobian_Matrix = Sub_Function_Jacobian_Matrix(Array_X_0, Array2D_A)
            Temp_Array_Part_2 = Array_Y - Array_Y_hat_0 - Array2D_Jacobian_Matrix.dot(Array_Delta_X)
            Temp_Value_Denominator_Part_2 = numpy.sum(Temp_Array_Part_2**2)

            Temp_Value_denominator = Temp_Value_Denominator_Part_1 - Temp_Value_Denominator_Part_2

            Value_rho = Temp_Value_numerator / Temp_Value_denominator
            return Value_rho

        Array2D_Weight_Matrix = numpy.eye(Array2D_A_LM.shape[0])
        Parameter_epsilon = 0.0001
        Array_X_LM = numpy.zeros(Array2D_A_LM.shape[1])
        Value_lambda = 100

        Array_X_LM = numpy.random.rand(Array2D_A_LM.shape[1])
        Value_lambda = 100
        Value_v = 2
        for i_Iteration in range(1000):
            Array_Delta_X_LM =  Sub_Function_Array_Delta_X(Value_lambda, Array_X_LM, Array2D_A_LM, Array_Y_LM, Array2D_Weight_Matrix)
            Value_rho =  Sub_Function_rho_Value(Array_X_LM, Array_Delta_X_LM, Array_Y_LM, Array2D_A_LM)
            if Value_rho >=0 and Value_rho <= Parameter_epsilon:
                # Convergent or saddle point
                break

            elif Value_rho > Parameter_epsilon:
                Array_X_LM = Array_X_LM + Array_Delta_X_LM
                Value_lambda = Value_lambda * numpy.max([1/3, 1 - (2 * Value_rho - 1)**3])
                Value_v = 2
            else:
                Value_lambda= Value_lambda * Value_v
                Value_v = Value_v * 2
        return Array_X_LM
        
    def Function_Moment_of_Scaling_Function(self, Parameter_Order_of_db, Int_Max_Order_of_Moment, \
                                                    Int_Min_Translation, Int_Max_Translation):
        """
        This function is based on (Bu, 2018, Page 6) with modification of summation intervals
        ----------
        This function is commented out since the new boundary codition proposed based on the free-vibration is used.
        """
        # Based on Amaratunga-1997-Wavelet-(Page24)
        def Sub_Function_Number_of_Coice(Int_a, Int_b):
            """
            Calculation of C_a^b
            a!/b!/(a-b)!
            """
            Int_Numbero_of_Choice = scipy.special.factorial(Int_a) / scipy.special.factorial(Int_b) / scipy.special.factorial(Int_a - Int_b)
            return Int_Numbero_of_Choice
        
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Array_a = numpy.sqrt(2) * numpy.array(Wavelet_DBN.filter_bank[2]) # Filter coefficients
        Array2D_u = numpy.zeros([Int_Max_Translation - Int_Min_Translation + 1, Int_Max_Order_of_Moment + 1])
        Array_u_Zero = numpy.zeros(Int_Max_Order_of_Moment + 1)
        Array_u_Zero[0] = 1
        for i_Col in numpy.arange(1, Int_Max_Order_of_Moment + 1):
            Temp_Sum = 0
            for i_a in range(Parameter_Order_of_db * 2):
                for i_l in numpy.arange(1, i_Col + 1):   
                    Temp_Value_Part_1 = Sub_Function_Number_of_Coice(i_Col, i_l)
                    Temp_Value_Part_2 = Array_a[i_a]
                    Temp_Value_Part_3 = i_a**i_l
                    Temp_Value_Part_4 = Array_u_Zero[i_Col - i_l]
                    Temp_Sum += Temp_Value_Part_1 * Temp_Value_Part_2 * Temp_Value_Part_3 * Temp_Value_Part_4
            Temp_Sum = Temp_Sum / 2 / (2**i_Col - 1)
            Array_u_Zero[i_Col] = Temp_Sum
        
        Temp_Array_Mapping_Row_to_Translation \
            = numpy.linspace(Int_Min_Translation, Int_Max_Translation, Int_Max_Translation - Int_Min_Translation + 1,\
                dtype = numpy.int)
        for i_Row in range(Temp_Array_Mapping_Row_to_Translation.size):
            Temp_Int_Translation = Temp_Array_Mapping_Row_to_Translation[i_Row]
            if Temp_Int_Translation == 0:
                Array2D_u[i_Row,:] = Array_u_Zero
            else:
                Array2D_u[i_Row, 0] = Array_u_Zero[0]
                for i_Col in numpy.arange(1, Int_Max_Order_of_Moment + 1):
                    Temp_Sum = 0
                    for i_i in numpy.arange(0, i_Col + 1):
                        Temp_Value_Part_1 = Sub_Function_Number_of_Coice(i_Col, i_i)
                        Temp_Value_Part_2 = Temp_Int_Translation**(i_Col - i_i)
                        Temp_Value_Part_3 = Array_u_Zero[i_i]
                        Temp_Sum += Temp_Value_Part_1 * Temp_Value_Part_2 * Temp_Value_Part_3
                    Array2D_u[i_Row, i_Col] = Temp_Sum
        return Array2D_u.T, Temp_Array_Mapping_Row_to_Translation

    def Function_Moment_of_Scaling_Function_Validation_Le_2015(self, Parameter_Order_of_db, Int_Max_Order_of_Moment, \
                                                    Int_Min_Translation, Int_Max_Translation):
        """
        This is a validation function for the calculation od moments, the altorithm is based on (Le, 2015, Page 9/31)
        """
        def Sub_Function_Selection_Case(Int_Selection, Int_Total):
            Int_Numbero_of_Choice = scipy.special.factorial(Int_Total) \
                                        / scipy.special.factorial(Int_Selection) \
                                        / scipy.special.factorial(Int_Total - Int_Selection)
            return Int_Numbero_of_Choice
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Array_a = numpy.sqrt(2) * numpy.array(Wavelet_DBN.filter_bank[2]) # Filter coefficients
        Array2D_u = numpy.zeros([Int_Max_Translation - Int_Min_Translation + 1, Int_Max_Order_of_Moment + 1])


        Array2D_Moment = numpy.zeros([Int_Max_Order_of_Moment + 1, Int_Max_Translation - Int_Min_Translation + 1])
        Temp_Array_Mapping_Col_to_Translation \
            = numpy.linspace(Int_Min_Translation, Int_Max_Translation, Int_Max_Translation - Int_Min_Translation + 1,\
                            dtype = numpy.int)
        Array_Moment_Zero_Translation = numpy.zeros(Int_Max_Order_of_Moment + 1)
        Array_Moment_Zero_Translation[0] = 1
        for Order_of_Moment in numpy.arange(1, Int_Max_Order_of_Moment + 1):
            Temp_Value_Sum = 0
            for i_q in range(Order_of_Moment):
                Temp_Value_Part_1 = Sub_Function_Selection_Case(i_q, Order_of_Moment)
                Temp_Value_Part_2 = Array_Moment_Zero_Translation[i_q]
                for i_a in range(Array_a.size):
                    Temp_Value_Part_3_1 = Array_a[i_a]
                    Temp_Value_Part_3_2 = i_a**(Order_of_Moment - i_q)
                    Temp_Value_Sum += Temp_Value_Part_1 * Temp_Value_Part_2 * Temp_Value_Part_3_1 * Temp_Value_Part_3_2
            Array_Moment_Zero_Translation[Order_of_Moment] = Temp_Value_Sum / 2 / (2**Order_of_Moment - 1)

        for Order_of_Moment in numpy.arange(1, Int_Max_Order_of_Moment + 1):
            for i_Col in range(Array2D_Moment.shape[1]):
                Temp_Int_Translation = Temp_Array_Mapping_Col_to_Translation[i_Col]
                if Temp_Int_Translation == 0:
                    Array2D_Moment[:, i_Col] = Array_Moment_Zero_Translation
                else:
                    Array2D_Moment[0, i_Col] = 1
                    Temp_Value_Sum = 0
                    for i_p in numpy.arange(0, Order_of_Moment + 1):
                        if i_p == 0:
                            Temp_Value_Sum = Temp_Int_Translation**Order_of_Moment * 1
                        else:
                            Temp_Value_Part_1 = Sub_Function_Selection_Case(i_p, Order_of_Moment)
                            Temp_Value_Part_2 = Temp_Int_Translation**(Order_of_Moment - i_p)
                            Temp_Value_Part_3 = 1 / 2 / (2**i_p - 1)
                            for i_q in range(i_p):
                                Temp_Value_Part_4_1 = Sub_Function_Selection_Case(i_q, i_p)
                                Temp_Value_Part_4_2 = Array_Moment_Zero_Translation[i_q]
                                for i_a in range(Array_a.size):
                                    Temp_Value_Part_5_1 = Array_a[i_a]
                                    Temp_Value_Part_5_2 = i_a**(i_p - i_q)
                                    Temp_Value_Sum += Temp_Value_Part_1 * Temp_Value_Part_2 * Temp_Value_Part_3 \
                                                        * Temp_Value_Part_4_1 * Temp_Value_Part_4_2 \
                                                        * Temp_Value_Part_5_1 * Temp_Value_Part_5_2
                    Array2D_Moment[Order_of_Moment, i_Col] = Temp_Value_Sum 
        return Array2D_Moment, Temp_Array_Mapping_Col_to_Translation

    def Function_Moment_of_Scaling_Function_Validation_Direct_Calculation(self, Parameter_Order_of_db, \
            Int_Max_Order_of_Moment, Int_Min_Translation, Int_Max_Translation):

        """
        This function calculates the moments of scaling function based on directly integration following the definition
        of moment.
        """
        ## Parameter setting
        Parameter_Calculation_Refinement_Scale = 12 
        ## Scaing function and corresponding values
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Calculation_Refinement_Scale)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        ## Calculation by integration
        Array2D_Moment = numpy.zeros([Int_Max_Order_of_Moment + 1, Int_Max_Translation - Int_Min_Translation + 1])
        Array_Mapping_Col_to_Translation \
            = numpy.linspace(Int_Min_Translation, Int_Max_Translation, Int_Max_Translation - Int_Min_Translation + 1,
                                dtype = numpy.int)
        for Order_of_Moment in range(Array2D_Moment.shape[0]):
            for i_Col in range(Array2D_Moment.shape[1]):
                Temp_Array_1 = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  (Int_Max_Translation - Int_Min_Translation))
                Int_Index_Start = i_Col * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_1[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function
                Temp_Array_2 \
                    = numpy.linspace(Value_Delta_T_Wavelet * Int_Unit_Length * Int_Min_Translation, \
                                Value_Delta_T_Wavelet * \
                                (Array_Time_Wavelet.size - 1 + Int_Unit_Length * Int_Max_Translation), \
                                Temp_Array_1.size)
                Array2D_Moment[Order_of_Moment, i_Col] = numpy.sum(Temp_Array_1 * Temp_Array_2**Order_of_Moment) * Value_Delta_T_Wavelet
        return Array2D_Moment, Array_Mapping_Col_to_Translation

    def Function_Essential_Connection_Coefficient_Matrix_Order_00(self, Parameter_Order_of_db):
        """
        This function calculates connection coefficient based on the method 
        in Romine and Peyton (1997)
        Note:
        Since this is the first order derivative, only the first part of the 
        equation is used.
        """
        ## Scaing function and corresponding values
        Parameter_Order_of_db = int(Parameter_Order_of_db)
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun()
        Array_Time_Wavelet = Tuple_Function_Return[2]
        # Array_Scaling_Function = Tuple_Function_Return[0]
        # Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        # Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max())
        # Known connection coefficients
        Array3D_Connection_Coefficient_Imroper \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1, 2])
        Array_Mapping_Row_to_Index \
            = numpy.linspace( - (Int_Number_of_Unitlength - 1), (Int_Number_of_Unitlength - 1), \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        Temp_Array2D_Connection_Coefficient = self.Function_Essential_Connection_Coefficient_Matrix_Improper(Parameter_Order_of_db, 2)
        Array2D_Connection_Coefficient = numpy.zeros([4, 2 * Int_Number_of_Unitlength - 1])
        Array2D_Connection_Coefficient[0:2,:] = Temp_Array2D_Connection_Coefficient[:2,:]
        Array2D_Connection_Coefficient[2,:] = Array2D_Connection_Coefficient[1,:].dot(Array2D_Connection_Coefficient[1,:])
        Array2D_Connection_Coefficient[3,:] = Array2D_Connection_Coefficient[1,:].dot(Array2D_Connection_Coefficient[2,:])
        for i_Level in range(2):
            for i_Row in range(Array3D_Connection_Coefficient_Imroper.shape[0]):
                for i_Col in range(Array3D_Connection_Coefficient_Imroper.shape[1]):
                    if Array_Mapping_Row_to_Index[i_Row] * Array_Mapping_Row_to_Index[i_Col] <= 0:
                        # This is the improper connection coefficient part
                        Temp_Int_Difference \
                            = Array_Mapping_Row_to_Index[i_Row] - Array_Mapping_Row_to_Index[i_Col]
                        Temp_Int_Index \
                            = numpy.argwhere(Array_Mapping_Row_to_Index == Temp_Int_Difference)
                        if Temp_Int_Index.size != 0:
                            Array3D_Connection_Coefficient_Imroper[i_Row, i_Col, i_Level] \
                                = Array2D_Connection_Coefficient[i_Level, Temp_Int_Index]
                    elif Array_Mapping_Row_to_Index[i_Row] > 0 and Array_Mapping_Row_to_Index[i_Col] > 0:
                        """
                        Those coefficients were directly set to the improper connection coefficient based on the relationsip
                        that the \\Gamma_1 + \\Gamma_2 = \\Lambda and the fact that the outside parts are zero.
                        """
                        Temp_Int_Difference \
                            = Array_Mapping_Row_to_Index[i_Row] - Array_Mapping_Row_to_Index[i_Col]
                        Temp_Int_Index \
                            = numpy.argwhere(Array_Mapping_Row_to_Index == Temp_Int_Difference)
                        Array3D_Connection_Coefficient_Imroper[i_Row, i_Col, i_Level] \
                            = Array2D_Connection_Coefficient[i_Level, Temp_Int_Index]

        ## Calculation of weighting coefficients
        Array_a = numpy.sqrt(2) * numpy.array(Wavelet_DBN.filter_bank[2]) # Filter coefficients
        Array2D_Filter_Coefficient = Array_a.reshape(-1,1).dot(Array_a.reshape(1,-1))
        Array2D_Weight_Matrix_Unknown \
            = numpy.zeros([(Int_Number_of_Unitlength - 1)**2, (Int_Number_of_Unitlength - 1)**2])
        Array2D_Weight_Matrix_Known \
            = numpy.zeros([(Int_Number_of_Unitlength - 1)**2, \
                (2 * Int_Number_of_Unitlength - 1)**2 - (Int_Number_of_Unitlength - 1)**2])
        for i_Row in range(Int_Number_of_Unitlength - 1):
            for i_Col in range(Int_Number_of_Unitlength - 1):
                ## Calculation of start and index on the whole matrix
                i_Unknown = i_Row * (Int_Number_of_Unitlength - 1) + i_Col
                Temp_Int_Index_Row_Start = numpy.argwhere(Array_Mapping_Row_to_Index == Array_Mapping_Row_to_Index[i_Row] * 2)
                if Temp_Int_Index_Row_Start.size != 0:
                    Int_Index_Row_Start = Temp_Int_Index_Row_Start[0][0] 
                else:
                    Int_Index_Row_Start = 0
                Temp_Int_Index_Col_Start = numpy.argwhere(Array_Mapping_Row_to_Index == Array_Mapping_Row_to_Index[i_Col] * 2)
                if Temp_Int_Index_Col_Start.size != 0:
                    Int_Index_Col_Start = Temp_Int_Index_Col_Start[0][0]
                else:
                    Int_Index_Col_Start = 0
                Temp_Int_Index_Row_Ended \
                    = numpy.argwhere(Array_Mapping_Row_to_Index == \
                        Array_Mapping_Row_to_Index[i_Row] * 2 + Int_Number_of_Unitlength + 1)
                if Temp_Int_Index_Row_Ended.size != 0:
                    Int_Index_Row_Ended = Temp_Int_Index_Row_Ended[0][0]
                else:
                    Int_Index_Row_Ended = Array_Mapping_Row_to_Index.size
                Temp_Int_Index_Col_Ended \
                    = numpy.argwhere(Array_Mapping_Row_to_Index == \
                        Array_Mapping_Row_to_Index[i_Col] * 2 + Int_Number_of_Unitlength + 1)
                if Temp_Int_Index_Col_Ended.size != 0:
                    Int_Index_Col_Ended = Temp_Int_Index_Col_Ended[0][0]
                else:
                    Int_Index_Col_Ended = Array_Mapping_Row_to_Index.size
                ## Calculate the selection matrix and assign corresponding values
                Array2D_Selection = numpy.zeros([2 * Int_Number_of_Unitlength - 1,2 * Int_Number_of_Unitlength - 1])
                Array2D_Selection[Int_Index_Row_Start:Int_Index_Row_Ended, Int_Index_Col_Start: Int_Index_Col_Ended] = 1
                Array2D_A = numpy.zeros([2 * Int_Number_of_Unitlength - 1,2 * Int_Number_of_Unitlength - 1])
                if Int_Index_Row_Start == 0:
                    Temp_Index_Array_A_Row_Start = Array_a.size - Int_Index_Row_Ended
                    Temp_Index_Array_A_Row_Ended = Array_a.size
                if Int_Index_Col_Start == 0:
                    Temp_Index_Array_A_Col_Start = Array_a.size - Int_Index_Col_Ended
                    Temp_Index_Array_A_Col_Ended = Array_a.size
                if Int_Index_Row_Ended == Array_Mapping_Row_to_Index.size:
                    Temp_Index_Array_A_Row_Start = 0
                    Temp_Index_Array_A_Row_Ended = Int_Index_Row_Ended - Int_Index_Row_Start
                if Int_Index_Col_Ended == Array_Mapping_Row_to_Index.size:
                    Temp_Index_Array_A_Col_Start = 0
                    Temp_Index_Array_A_Col_Ended = Int_Index_Col_Ended - Int_Index_Col_Start
                Array2D_A[Int_Index_Row_Start:Int_Index_Row_Ended, Int_Index_Col_Start: Int_Index_Col_Ended] \
                        = Array2D_Filter_Coefficient[Temp_Index_Array_A_Row_Start: Temp_Index_Array_A_Row_Ended, \
                                                    Temp_Index_Array_A_Col_Start: Temp_Index_Array_A_Col_Ended]
                ## Reshape the weight matrix to weight arrray for both unknowns and knowns
                Temp_Array2D_Weight_Matrix = Array2D_Selection * Array2D_A
                Temp_Array_Weight_Matrix_Unknown \
                    = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,:Int_Number_of_Unitlength - 1].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known_1 \
                    = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1:].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known_2 \
                    = Temp_Array2D_Weight_Matrix[Int_Number_of_Unitlength - 1:,].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known \
                    = numpy.append(Temp_Array_Weight_Matrix_Known_1, Temp_Array_Weight_Matrix_Known_2)
                Array2D_Weight_Matrix_Unknown[i_Unknown,:] = Temp_Array_Weight_Matrix_Unknown
                Array2D_Weight_Matrix_Known[i_Unknown,:] = Temp_Array_Weight_Matrix_Known
                # print(numpy.sum(Temp_Array2D_Weight_Matrix) - \
                #             numpy.sum(numpy.append(Temp_Array_Weight_Matrix_Unknown, Temp_Array_Weight_Matrix_Known)))
        Temp_Array2D_Connection_Coefficient \
            = Array3D_Connection_Coefficient_Imroper[:,:,0]
        Temp_Array2D_Connection_Coefficient_Known_1 \
            = Temp_Array2D_Connection_Coefficient[:Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1:].\
                reshape(-1, order = 'C')
        Temp_Array2D_Connection_Coefficient_Known_2 \
            = Temp_Array2D_Connection_Coefficient[Int_Number_of_Unitlength - 1:,].\
                reshape(-1, order = 'C')
        Temp_Array2D_Connection_Coefficient_Known \
            = numpy.append(Temp_Array2D_Connection_Coefficient_Known_1, Temp_Array2D_Connection_Coefficient_Known_2)
        
        Array2D_Identity = numpy.eye((Int_Number_of_Unitlength - 1)**2)
        Temp_Array2D_Left = Array2D_Identity - Array2D_Weight_Matrix_Unknown * 2**(-1)
        Temp_Array2D_Left_inv = numpy.linalg.inv(Temp_Array2D_Left)
        Temp_Array_Right = Array2D_Weight_Matrix_Known.dot(Temp_Array2D_Connection_Coefficient_Known) * 2**(-1)

        Temp_Array_Unknown \
            = Temp_Array2D_Left_inv.dot(Temp_Array_Right)
        Temp_Array2D_Unknown = Temp_Array_Unknown.reshape((Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1))
        Temp_Array2D_Connection_Coefficient[:Int_Number_of_Unitlength - 1, :Int_Number_of_Unitlength - 1] \
            = Temp_Array2D_Unknown
        Array2D_Connection_Coefficient_Order_00 = Temp_Array2D_Connection_Coefficient
        Array_Mapping_Col_to_Translation \
            = numpy.linspace(- (Int_Number_of_Unitlength - 1), Int_Number_of_Unitlength - 1, \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        return Array2D_Connection_Coefficient_Order_00, Array_Mapping_Col_to_Translation

    def Function_Essential_Connection_Coefficient_Matrix_Proper_Order_0N(self, Parameter_Order_of_db, \
            Parameter_Int_Order_of_Derivative):
        """
        This function calculates connection coefficient based on the method 
        in Romine and Peyton (1997)
        """
        ## Scaing function and corresponding values
        Parameter_Order_of_db = int(Parameter_Order_of_db)
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun()
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max())
        # Known connection coefficients
        Array2D_Connection_Coefficient_Proper_Order_0N \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1])
        Array_Mapping_Row_to_Index \
            = numpy.linspace( - (Int_Number_of_Unitlength - 1), (Int_Number_of_Unitlength - 1), \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        Array2D_Essential_Connection_Coefficient_Improper \
            = self.Function_Essential_Connection_Coefficient_Matrix_Improper(\
                    Parameter_Order_of_db, Parameter_Int_Order_of_Derivative)

        for i_Row in range(Array2D_Connection_Coefficient_Proper_Order_0N.shape[0]):
            for i_Col in range(Array2D_Connection_Coefficient_Proper_Order_0N.shape[1]):
                if Array_Mapping_Row_to_Index[i_Row] * Array_Mapping_Row_to_Index[i_Col] <= 0:
                    """
                    # This is the improper connection coefficient part:
                    # lower lef t and upper right
                    """
                    Temp_Int_Difference \
                        = Array_Mapping_Row_to_Index[i_Row] - Array_Mapping_Row_to_Index[i_Col]
                    Temp_Array_Index \
                        = numpy.argwhere(Array_Mapping_Row_to_Index == Temp_Int_Difference)
                    if Temp_Array_Index.size != 0:
                        Temp_Int_Index = Temp_Array_Index[0][0]
                        Array2D_Connection_Coefficient_Proper_Order_0N\
                            [i_Row, i_Col] \
                                = Array2D_Essential_Connection_Coefficient_Improper\
                                    [Parameter_Int_Order_of_Derivative, Temp_Int_Index]
                elif Array_Mapping_Row_to_Index[i_Row] > 0 and Array_Mapping_Row_to_Index[i_Col] > 0:
                    """
                    Those coefficients were directly set to the improper 
                    connection coefficient based on the relationsip
                    that the \\Gamma_1 + \\Gamma_2 = \\Lambda and the fact 
                    that the outside parts are zero.
                    """
                    Temp_Int_Difference \
                        = Array_Mapping_Row_to_Index[i_Row] - Array_Mapping_Row_to_Index[i_Col]
                    Temp_Array_Index \
                        = numpy.argwhere(Array_Mapping_Row_to_Index == Temp_Int_Difference)
                    Temp_Int_Index = Temp_Array_Index[0][0]
                    Array2D_Connection_Coefficient_Proper_Order_0N[i_Row, i_Col] \
                        = Array2D_Essential_Connection_Coefficient_Improper\
                            [Parameter_Int_Order_of_Derivative, Temp_Int_Index]

        ## Calculation of weighting coefficients
        Array_a = numpy.sqrt(2) * numpy.array(Wavelet_DBN.filter_bank[2]) # Filter coefficients
        Array2D_Filter_Coefficient = Array_a.reshape(-1,1).dot(Array_a.reshape(1,-1))
        Array2D_Weight_Matrix_Unknown_Part_1 \
            = numpy.zeros([(Int_Number_of_Unitlength - 1)**2, (Int_Number_of_Unitlength - 1)**2])
        Array2D_Weight_Matrix_Known_Part_1 \
            = numpy.zeros([(Int_Number_of_Unitlength - 1)**2, \
                (2 * Int_Number_of_Unitlength - 1)**2 - (Int_Number_of_Unitlength - 1)**2])
        for i_Row in range(Int_Number_of_Unitlength - 1):
            for i_Col in range(Int_Number_of_Unitlength - 1):
                ## Calculation of start and index on the whole matrix
                i_Unknown = i_Row * (Int_Number_of_Unitlength - 1) + i_Col
                Temp_Int_Index_Row_Start \
                    = numpy.argwhere(Array_Mapping_Row_to_Index == Array_Mapping_Row_to_Index[i_Row] * 2)
                if Temp_Int_Index_Row_Start.size != 0:
                    Int_Index_Row_Start = Temp_Int_Index_Row_Start[0][0] 
                else:
                    Int_Index_Row_Start = 0
                Temp_Int_Index_Col_Start \
                    = numpy.argwhere(Array_Mapping_Row_to_Index == Array_Mapping_Row_to_Index[i_Col] * 2)
                if Temp_Int_Index_Col_Start.size != 0:
                    Int_Index_Col_Start = Temp_Int_Index_Col_Start[0][0]
                else:
                    Int_Index_Col_Start = 0
                Temp_Int_Index_Row_Ended \
                    = numpy.argwhere(Array_Mapping_Row_to_Index == \
                        Array_Mapping_Row_to_Index[i_Row] * 2 + Int_Number_of_Unitlength + 1)
                if Temp_Int_Index_Row_Ended.size != 0:
                    Int_Index_Row_Ended = Temp_Int_Index_Row_Ended[0][0]
                else:
                    Int_Index_Row_Ended = Array_Mapping_Row_to_Index.size
                Temp_Int_Index_Col_Ended \
                    = numpy.argwhere(Array_Mapping_Row_to_Index == \
                        Array_Mapping_Row_to_Index[i_Col] * 2 + Int_Number_of_Unitlength + 1)
                if Temp_Int_Index_Col_Ended.size != 0:
                    Int_Index_Col_Ended = Temp_Int_Index_Col_Ended[0][0]
                else:
                    Int_Index_Col_Ended = Array_Mapping_Row_to_Index.size
                ## Calculate the selection matrix and assign corresponding values
                Array2D_Selection = numpy.zeros([2 * Int_Number_of_Unitlength - 1,2 * Int_Number_of_Unitlength - 1])
                Array2D_Selection[Int_Index_Row_Start:Int_Index_Row_Ended, Int_Index_Col_Start: Int_Index_Col_Ended] = 1
                Array2D_A = numpy.zeros([2 * Int_Number_of_Unitlength - 1,2 * Int_Number_of_Unitlength - 1])
                if Int_Index_Row_Start == 0:
                    Temp_Index_Array_A_Row_Start = Array_a.size - Int_Index_Row_Ended
                    Temp_Index_Array_A_Row_Ended = Array_a.size
                if Int_Index_Col_Start == 0:
                    Temp_Index_Array_A_Col_Start = Array_a.size - Int_Index_Col_Ended
                    Temp_Index_Array_A_Col_Ended = Array_a.size
                if Int_Index_Row_Ended == Array_Mapping_Row_to_Index.size:
                    Temp_Index_Array_A_Row_Start = 0
                    Temp_Index_Array_A_Row_Ended = Int_Index_Row_Ended - Int_Index_Row_Start
                if Int_Index_Col_Ended == Array_Mapping_Row_to_Index.size:
                    Temp_Index_Array_A_Col_Start = 0
                    Temp_Index_Array_A_Col_Ended = Int_Index_Col_Ended - Int_Index_Col_Start
                Array2D_A[Int_Index_Row_Start:Int_Index_Row_Ended, Int_Index_Col_Start: Int_Index_Col_Ended] \
                        = Array2D_Filter_Coefficient[Temp_Index_Array_A_Row_Start: Temp_Index_Array_A_Row_Ended, \
                                                    Temp_Index_Array_A_Col_Start: Temp_Index_Array_A_Col_Ended]
                ## Reshape the weight matrix to weight arrray for both unknowns and knowns
                Temp_Array2D_Weight_Matrix = Array2D_Selection * Array2D_A
                Temp_Array_Weight_Matrix_Unknown \
                    = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,:Int_Number_of_Unitlength - 1].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known_1 \
                    = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1:].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known_2 \
                    = Temp_Array2D_Weight_Matrix[Int_Number_of_Unitlength - 1:,].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known \
                    = numpy.append(Temp_Array_Weight_Matrix_Known_1, Temp_Array_Weight_Matrix_Known_2)
                Array2D_Weight_Matrix_Unknown_Part_1[i_Unknown,:] = Temp_Array_Weight_Matrix_Unknown
                Array2D_Weight_Matrix_Known_Part_1[i_Unknown,:] = Temp_Array_Weight_Matrix_Known
        # Moment coefficients
        Tuple_Function_Return \
            = self.Function_Moment_of_Scaling_Function(Parameter_Order_of_db, Parameter_Int_Order_of_Derivative, \
                                                        - Int_Number_of_Unitlength + 1, Int_Number_of_Unitlength - 1)
        Array2D_Moment_Coefficient = Tuple_Function_Return[0]
        # Equation set - Part 2:
        Array2D_Weight_Matrix_Unknown_Part_2 \
            = numpy.zeros([Parameter_Int_Order_of_Derivative - 1, (Int_Number_of_Unitlength - 1)**2])
        Array2D_Weight_Matrix_Known_Part_2 \
            = numpy.zeros([Parameter_Int_Order_of_Derivative - 1, \
                (2 * Int_Number_of_Unitlength - 1)**2 - (Int_Number_of_Unitlength - 1)**2])
        for i_Derivative_2 in numpy.arange(1, Parameter_Int_Order_of_Derivative):
                ## Calculation of start and index on the whole matrix
                Temp_Array2D_Weight_Matrix  \
                    = numpy.zeros([2 * Int_Number_of_Unitlength - 1,2 * Int_Number_of_Unitlength - 1])
                Temp_Array2D_Weight_Matrix[:,i_Derivative_2] = Array2D_Moment_Coefficient[i_Derivative_2, :]
                Temp_Array_Weight_Matrix_Unknown \
                    = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,:Int_Number_of_Unitlength - 1].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known_1 \
                    = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1:].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known_2 \
                    = Temp_Array2D_Weight_Matrix[Int_Number_of_Unitlength - 1:,].\
                        reshape(-1, order = 'C')
                Temp_Array_Weight_Matrix_Known \
                    = numpy.append(Temp_Array_Weight_Matrix_Known_1, Temp_Array_Weight_Matrix_Known_2)
                Array2D_Weight_Matrix_Unknown_Part_2[i_Derivative_2 - 1,:] = Temp_Array_Weight_Matrix_Unknown
                Array2D_Weight_Matrix_Known_Part_2[i_Derivative_2 - 1,:] = Temp_Array_Weight_Matrix_Known
        # Equation set - Part 3:
        Array2D_Weight_Matrix_Unknown_Part_3 \
            = numpy.zeros([1, (Int_Number_of_Unitlength - 1)**2])
        Array2D_Weight_Matrix_Known_Part_3 \
            = numpy.zeros([1, \
                (2 * Int_Number_of_Unitlength - 1)**2 - (Int_Number_of_Unitlength - 1)**2])
        ## Calculation of start and index on the whole matrix
        Temp_Array2D_Weight_Matrix \
            = numpy.dot(Array2D_Moment_Coefficient[0,:].reshape(-1,1), \
                        Array2D_Moment_Coefficient[Parameter_Int_Order_of_Derivative, :].reshape(1,-1))
        Temp_Array_Weight_Matrix_Unknown \
            = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,:Int_Number_of_Unitlength - 1].\
                reshape(-1, order = 'C')
        Temp_Array_Weight_Matrix_Known_1 \
            = Temp_Array2D_Weight_Matrix[:Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1:].\
                reshape(-1, order = 'C')
        Temp_Array_Weight_Matrix_Known_2 \
            = Temp_Array2D_Weight_Matrix[Int_Number_of_Unitlength - 1:,].\
                reshape(-1, order = 'C')
        Temp_Array_Weight_Matrix_Known \
            = numpy.append(Temp_Array_Weight_Matrix_Known_1, Temp_Array_Weight_Matrix_Known_2)
        Array2D_Weight_Matrix_Unknown_Part_3[0,:] = Temp_Array_Weight_Matrix_Unknown
        Array2D_Weight_Matrix_Known_Part_3[0,:] = Temp_Array_Weight_Matrix_Known

        # Calculation of known and unknown coefficient array
        Temp_Array2D_Connection_Coefficient_Known_1 \
            = Array2D_Connection_Coefficient_Proper_Order_0N\
                [:Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1:].reshape(-1, order = 'C')
        Temp_Array2D_Connection_Coefficient_Known_2 \
            = Array2D_Connection_Coefficient_Proper_Order_0N[Int_Number_of_Unitlength - 1:,].\
                reshape(-1, order = 'C')
        Temp_Array2D_Connection_Coefficient_Known \
            = numpy.append(Temp_Array2D_Connection_Coefficient_Known_1, Temp_Array2D_Connection_Coefficient_Known_2)
        # Convert to matrix product form
        Array2D_Identity = numpy.eye((Int_Number_of_Unitlength - 1)**2)


        Temp_Array2D_Left_Part_1 \
            = Array2D_Identity - Array2D_Weight_Matrix_Unknown_Part_1 * 2**(Parameter_Int_Order_of_Derivative - 1)
        Temp_Array_Right_Part_1 \
            = Array2D_Weight_Matrix_Known_Part_1.dot(Temp_Array2D_Connection_Coefficient_Known) * 2**(-1)
        Temp_Array2D_Left_Part_2 \
            = Array2D_Weight_Matrix_Unknown_Part_2
        Temp_Array_Right_Part_2 \
            = - Array2D_Weight_Matrix_Known_Part_2.dot(Temp_Array2D_Connection_Coefficient_Known) * 2**(-1)
        Temp_Array2D_Left_Part_3 \
            = Array2D_Weight_Matrix_Unknown_Part_3
        Temp_Array_Right_Part_3 \
            = (Int_Number_of_Unitlength - 1) * scipy.special.factorial(Parameter_Int_Order_of_Derivative) \
            - Array2D_Weight_Matrix_Known_Part_3.dot(Temp_Array2D_Connection_Coefficient_Known)

        Temp_Array2D_Left = Temp_Array2D_Left_Part_1.copy()
        Temp_Array2D_Left[-Parameter_Int_Order_of_Derivative:-1,:] \
            = Temp_Array2D_Left_Part_2.copy()
        Temp_Array2D_Left[-1,:] = Temp_Array2D_Left_Part_3

        Temp_Array_Right = Temp_Array_Right_Part_1.copy()
        Temp_Array_Right[-Parameter_Int_Order_of_Derivative:-1] \
            = Temp_Array_Right_Part_2.copy()
        Temp_Array_Right[-1] = Temp_Array_Right_Part_3


        Temp_Array2D_Left_inv = numpy.linalg.inv(Temp_Array2D_Left)
        Temp_Array_Unknown \
            = Temp_Array2D_Left_inv.dot(Temp_Array_Right)
        Temp_Array2D_Unknown = Temp_Array_Unknown.reshape((Int_Number_of_Unitlength - 1,Int_Number_of_Unitlength - 1))
        Array2D_Connection_Coefficient_Proper_Order_0N[:Int_Number_of_Unitlength - 1, :Int_Number_of_Unitlength - 1] \
            = Temp_Array2D_Unknown
        Array_Mapping_Col_to_Translation \
            = numpy.linspace(- (Int_Number_of_Unitlength - 1), Int_Number_of_Unitlength - 1, \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        Array2D_Connection_Coefficient_Proper_Order_0N[Int_Number_of_Unitlength:, Int_Number_of_Unitlength:] \
            = Array2D_Connection_Coefficient_Proper_Order_0N[Int_Number_of_Unitlength:, Int_Number_of_Unitlength:]\
            - Array2D_Connection_Coefficient_Proper_Order_0N[:Int_Number_of_Unitlength-1,:Int_Number_of_Unitlength-1]
        return Array2D_Connection_Coefficient_Proper_Order_0N, Array_Mapping_Col_to_Translation

    def Function_Essential_Connection_Coefficient_Matrix_Proper_Order_00_Validation_Direct_Calculation(self, \
            Parameter_Order_of_db):
        """
        This function calculates the moments of scaling function based on directly integration following the definition
        of moment.
        """
        ## Parameter setting
        Parameter_Calculation_Refinement_Scale = 12 
        ## Scaing function and corresponding values
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Calculation_Refinement_Scale)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max())
        ## Calculation by integration
        Array2D_Connection_Coefficient_Order_00 \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1])
        Array_Mapping_Col_to_Translation \
            = numpy.linspace(- (Int_Number_of_Unitlength - 1), Int_Number_of_Unitlength - 1, \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        for i_Row in range(Array2D_Connection_Coefficient_Order_00.shape[0]):
            for i_Col in range(Array2D_Connection_Coefficient_Order_00.shape[1]):
                Temp_Array_1 \
                    = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  2 * (Int_Number_of_Unitlength - 1))
                Int_Index_Start = i_Row * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_1[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function
                Temp_Array_2 \
                    = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  2 * (Int_Number_of_Unitlength - 1))
                Int_Index_Start = i_Col * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_2[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function
                Int_Index_Start = (Int_Number_of_Unitlength - 1) * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Array2D_Connection_Coefficient_Order_00[i_Row, i_Col] \
                    = numpy.sum(Temp_Array_1[Int_Index_Start:Int_Index_Ended] \
                                    * Temp_Array_2[Int_Index_Start:Int_Index_Ended] \
                                    * Value_Delta_T_Wavelet)
        return Array2D_Connection_Coefficient_Order_00, Array_Mapping_Col_to_Translation

    def Function_Essential_Connection_Coefficient_Matrix_Proper_Order_01_Validation_Direct_Calculation(self, \
                Parameter_Order_of_db, Parameter_Calculation_Refinement_Scale = 12):
        """
        This function calculates the moments of scaling function based on 
        directly integration following the definition of moment.
        ----------
        Note:
            The calculation of derivative of wavelet function is a close-form 
            method, the validation of this close-form method has been done.
        ----------
        Output:
            Array2D_Connection_Coefficient_Order_01
            Array_Mapping_Col_to_Translation
        """
        
        ## Scaing function and corresponding values
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Calculation_Refinement_Scale)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max())

        ## Calculation by integration
        Tuple_Function_Return \
            = self.Function_Wavelet_Function_Derivative_Connection_Coefficient_Method(\
                    Parameter_Order_of_db, Parameter_Calculation_Refinement_Scale, 1)
        Array2D_Wavelet_Function_Derivative = Tuple_Function_Return[0]
        Array_Scaling_Function_Derivative_Order_1 = Array2D_Wavelet_Function_Derivative[:,1]

        Array2D_Connection_Coefficient_Order_01 \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1])
        Array_Mapping_Col_to_Translation \
            = numpy.linspace(- (Int_Number_of_Unitlength - 1), Int_Number_of_Unitlength - 1, \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        for i_Row in range(Array2D_Connection_Coefficient_Order_01.shape[0]):
            for i_Col in range(Array2D_Connection_Coefficient_Order_01.shape[1]):
                Temp_Array_1 \
                    = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  2 * (Int_Number_of_Unitlength - 1))
                Int_Index_Start = i_Row * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_1[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function
                Temp_Array_2 \
                    = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  2 * (Int_Number_of_Unitlength - 1))
                Int_Index_Start = i_Col * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_2[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function_Derivative_Order_1
                Int_Index_Start = (Int_Number_of_Unitlength - 1) * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Array2D_Connection_Coefficient_Order_01[i_Row, i_Col] \
                    = numpy.sum(Temp_Array_1[Int_Index_Start:Int_Index_Ended] \
                                    * Temp_Array_2[Int_Index_Start:Int_Index_Ended] \
                                    * Value_Delta_T_Wavelet)
        return Array2D_Connection_Coefficient_Order_01, Array_Mapping_Col_to_Translation

    def Function_Essential_Connection_Coefficient_Matrix_Proper_Order_0N_Validation_Direct_Calculation(self, \
            Parameter_Order_of_db, Parameter_Int_Order_of_Derivative):
        """
        This function calculates the moments of scaling function based on 
        directly integration following the definition of moment.
        ----------
        Note:
            The calculation of derivative of wavelet function is a close-form 
            method, the validation of this close-form method has been done.
        ----------
        Output:
            Array2D_Connection_Coefficient_Order_01
            Array_Mapping_Col_to_Translation
        """
        ## Parameter setting
        Parameter_Calculation_Refinement_Scale = 12
        ## Scaing function and corresponding values
        Str_Name_Wavelet = 'db{}'.format(Parameter_Order_of_db)
        Wavelet_DBN = pywt.Wavelet(Str_Name_Wavelet)
        Tuple_Function_Return = Wavelet_DBN.wavefun(Parameter_Calculation_Refinement_Scale)
        Array_Time_Wavelet = Tuple_Function_Return[2]
        Array_Scaling_Function = Tuple_Function_Return[0]
        Value_Delta_T_Wavelet = numpy.diff(Array_Time_Wavelet).mean()
        Int_Unit_Length = numpy.argwhere(Array_Time_Wavelet == 1)[0][0]
        Int_Number_of_Unitlength = int(Array_Time_Wavelet.max())

        ## Calculation by integration
        Parameter_Int_Order_of_Derivative = int(1)
        Tuple_Function_Return \
            = self.Function_Wavelet_Function_Derivative_Connection_Coefficient_Method(\
                    Parameter_Order_of_db, Parameter_Calculation_Refinement_Scale, Parameter_Int_Order_of_Derivative)
        Array2D_Wavelet_Function_Derivative = Tuple_Function_Return[0]
        Array_Scaling_Function_Derivative_Order_N \
            = Array2D_Wavelet_Function_Derivative[:,Parameter_Int_Order_of_Derivative]

        Array2D_Connection_Coefficient_Order_0N \
            = numpy.zeros([Int_Number_of_Unitlength * 2 - 1, Int_Number_of_Unitlength * 2 - 1])
        Array_Mapping_Col_to_Translation \
            = numpy.linspace(- (Int_Number_of_Unitlength - 1), Int_Number_of_Unitlength - 1, \
                                Int_Number_of_Unitlength * 2 - 1, dtype = numpy.int)
        for i_Row in range(Array2D_Connection_Coefficient_Order_0N.shape[0]):
            for i_Col in range(Array2D_Connection_Coefficient_Order_0N.shape[1]):
                Temp_Array_1 \
                    = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  2 * (Int_Number_of_Unitlength - 1))
                Int_Index_Start = i_Row * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_1[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function
                Temp_Array_2 \
                    = numpy.zeros(Array_Time_Wavelet.size + Int_Unit_Length *  2 * (Int_Number_of_Unitlength - 1))
                Int_Index_Start = i_Col * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Temp_Array_2[Int_Index_Start: Int_Index_Ended] = Array_Scaling_Function_Derivative_Order_N
                Int_Index_Start = (Int_Number_of_Unitlength - 1) * Int_Unit_Length
                Int_Index_Ended = Int_Index_Start + Array_Time_Wavelet.size
                Array2D_Connection_Coefficient_Order_0N[i_Row, i_Col] \
                    = numpy.sum(Temp_Array_1[Int_Index_Start:Int_Index_Ended] \
                                    * Temp_Array_2[Int_Index_Start:Int_Index_Ended] \
                                    * Value_Delta_T_Wavelet)
        return Array2D_Connection_Coefficient_Order_0N, Array_Mapping_Col_to_Translation


class Class_Simulation_Hilbert_wavelet():
    """
    ## Note
    # The calculation default: (Other format of parameters or variables need 
    # to be converted to the following form for the convenience of calculation)
    # Parameter Matrix A (Row for Points, Col for Unknowns)
    # The code is based on the close form simulation
    # This code is generated for numerical solutions
    # A test to see if the single time integration can accelerate the simulation
    # speed.
    """
    
    def __init__(self):
        self.Class_Name = 'Class of Hilbert-wavelet-based simulation'
    ## Functions
    def Function_Low_Frequency_Random_Process(self, Array_Time, Array_Z_Coordinate, Parameter_Stationary_Period):
        numpy.random.seed()
        Int_Number_Frequencies = 100
        Array_Low_Frequency_Random_Process = numpy.zeros(Array_Time.shape)
        for i_Frequency in range(Int_Number_Frequencies):
            Value_Initial_Phase = numpy.random.rand(1) * 2 * numpy.pi
            Parameter_Frequency_1 =  2 / (1 + i_Frequency / Int_Number_Frequencies) / Parameter_Stationary_Period
            Array_Low_Frequency_Random_Process \
                += numpy.cos(Parameter_Frequency_1 * Array_Time * 2 * numpy.pi + Value_Initial_Phase)
        return Array_Low_Frequency_Random_Process

    def Function_CDF_Mapping_Uniform(self, \
            Array_Low_Frequency_Random_Process):
        Array_Low_Frequency_Random_Process_Sorted \
            = numpy.sort(Array_Low_Frequency_Random_Process)
        # This mapping process still needs modification to make the 
        # projection perfect
        Array_CDF \
            = numpy.linspace(0, 1, \
                            Array_Low_Frequency_Random_Process_Sorted.size)
        Function_Array_CDF \
            = interpolate.interp1d(Array_Low_Frequency_Random_Process_Sorted, \
                                    Array_CDF, \
                                    kind = 'linear')
        Array_PHI = Function_Array_CDF(Array_Low_Frequency_Random_Process)
        Array_Low_Frequency_Random_Process_Normal = Array_PHI.copy()
        return Array_Low_Frequency_Random_Process_Normal

    def Function_Instantaneous_Phase_Generation(self, \
            Array_Instantaneous_Phase_Measurement, Str_Method):
        """
        Return Generated instantaneous phase
        Three methods can be used:
        1. DWT_based
        2. FFT_based
        3. Original : Direct return original instantanous phase
        """
        if Str_Method == 'DWT_based':
            Array_Instantaneous_Frequency_Measurement \
                = numpy.diff(Array_Instantaneous_Phase_Measurement)
            WPT_Instantaneous_Frequency \
                = pywt.WaveletPacket(\
                    data = Array_Instantaneous_Frequency_Measurement, \
                    wavelet = Parameter_Str_Wavelet_Name, \
                    mode = 'symmetric')
            Value_Maximum_Level = WPT_Instantaneous_Frequency.maxlevel
            WPT_Instantaneous_Frequency_Reconstruct \
                = pywt.WaveletPacket\
                        (data = numpy.zeros\
                                (Array_Instantaneous_Phase_Measurement.shape), \
                        wavelet = Parameter_Str_Wavelet_Name, mode='symmetric')
            for node in WPT_Instantaneous_Frequency\
                            .get_level(Value_Maximum_Level, 'natural'):
                WPT_Instantaneous_Frequency_Reconstruct[node.path] \
                    = WPT_Instantaneous_Frequency[node.path].data[numpy.random\
                        .permutation\
                            (WPT_Instantaneous_Frequency[node.path].data.size)]
            Array_Instantaneous_Frequency_Simulation \
                = WPT_Instantaneous_Frequency_Reconstruct.reconstruct()
            Array_Instantaneous_Phase_Simulation \
                = Array_Instantaneous_Phase_Measurement
            Int_Length_Time = Array_Instantaneous_Phase_Measurement.size
            for i_Time in numpy.arange(1, Int_Length_Time,1):
                Array_Instantaneous_Phase_Simulation[i_Time] \
                    = Array_Instantaneous_Phase_Measurement[0] \
                        + numpy.sum(Array_Instantaneous_Frequency_Simulation\
                                        [0:i_Time-1])
        elif Str_Method == 'FFT_based':
            Array_Frequency \
                = numpy.zeros(Array_Instantaneous_Phase_Measurement.size)
            Array_Frequency[1:] \
                = numpy.diff(Array_Instantaneous_Phase_Measurement)
            Array_FFT = numpy.fft.fft(Array_Frequency)
            Array_FFT_Amplitude = numpy.abs(Array_FFT)
            Array_FFT_Phase = numpy.angle(Array_FFT)
            if Array_FFT.size % 2 == 0:
                Temp_Array_Half_Phase_1 \
                    = Array_FFT_Phase[1 : Array_FFT.size // 2]
            else:
                Temp_Array_Half_Phase_1 \
                    = Array_FFT_Phase[1 : (Array_FFT.size + 1) // 2]

            numpy.random.shuffle(Temp_Array_Half_Phase_1)
            Temp_Array_Half_Phase_2 \
                = -numpy.flipud(Temp_Array_Half_Phase_1)
            Array_FFT_Phase[1: 1 + Temp_Array_Half_Phase_1.size] \
                = Temp_Array_Half_Phase_1
            Array_FFT_Phase[- Temp_Array_Half_Phase_2.size:] \
                = Temp_Array_Half_Phase_2

            Array_FFT_Simulation \
                = Array_FFT_Amplitude * numpy.exp(1j * Array_FFT_Phase)
            Array_Frequency_Simulation \
                = numpy.real(numpy.fft.ifft(Array_FFT_Simulation))
            Array_Instantaneous_Phase_Simulation \
                = Array_Instantaneous_Phase_Measurement.copy()
            Int_Length_Time = Array_Instantaneous_Phase_Measurement.size
            for i_Time in numpy.arange(1, Int_Length_Time,1):
                Array_Instantaneous_Phase_Simulation[i_Time] \
                    = Array_Instantaneous_Phase_Simulation[i_Time - 1] \
                        + Array_Frequency_Simulation[i_Time]
        elif Str_Method == 'Original':
            Array_Instantaneous_Phase_Simulation \
                = Array_Instantaneous_Phase_Measurement
        else:
            print('Please specify the instantaneous phase generation method')
        return Array_Instantaneous_Phase_Simulation

    def Function_Current_Node_Simulation_Single_Point(self, \
            Array_Instantaneous_Phase_Measurement, \
            Array_Instantaneous_Amplitude_Measurement,
            Str_Method):
        """
        Single point, Sigle subcomponent simulation
        ------------------------------------------------------------------------
        Input:
            Array_Instantaneous_Phase_Measurement
            Array_Instantaneous_Amplitude_Measurement
            Str_Method:
                1. DWT_based
                2. FFT_based
                3. Original : Direct return original instantanous phase
        ------------------------------------------------------------------------
        Output:
            Array_Simulation_Current_Node
            Array_Instantaneous_Phase_Simulation
        """
        Array_Instantaneous_Phase_Simulation \
            = self.Function_Instantaneous_Phase_Generation\
                    (Array_Instantaneous_Phase_Measurement, Str_Method)
        Array_Simulation_Current_Node \
            = numpy.cos(Array_Instantaneous_Phase_Simulation) \
                * Array_Instantaneous_Amplitude_Measurement
        return Array_Simulation_Current_Node, \
                Array_Instantaneous_Phase_Simulation

    def Function_Simulation_Single_Point_Single_Thread(self,
            Array_Signal, Array_Time, \
            Array2D_Signal_Decomposed_Amplitude, 
            Array2D_Signal_Decomposed_Phase, \
            Array2D_Signal_Decomposed_Frequency, \
            Array_Signal_Decomposed_Center_Frequency):
        """
        Single point simulation
        ------------------------------------------------------------------------
        Input:
            Array_Signal, Array_Time,
            Array2D_Signal_Decomposed_Amplitude,
            Array2D_Signal_Decomposed_Phase,
            Array2D_Signal_Decomposed_Frequency,
            Array_Signal_Decomposed_Center_Frequency
        ------------------------------------------------------------------------
        Output:
            Array2D_Node_Simulation_Decomposed
            Str_Name_Description
        """
        Int_Number_Nodes = Array_Signal_Decomposed_Center_Frequency.size
        Array2D_Node_Simulation_Decomposed \
            = numpy.zeros((Array_Signal.shape[0], \
                            Array_Signal_Decomposed_Center_Frequency.shape[0]))
        for i_Current_Node in numpy.arange(0, Int_Number_Nodes):
            Array_Instantaneous_Phase_Measurement \
                = Array2D_Signal_Decomposed_Phase[:,i_Current_Node]
            Array_Instantaneous_Amplitude_Measurement \
                = Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node]
            Temp_Tuple_Return_Value \
                = self.Function_Current_Node_Simulation_Single_Point\
                        (Array_Instantaneous_Phase_Measurement, \
                            Array_Instantaneous_Amplitude_Measurement)
            Array2D_Node_Simulation_Decomposed[:,i_Current_Node] \
                = Temp_Tuple_Return_Value[0]
        Str_Name_Description = 'Single point simulation'
        return Array2D_Node_Simulation_Decomposed, Str_Name_Description

    def Function_Function_Value(self, \
            Array2D_A, Array2D_X, Array2D_Matrix_Correlation):
        """
        This function calculates the function value of the target function
        At the bottom of the code block, the validation program is written and 
        commeented
        ------------------------------------------------------------------------
        Input:
            Array2D_A, Array2D_X, Array2D_Matrix_Correlation
        ------------------------------------------------------------------------
        Output:
            Array_Function_Value_F
        """
        Int_Number_Discrete_x = Array2D_X.shape[1]
        Int_Number_Points = Array2D_A.shape[0]
        Array_Correlation_Coefficient \
            = numpy.zeros(int(Int_Number_Points * (Int_Number_Points - 1) / 2))
        i_Start = 0
        for i_Point in numpy.arange(1, Int_Number_Points, 1):
            Temp_Array_Correlation_Coefficient_Diag \
                = numpy.diag(Array2D_Matrix_Correlation, i_Point)
            Array_Correlation_Coefficient\
                [i_Start\
                    : i_Start + Temp_Array_Correlation_Coefficient_Diag.size] \
                = Temp_Array_Correlation_Coefficient_Diag
            i_Start += Temp_Array_Correlation_Coefficient_Diag.size

        Array_Function_Value_f_cos \
            = numpy.zeros(Array_Correlation_Coefficient.size)
        ### Acceleration trial 1 --- Vectrorize ###
        # i_Array_Function_Value_f_cos = int(-1)
        # for i_Delta_n in numpy.arange(1, Int_Number_Points, 1):
        #     for i_Point in range(Int_Number_Points - i_Delta_n):
        #         i_Array_Function_Value_f_cos += 1
        #         Array_Delta_A = Array2D_A[i_Point + i_Delta_n, :] - Array2D_A[i_Point, :]
        #         Temp_Value_f_cos = 0
        #         # for i_Discrete_X in range(Int_Number_Discrete_x):
        #         #     Temp_Value_f_cos += numpy.cos(numpy.sum(Array_Delta_A * Array2D_X[:, i_Discrete_X]))
        #         Temp_Value_f_cos = \
        #             numpy.sum(numpy.cos(numpy.sum(\
        #                 numpy.dot(Array_Delta_A.reshape(-1,1), numpy.ones((1, Int_Number_Discrete_x))) \
        #                 * Array2D_X, axis = 0)))
        #         Array_Function_Value_f_cos[i_Array_Function_Value_f_cos] = Temp_Value_f_cos
        ### Acceleration trial 2 -- Reduce a further loop ###
        i_Start = 0
        Int_Range = 0
        for i_Delta_n in numpy.arange(1, Int_Number_Points, 1):
            i_Start += Int_Range
            Int_Range = Int_Number_Points - i_Delta_n
            Array2D_Delta_A \
                = (Array2D_A[i_Delta_n:, :] - Array2D_A[:-i_Delta_n, :]).T
            Array3D_Delta_A \
                = numpy.repeat(Array2D_Delta_A[:, numpy.newaxis, :],  \
                                Int_Number_Discrete_x, axis = 1)
            Array3D_X = numpy.repeat(Array2D_X[:,:, numpy.newaxis], \
                                        Int_Number_Points - i_Delta_n, axis = 2)
            Array_i_Delta_n_Function_Value_f_cos\
                = numpy.sum(numpy.cos(numpy.sum(Array3D_Delta_A * Array3D_X, \
                                                axis = 0)), \
                            axis = 0)
            Array_Function_Value_f_cos[i_Start: i_Start + Int_Range] \
                = Array_i_Delta_n_Function_Value_f_cos

        Array_Function_Value_f_cos \
            = Array_Function_Value_f_cos / Int_Number_Discrete_x
        Array_Function_Value_g \
            = Array_Function_Value_f_cos - Array_Correlation_Coefficient
        Array_Function_Value_F = Array_Function_Value_g
        """Following is the validation code block"""
        """ This function has been validated by the validation code block"""
        # def Sub_Function_Value_M(Array2D_A, i_Point,Value_x):
        #     Array_A = Array2D_A[i_Point, :]
        #     Array_x = numpy.zeros(Array_A.shape)
        #     for i_u in range(Array_A.size):
        #         Array_x[i_u] = Value_x**i_u
        #     Value_M = numpy.sum(Array_A * Array_x)
        #     return Value_M
        # def Sub_Function_Function_Value_f(Array2D_A, Array2D_X, i_Point, i_Delta_n):
        #     Int_Number_Discrete_x = Array2D_X.shape[1]
        #     Temp_Value_Function_Value_f = 0
        #     for i_x in range(Int_Number_Discrete_x):
        #         Temp_Value_M_m = Sub_Function_Value_M(Array2D_A, i_Point + i_Delta_n, Array2D_X[1, i_x])
        #         Temp_Value_M_l = Sub_Function_Value_M(Array2D_A, i_Point, Array2D_X[1,i_x])
        #         Temp_Value_Function_Value_f += numpy.cos(Temp_Value_M_m - Temp_Value_M_l)
        #     Value_Function_f = Temp_Value_Function_Value_f / Int_Number_Discrete_x
        #     return Value_Function_f
        # def Sub_Function_Function_Value_F(Array2D_A, Array2D_X, i_Point, i_Delta_n, Array2D_Matrix_Correlation):
        #     Value_Function_f = Sub_Function_Function_Value_f(Array2D_A, Array2D_X, i_Point, i_Delta_n)
        #     Value_Function_F = (Value_Function_f - Array2D_Matrix_Correlation[i_Point, i_Delta_n + i_Point])
        #     return Value_Function_F
        # def Sub_Function_Function_Array_F(Array2D_A, Array2D_X):
        #     Array_Function_Value_F = numpy.zeros(int(Array2D_A.shape[0] * (Array2D_A.shape[0] - 1) / 2))
        #     i_Array_Function_Value_F = -1
        #     for i_Delta_n in numpy.arange(1, Array2D_A.shape[0], 1):
        #         for i_Point in range(Int_Number_Points - i_Delta_n):
        #             i_Array_Function_Value_F += 1
        #             Array_Function_Value_F[i_Array_Function_Value_F]\
        #             =Sub_Function_Function_Value_F(Array2D_A, Array2D_X,i_Point,i_Delta_n,Array2D_Matrix_Correlation)
        #     return Array_Function_Value_F
        # def Sub_Function_Validation(Array2D_A, Array2D_X, Array2D_Matrix_Correlation):
        #     Array_Function_Value_F_Validate = Sub_Function_Function_Array_F(Array2D_A, Array2D_X)
        #     Array_Function_Value_F = Function_Function_Value(Array2D_A, Array2D_X, Array2D_Matrix_Correlation)
        #     Array_Difference = Array_Function_Value_F_Validate - Array_Function_Value_F
        #     return Array_Difference
        return Array_Function_Value_F

    def Function_Jacobian_Function_Value(self, Array2D_A, Array2D_X, Array2D_Matrix_Correlation):
        """ 
        This is the Jacobian Functio.
        The main code is in the first code block and has been validated by the validation code block. 
        The validation code is in the second code block.
        """
        ## """Main Code"""
        Int_Number_Discrete_x = Array2D_X.shape[1]
        Int_Number_Ploy_Power = Array2D_A.shape[1]
        Int_Number_Points = Array2D_A.shape[0]
        Int_Size_Array_Function = int(Array2D_A.shape[0] * (Array2D_A.shape[0] - 1) / 2)
        Array2D_Jacobian_Function_Value \
            = numpy.zeros((Int_Size_Array_Function, Int_Number_Points * Int_Number_Ploy_Power))
        ## Acceleration 1 - Vectorize #
        # i_Array_Function_Value_f = int(-1)
        # for i_Delta_n in numpy.arange(1, Int_Number_Points, 1):
        #     for i_Point in range(Int_Number_Points - i_Delta_n):
        #         i_Array_Function_Value_f += 1
        #         Array_Delta_A = Array2D_A[i_Point + i_Delta_n, :] - Array2D_A[i_Point, :]
        #         i_Point_Col = i_Point
        #         Temp_Basic_Sinelement = numpy.dot(Array_Delta_A.reshape(-1,1), numpy.ones((1, Int_Number_Discrete_x)))
        #         Temp_Basic_Sine = numpy.sin(numpy.sum(Temp_Basic_Sinelement * Array2D_X, axis = 0))
        #         Temp_Multiply = numpy.dot(numpy.ones((Int_Number_Ploy_Power, 1)), Temp_Basic_Sine.reshape(1, -1))\
        #                         * Array2D_X
        #         Temp_Array_Jacobian = numpy.sum(Temp_Multiply, axis = 1) * 1 / Int_Number_Discrete_x
        #         Array2D_Jacobian_Function_Value[i_Array_Function_Value_f,\
        #                 i_Point_Col * Int_Number_Ploy_Power: i_Point_Col * Int_Number_Ploy_Power + Int_Number_Ploy_Power] \
        #                 = Temp_Array_Jacobian
        #         # for i_u in range(Int_Number_Ploy_Power):
        #             # i_Col = i_Point_Col * Int_Number_Ploy_Power + i_u
        #             # Temp_Value_f_sin = 0
        #             # for i_Discrete_X in range(Int_Number_Discrete_x):
        #             #     Temp_Value_f_sin += numpy.sin\
        #             #         (numpy.sum(Array_Delta_A * Array2D_X[:, i_Discrete_X])) * Array2D_X[i_u, i_Discrete_X]
        #             # Temp_Value_Jacobian = - Temp_Value_f_sin * 1 / Int_Number_Discrete_x
        #             # Array2D_Jacobian_Function_Value[i_Array_Function_Value_f, i_Col] = Temp_Value_Jacobian
        #         i_Point_Col = i_Point + i_Delta_n
        #         # for i_u in range(Int_Number_Ploy_Power):
        #         #     i_Col = i_Point_Col * Int_Number_Ploy_Power + i_u
        #         #     # Temp_Value_f_sin = 0
        #         #     # for i_Discrete_X in range(Int_Number_Discrete_x):
        #         #     #     Temp_Value_f_sin += numpy.sin\
        #         #     #         (numpy.sum(Array_Delta_A * Array2D_X[:, i_Discrete_X])) * Array2D_X[i_u, i_Discrete_X]
        #         #     Temp_Value_f_sin \
        #         #         = numpy.sum(numpy.sin(numpy.sum(\
        #         #             numpy.dot(Array_Delta_A.reshape(-1,1), numpy.ones((1, Int_Number_Discrete_x))) \
        #         #                 * Array2D_X, axis = 0)) \
        #         #                 * Array2D_X[i_u,:])
        #         #     Temp_Value_Jacobian = - Temp_Value_f_sin * 1 / Int_Number_Discrete_x
        #         #     Array2D_Jacobian_Function_Value[i_Array_Function_Value_f, i_Col] = Temp_Value_Jacobian
        #         Array2D_Jacobian_Function_Value[i_Array_Function_Value_f,\
        #                 i_Point_Col * Int_Number_Ploy_Power: i_Point_Col * Int_Number_Ploy_Power + Int_Number_Ploy_Power] \
        #                 = - Temp_Array_Jacobian
        ## Acceleration 2 -- Further simplify ##
        i_Array_Function_Value_f = int(-1)
        for i_Delta_n in numpy.arange(1, Int_Number_Points, 1):
            Array2D_Delta_A \
                = (Array2D_A[i_Delta_n:, :] - Array2D_A[:-i_Delta_n, :]).T
            Array3D_Delta_A \
                = numpy.repeat(Array2D_Delta_A[:, numpy.newaxis, :],  \
                                Int_Number_Discrete_x, \
                                axis = 1)
            Array3D_X \
                = numpy.repeat(Array2D_X[:,:, numpy.newaxis], \
                                Int_Number_Points - i_Delta_n, \
                                axis = 2)
            Temp_Array2D_Basic_Sine \
                = numpy.sin(numpy.sum(Array3D_Delta_A * Array3D_X, axis = 0))
            Temp_Array3D_Basic_Sine \
                = numpy.repeat(Temp_Array2D_Basic_Sine[numpy.newaxis, :, :], \
                                Int_Number_Ploy_Power, \
                                axis = 0)
            Temp_Array3D_Multiply = Temp_Array3D_Basic_Sine * Array3D_X
            Temp_Array2D_Part_Jacobian \
                = numpy.sum(Temp_Array3D_Multiply, axis = 1) \
                    / Int_Number_Discrete_x
            for i_Point in range(Int_Number_Points - i_Delta_n):
                i_Array_Function_Value_f += 1
                Temp_Array_Jacobian = Temp_Array2D_Part_Jacobian[:,i_Point]
                i_Point_Col = i_Point
                Array2D_Jacobian_Function_Value\
                    [i_Array_Function_Value_f,\
                    i_Point_Col * Int_Number_Ploy_Power \
                        : i_Point_Col * Int_Number_Ploy_Power \
                            + Int_Number_Ploy_Power] \
                    = Temp_Array_Jacobian
                i_Point_Col = i_Point + i_Delta_n
                Array2D_Jacobian_Function_Value\
                    [i_Array_Function_Value_f,\
                    i_Point_Col * Int_Number_Ploy_Power \
                        : i_Point_Col * Int_Number_Ploy_Power \
                            + Int_Number_Ploy_Power] \
                    = - Temp_Array_Jacobian
        """ Validation Code Block"""
        # def Sub_Function_Validate_Jacobian_Function_Array2D(Array2D_A, Array2D_X):
        #     Int_Number_Ploy_Power = Array2D_A.shape[1]
        #     Int_Number_Points = Array2D_A.shape[0]
        #     Int_Size_Array_Function = int(Array2D_A.shape[0] * (Array2D_A.shape[0] - 1) / 2)
        #     Array2D_Jacobian_Function_Value \
        #         = numpy.zeros((Int_Size_Array_Function, Int_Number_Points * Int_Number_Ploy_Power))
        #     i_Array_Function_Value_f = int(-1)
        #     for i_Delta_n in numpy.arange(1, Int_Number_Points, 1):
        #         for i_Point in range(Int_Number_Points - i_Delta_n):
        #             i_Array_Function_Value_f += 1
        #             for i_A_Row in range(Int_Number_Points):
        #                 for i_A_Col in range(Int_Number_Ploy_Power):
        #                     i_Jacobiean_Col = i_A_Row * Int_Number_Ploy_Power + i_A_Col
        #                     Array2D_Jacobian_Function_Value[i_Array_Function_Value_f, i_Jacobiean_Col] \
        #                         =  Sub_Function_Validate_Jacobian_Function_Value\
        #                             (Array2D_A, i_A_Row, i_A_Col, Array2D_X, i_Point, i_Delta_n)
        #     return Array2D_Jacobian_Function_Value
        # def Sub_Function_Validate_Jacobian_Function_Value(Array2D_A, i_A_Row, i_A_Col, Array2D_X, i_Point, i_Delta_n):
        #     Int_Number_Discrete_x = Array2D_X.shape[1]
        #     Value_sin = Sub_Function_Function_Value_sin(Array2D_A, i_A_Row, i_A_Col, Array2D_X, i_Point, i_Delta_n)
        #     Value_Jacobian_Function = 1 / Int_Number_Discrete_x * Value_sin
        #     return Value_Jacobian_Function
        # def Sub_Function_Function_Value_sin(Array2D_A, i_A_Row, i_A_Col, Array2D_X, i_Point, i_Delta_n):
        #     Int_Number_Discrete_x = Array2D_X.shape[1]
        #     Temp_Value_f_sin = 0
        #     for i_u in range(Int_Number_Discrete_x):
        #         Value_x = Array2D_X[1, i_u]
        #         Temp_Value_M_l = Sub_Function_Value_M(Array2D_A, i_Point, Value_x)
        #         Temp_Value_M_m = Sub_Function_Value_M(Array2D_A, i_Point + i_Delta_n, Value_x)
        #         if i_A_Row == i_Point:
        #             Temp_Value_Delta = -1
        #         elif i_A_Row == i_Point + i_Delta_n:
        #             Temp_Value_Delta = 1
        #         else:
        #             Temp_Value_Delta = 0
        #         Temp_Value_f_sin += numpy.sin(Temp_Value_M_m - Temp_Value_M_l) * Array2D_X[i_A_Col,i_u] * Temp_Value_Delta
        #     Value_sin = - Temp_Value_f_sin
        #     return Value_sin
        # def Sub_Function_Value_M(Array2D_A, i_Point,Value_x):
        #     Array_A = Array2D_A[i_Point, :]
        #     Array_x = numpy.zeros(Array_A.shape)
        #     for i_u in range(Array_A.size):
        #         Array_x[i_u] = Value_x**i_u
        #     Value_M = numpy.sum(Array_A * Array_x)
        #     return Value_M
        """Return Value"""
        return Array2D_Jacobian_Function_Value

    def Function_Parameter_Array2D_A(self, Array2D_Matrix_Correlation, i_Current_Node):
        # Matrix_A_Row: Point
        # Matrix_A_Column: Polynominal coefficients
        # The target value of the function is a zero vector
        Flag_Debug = False
        Array2D_X = numpy.zeros((Int_Number_Ploy_Power, Int_Number_Discrete_x))
        for i_Power in range(Int_Number_Ploy_Power):
            Array2D_X[i_Power,:] \
                = numpy.linspace(-0.5, 0.5, Int_Number_Discrete_x)**(i_Power)
        Int_Number_Points = Array2D_Matrix_Correlation.shape[0]
        Temp_Int_Count_Trial = 0
        Int_Max_Iteration = 400
        Flag_Stop_Trial = False
        while Flag_Stop_Trial == False:
            Array_Sum_Erro = numpy.zeros(Int_Max_Iteration * 4)
            Temp_Int_Count_Trial += 1
            numpy.random.seed()
            Array2D_A \
                = numpy.zeros((Int_Number_Points, \
                                Int_Number_Ploy_Power)) # Coefficient matrix
            Array2D_A \
                = numpy.random.rand(Int_Number_Points, \
                                    Int_Number_Ploy_Power) / 10
            Array_Target_Function_Value_F \
                = numpy.zeros(int(Int_Number_Points * (Int_Number_Points - 1) \
                                / 2))
            Flag_Iteration = True
            Temp_Value_lambda = 1E5
            Value_L = 10
            Value_epsilon = numpy.sqrt(Value_Error / Int_Number_Points)
            Temp_Int_Count_Iteration = 0
            while Flag_Iteration == True:
                Time_Start = time.time()
                Temp_Int_Count_Iteration += 1
                Array_Function_Value_F \
                    = self.Function_Function_Value\
                            (Array2D_A, Array2D_X, Array2D_Matrix_Correlation)
                Array2D_Jacobian_Function_Value \
                    = self.Function_Jacobian_Function_Value\
                            (Array2D_A, Array2D_X, Array2D_Matrix_Correlation)
                Array_Delta_Function_Value \
                    = Array_Target_Function_Value_F - Array_Function_Value_F
                Array_Right \
                    = numpy.dot(Array2D_Jacobian_Function_Value.T, \
                                numpy.reshape(Array_Delta_Function_Value, \
                                            (Array_Function_Value_F.size, 1)))
                Array2D_JWJ \
                    = numpy.dot(Array2D_Jacobian_Function_Value.T, \
                                Array2D_Jacobian_Function_Value)
                Array2D_Left \
                    = Array2D_JWJ \
                        + Temp_Value_lambda \
                            * numpy.diag(numpy.diag(Array2D_JWJ))
                Array_Delta_A_Update \
                    = numpy.dot(numpy.linalg.inv(Array2D_Left), Array_Right)
                Array2D_Delta_A_Update \
                    = Array_Delta_A_Update\
                        .reshape(Int_Number_Points, Int_Number_Ploy_Power)
                Array_Function_Value_F_Update \
                    = self.Function_Function_Value\
                            (Array2D_A + Array2D_Delta_A_Update, \
                            Array2D_X, \
                            Array2D_Matrix_Correlation)
                Value_Chi_Square_Delta \
                    = numpy.sum(Array_Function_Value_F**2) \
                        - numpy.sum(Array_Function_Value_F_Update**2)
                Temp_Value_Denominator_Part_21 \
                    = numpy.dot(Temp_Value_lambda \
                                    * numpy.diag(numpy.diag(Array2D_JWJ)), \
                                Array_Delta_A_Update\
                                    .reshape(Array_Delta_A_Update.size, 1))
                Temp_Value_Denominator_Part_22 = Array_Right
                Temp_Value_Denominator_Part_2 \
                    = Temp_Value_Denominator_Part_21 \
                        + Temp_Value_Denominator_Part_22
                Temp_Value_Denominator \
                    = numpy.dot(Array_Delta_A_Update\
                                    .reshape(1,Array_Delta_A_Update.size), \
                                Temp_Value_Denominator_Part_2)
                Value_rho = Value_Chi_Square_Delta / Temp_Value_Denominator
                # print(Value_rho)
                Time_End = time.time()
                Array_Sum_Erro[Temp_Int_Count_Iteration] \
                    = numpy.sum(numpy.abs(Array_Function_Value_F))

                if Temp_Int_Count_Iteration >= Array_Sum_Erro.size - 2:
                    Flag_Iteration = False
                    Flag_Compelete = False
                    Array2D_A = 0
                    if Flag_Debug:
                        print('Too many iteration and start new trial \n')
                elif Temp_Int_Count_Iteration >= 30 \
                        and (Array_Sum_Erro[Temp_Int_Count_Iteration - 25] \
                                - Array_Sum_Erro[Temp_Int_Count_Iteration]) \
                                    < 0.005:
                    Flag_Iteration = False
                    Flag_Compelete = False
                    Array_A = 0
                    if Flag_Debug:
                        print('Saddle point and start new trial \n')
                # if Temp_Int_Count_Iteration > 100 and numpy.max(numpy.abs(Array_Function_Value_F)) >= 0.2:
                #     Flag_Iteration = False
                #     Flag_Compelete = False
                #     Array2D_A = 0
                #     if Flag_Debug:
                #         print('Bad Initial Value \n')
                if numpy.max(numpy.abs(Array_Function_Value_F)) \
                        <= Value_Error_Threshold:
                    Flag_Iteration = False
                    Flag_Compelete = True
                    if Flag_Debug:
                        print('--------------------------------------------\n')
                        Str_Output = 'Solution obtained \n\tNode: {}\n\tNumber of Intration: {} \n'\
                                        .format(i_Current_Node, Temp_Int_Count_Iteration)
                        print(Str_Output)
                        Str_Output = 'Array2D_A first 3x3 elements \n\t'
                        print(Str_Output)
                        print(DataFrame(Array2D_A[0:3,0:3]))
                        print('\n')
                        print('--------------------------------------------\n')
                elif Value_rho >= Value_epsilon:
                    # print('update Array2D_A \n')
                    Array2D_A += Array2D_Delta_A_Update
                    Array2D_A[0,:] = 0
                    Temp_Value_lambda = numpy.max([Temp_Value_lambda / Value_L, 1E-7])
                    Str_Output = 'Sum error: {:.3f}, Max error: {:.3f}, Iteration: {}, Time: {:.3f}'\
                            .format(numpy.sum(numpy.abs(Array_Function_Value_F)), \
                                numpy.max(numpy.abs(Array_Function_Value_F)), \
                                Temp_Int_Count_Iteration, \
                                Time_End - Time_Start)
                    if Flag_Debug:
                        print(Str_Output)
                elif Temp_Int_Count_Iteration > Int_Max_Iteration:
                    Flag_Iteration = False
                    Flag_Compelete = False
                    Array2D_A = 0
                    if Flag_Debug:
                        print('Too many iteration and start new trial \n')
                else:
                    Temp_Value_lambda \
                        = numpy.min([Temp_Value_lambda * Value_L, 1E7])
            if Flag_Compelete:
                Flag_Stop_Trial = True
        return Array2D_A

    def Function_Current_Node_Simulation_Decomposed_Non_Gaussian_Mapping_SingleInt\
            (self, \
            Array3D_TV_Correlation, \
            Array_Time_Selected, \
            Array_Index_Time_Selected_Edge, \
            Array_Instantaneous_Phase_Measurement, \
            Array_Instantaneous_Amplitude_Measurement, \
            Array_Time, \
            Array_Z_Coordinate, \
            i_Current_Node, \
            Parameter_Stationary_Period, \
            Flag_Debug = False):
        Time_Start = time.time()
        if Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = 'Node {} Started\n\t Time: {} \n'\
                    .format(i_Current_Node, Str_DateTime_Start)
            print(Str_Output)
        Array2D_Matrix_Correlation = Array3D_TV_Correlation[:,:,0].copy()
        # Parameter Matrix generation (Most time consuming)
        if numpy.mean(numpy.diff(Array3D_TV_Correlation, axis = 2)) == 0:
            Flag_Constant_Correlation = True
            Temp_Str_Cache_Name \
                = 'PA_CT_H{:d}_P{}_N{}'\
                    .format(int(Value_Total_Height), \
                            Int_Number_Points, \
                            i_Current_Node)
        else:
            Flag_Constant_Correlation = False
            Temp_Str_Cache_Name \
                = 'PA_TV_H{:d}_P{}_N{}'\
                    .format(int(Value_Total_Height), \
                            Int_Number_Points, \
                            i_Current_Node)
        os.chdir('Data')
        os.chdir('DataBase_Cache')
        if Flag_Use_Stored_PA:
            print('Use stored PA')
            if Flag_Debug:
                print('Using DataBase of Parameter A')
            RawData = io.loadmat(Temp_Str_Cache_Name)
            Array4D_Parameter_a_Time_Varying_Essential \
                = RawData['Array4D_Parameter_a_Time_Varying_Essential']
            Int_Total_Number_PA \
                = Array4D_Parameter_a_Time_Varying_Essential.shape[3]
            Array3D_Parameter_a_Time_Varying_Essential \
                = Array4D_Parameter_a_Time_Varying_Essential\
                    [:,:,:,int(numpy.random.rand() / 1 * Int_Total_Number_PA)]
        else:
            Array2D_Parameter_a \
                = self.Function_Parameter_Array2D_A\
                        (Array2D_Matrix_Correlation, i_Current_Node)
            Array3D_Parameter_a_Time_Varying_Essential \
                = numpy.zeros((Array2D_Parameter_a.shape[0], \
                                Array2D_Parameter_a.shape[1], \
                                Array_Time_Selected.size))
            i_Array2D_Parameter_a = 0
            for i_Array2D_Parameter_a in range(Array_Time_Selected.size):
                if i_Array2D_Parameter_a == 0:
                    Array3D_Parameter_a_Time_Varying_Essential\
                        [:,:,i_Array2D_Parameter_a] \
                            = Array2D_Parameter_a
                else:
                    Array2D_Matrix_Correlation \
                        = Array3D_TV_Correlation[:,:,i_Array2D_Parameter_a]
                    Value_Max_Diff \
                        = numpy.max(\
                            numpy.abs(Array2D_Matrix_Correlation \
                                        - Array3D_TV_Correlation\
                                            [:,:,i_Array2D_Parameter_a - 1]))
                    if  Value_Max_Diff <= 0.000001:
                        Array3D_Parameter_a_Time_Varying_Essential\
                            [:,:,i_Array2D_Parameter_a] \
                                = Array2D_Parameter_a
                        if Flag_Debug:
                            print('Constant correlation detected, Numerically solved once\n')
                    else:   
                        Array3D_Parameter_a_Time_Varying_Essential\
                            [:,:,i_Array2D_Parameter_a] \
                                = self.Function_Parameter_Array2D_A(\
                                        Array2D_Matrix_Correlation, \
                                        i_Current_Node)
                        if Flag_Debug:
                            Str_Output = '\t Calculating {}th PA'.format(i_Array2D_Parameter_a)
                            print(Str_Output)
            if os.path.isfile(Temp_Str_Cache_Name + '.mat'):
                RawData = io.loadmat(Temp_Str_Cache_Name)
                Array4D_Parameter_a_Time_Varying_Essential = RawData['Array4D_Parameter_a_Time_Varying_Essential']
                Array4D_Parameter_a_Time_Varying_Essential \
                    = numpy.append(Array4D_Parameter_a_Time_Varying_Essential, \
                        Array3D_Parameter_a_Time_Varying_Essential.reshape(Array2D_Parameter_a.shape[0], \
                        Array2D_Parameter_a.shape[1], Array_Time_Selected.size, 1), \
                        axis = 3)
                io.savemat(Temp_Str_Cache_Name, {'Array4D_Parameter_a_Time_Varying_Essential':Array4D_Parameter_a_Time_Varying_Essential})    
            else:
                Array4D_Parameter_a_Time_Varying_Essential \
                    = Array3D_Parameter_a_Time_Varying_Essential.reshape(\
                        Array2D_Parameter_a.shape[0], Array2D_Parameter_a.shape[1], Array_Time_Selected.size, 1)
                io.savemat(Temp_Str_Cache_Name, {'Array4D_Parameter_a_Time_Varying_Essential':Array4D_Parameter_a_Time_Varying_Essential})
        os.chdir('..')
        os.chdir('..')
        # Normalized low frequency random process generation
        Array_Low_Frequency_Random_Process \
            = self.Function_Low_Frequency_Random_Process(Array_Time, Array_Z_Coordinate, Parameter_Stationary_Period)
        Array_Low_Frequency_Random_Process_Normal = Array_Low_Frequency_Random_Process.copy()
        for i_Array2D_Parameter_a in range(Array_Time_Selected.size):
            Int_Start_Index = Array_Index_Time_Selected_Edge[i_Array2D_Parameter_a]
            Int_Ended_Index = Array_Index_Time_Selected_Edge[i_Array2D_Parameter_a + 1]
            Array_Low_Frequency_Random_Process_Normal[Int_Start_Index: Int_Ended_Index] \
                = self.Function_CDF_Mapping_Uniform(Array_Low_Frequency_Random_Process[Int_Start_Index: Int_Ended_Index])
        Array_Low_Frequency_Random_Process_Normal[-1] = 0
        Array_Low_Frequency_Random_Process_Normal_Zero_Mean = Array_Low_Frequency_Random_Process_Normal - 1 / 2
        Array2D_Low_Frequency_Random_Process_With_Power \
            = numpy.zeros((Int_Number_Ploy_Power, Array_Low_Frequency_Random_Process_Normal_Zero_Mean.size))
        for i_Power in range(Int_Number_Ploy_Power):
            Array2D_Low_Frequency_Random_Process_With_Power[i_Power,:] \
                = Array_Low_Frequency_Random_Process_Normal_Zero_Mean**i_Power
        # Instantaneous phase difference generation for different points
        Array2D_Instantaneous_Phase_Differences = numpy.zeros((Int_Number_Points, Array_Time.size))
        for i_Array2D_Parameter_a in range(Array_Time_Selected.size):
            Array2D_Parameter_a = Array3D_Parameter_a_Time_Varying_Essential[:,:,i_Array2D_Parameter_a]
            Int_Start_Index = Array_Index_Time_Selected_Edge[i_Array2D_Parameter_a]
            Int_Ended_Index = Array_Index_Time_Selected_Edge[i_Array2D_Parameter_a + 1]
            for i_Point in range(Int_Number_Points):
                Array2D_Instantaneous_Phase_Differences[i_Point, Int_Start_Index: Int_Ended_Index] \
                    = numpy.dot(Array2D_Parameter_a[i_Point,:].reshape(1, Int_Number_Ploy_Power), \
                                Array2D_Low_Frequency_Random_Process_With_Power[:, Int_Start_Index: Int_Ended_Index])
        # Instantaneous phase for first point
        Array_Instantaneous_Phase_Simulation \
            = self.Function_Instantaneous_Phase_Generation(Array_Instantaneous_Phase_Measurement, Str_Method = 'Original')
        # Add instantaneous phase difference
        Array2D_Instantaneous_Phase_Simulation = numpy.zeros((Int_Number_Points, Array_Time.size))
        for i_Point in range(Int_Number_Points):
            Array2D_Instantaneous_Phase_Simulation[i_Point,:] \
                = Array_Instantaneous_Phase_Simulation + Array2D_Instantaneous_Phase_Differences[i_Point, :]
        Matrix_Instantaneous_Amplitude_Simulation \
            = numpy.ones((Array_Z_Coordinate.shape[0],1)).dot(numpy.matrix(Array_Instantaneous_Amplitude_Measurement))
        Matrix_Simulation_Current_Node \
            = numpy.array(numpy.cos(Array2D_Instantaneous_Phase_Simulation)) \
                * numpy.array(Matrix_Instantaneous_Amplitude_Simulation)
        Time_Ended = time.time()
        if Flag_Debug:
            Str_DateTime_Start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output = 'Node {} Ended\n\t Time: {} \n' .format(i_Current_Node, Str_DateTime_Start)
            print(Str_Output)
            Str_Output = 'Time consumption: {:.3f} \n' .format(Time_Ended - Time_Start)
            print(Str_Output)
        
        Str_Output = 'Finished the calculation for {}th node, time comsumption: {:.3f} s'\
                        .format(i_Current_Node, Time_Ended - Time_Start)
        print(Str_Output)
        return Matrix_Simulation_Current_Node.T, Matrix_Instantaneous_Amplitude_Simulation

    def Function_Simulation_All_Node_Single_Thread_Time_Varying_Coherence(self,
            Array_Signal, Array_Signal_Trend, Array_Signal_Fluctuation, Array_Z_Coordinate, \
            Array_Time, Array_Time_Selected, Array_Index_Time_Selected_Edge, \
            Array_Time_Selected_Edge, Array_Index_Time_Selected, Array2D_Signal_Decomposed, \
            Array2D_Signal_Decomposed_Amplitude, Array2D_Signal_Decomposed_Phase, Array2D_Signal_Decomposed_Frequency, \
            Array_Signal_Decomposed_Center_Frequency, Array4D_TV_Correlation):
        Int_Number_Nodes = Array2D_Signal_Decomposed.shape[1]
        Array3D_Simulation_Decomposed = numpy.zeros((Array_Signal.shape[0], Array_Z_Coordinate.shape[0], Array_Signal_Decomposed_Center_Frequency.shape[0]))
        print('Start_Single_Thread_Computation')
        for i_Current_Node in range(Int_Number_Nodes):
            Array3D_TV_Correlation = Array4D_TV_Correlation[:,:,i_Current_Node,:]
            Array_Instantaneous_Phase_Measurement = Array2D_Signal_Decomposed_Phase[:,i_Current_Node]
            Array_Instantaneous_Amplitude_Measurement = Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node]
            Temp_Tuple_Return_Value = self.Function_Current_Node_Simulation_Decomposed_Non_Gaussian_Mapping_SingleInt\
                                        (Array3D_TV_Correlation, Array_Time_Selected, Array_Index_Time_Selected_Edge, \
                                        Array_Instantaneous_Phase_Measurement, Array_Instantaneous_Amplitude_Measurement, \
                                        Array_Time, Array_Z_Coordinate, i_Current_Node, Parameter_Stationary_Period)
            Array3D_Simulation_Decomposed[:,:,i_Current_Node] = Temp_Tuple_Return_Value[0]
        return Array3D_Simulation_Decomposed, Array4D_TV_Correlation

    def Function_Simulation_All_Node_Multi_Thread_Time_Varying_Coherence(self, \
            Array_Signal, Array_Signal_Trend, Array_Signal_Fluctuation, \
            Array_Z_Coordinate, Array_Time, Array_Time_Selected, Array_Index_Time_Selected_Edge, \
            Array_Time_Selected_Edge, Array_Index_Time_Selected, Array2D_Signal_Decomposed, \
            Array2D_Signal_Decomposed_Amplitude, Array2D_Signal_Decomposed_Phase, Array2D_Signal_Decomposed_Frequency, \
            Array_Signal_Decomposed_Center_Frequency, Array4D_TV_Correlation):
        
        def Function_Current_Node_Simulation_Decomposed_Non_Gaussian_Mapping_SingleInt():
            Tuple_Function_Return \
                = self.Function_Current_Node_Simulation_Decomposed_Non_Gaussian_Mapping_SingleInt(Array3D_TV_Correlation, \
                    Array_Time_Selected, Array_Index_Time_Selected_Edge, \
                    Array_Instantaneous_Phase_Measurement, Array_Instantaneous_Amplitude_Measurement, Array_Time, \
                    Array_Z_Coordinate, i_Current_Node, Parameter_Stationary_Period)
            Array2D_Simulation_Current_Node, Array2D_Instantaneous_Amplitude_Simulation = Tuple_Function_Return
            return Array2D_Simulation_Current_Node, Array2D_Instantaneous_Amplitude_Simulation
        Int_Number_Nodes = Array2D_Signal_Decomposed.shape[1]
        Array3D_Simulation_Decomposed \
            = numpy.zeros((Array_Signal.shape[0], Array_Z_Coordinate.shape[0], \
                            Array_Signal_Decomposed_Center_Frequency.shape[0]))
        Int_Number_of_Cores = cpu_count()
        Pool_Run = Pool(processes = Int_Number_of_Cores)
        Multi_Threads = [None] * Int_Number_of_Cores
        Pool_Result = [None] * Int_Number_of_Cores
        if Flag_Debug:
            print('Start_Parallel_Computation')
        for i_Number_Loop in range(Int_Number_Nodes // Int_Number_of_Cores + 1):
            i_Start_Node = i_Number_Loop * Int_Number_of_Cores
            i_Ended_Node = min((i_Number_Loop + 1) * Int_Number_of_Cores, Int_Number_Nodes)
            i_Thread = int(-1)
            for i_Current_Node in numpy.arange(i_Start_Node, i_Ended_Node, 1): #range(len(Multi_Threads)):
                i_Thread += 1
                Array3D_TV_Correlation = Array4D_TV_Correlation[:,:,i_Current_Node,:]
                Array_Instantaneous_Phase_Measurement = Array2D_Signal_Decomposed_Phase[:,i_Current_Node]
                Array_Instantaneous_Amplitude_Measurement = Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node]
                Multi_Threads[i_Thread] \
                    = Pool_Run.apply_async(Function_Current_Node_Simulation_Decomposed_Non_Gaussian_Mapping_SingleInt, \
                        (Array3D_TV_Correlation, Array_Time_Selected, Array_Index_Time_Selected_Edge, \
                        Array_Instantaneous_Phase_Measurement, Array_Instantaneous_Amplitude_Measurement, Array_Time, \
                        Array_Z_Coordinate, i_Current_Node, Parameter_Stationary_Period))
            i_Thread = int(-1)
            for i_Current_Node in numpy.arange(i_Start_Node, i_Ended_Node, 1): #range(len(Multi_Threads)):
                i_Thread += 1
                Pool_Result[i_Thread] = Multi_Threads[i_Thread].get()
                Array3D_Simulation_Decomposed[:,:,i_Current_Node] = Pool_Result[i_Thread][0]
        Pool_Run.terminate()
        return Array3D_Simulation_Decomposed, Array4D_TV_Correlation
