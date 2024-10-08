"""
Creation data:
    20190318
Modification records:
    20190318
        - Moved from Sub_Class
"""
import numpy
import os

from matplotlib import pyplot
from scipy import io
from scipy import interpolate

PATH_Root_PK_Simulation, PATH_Root_PK_Simulation_Class_SG_File\
     = os.path.split(__file__)
PATH_Root_PK_Simulation = PATH_Root_PK_Simulation[:-13]
print(PATH_Root_PK_Simulation)

class Class_Signal_Generation():
    """
    Class for signal generation including:
        1. Artificiail synthesized
        2. Measurement
    ----------------------------------------------------------------------------
    Funciton list:

    """
    def __init__(self):
        self.Class_Name = 'Class for load both measured and synthesized signals'
        self.Bool_Flag_Debug = False

    def Function_Impulse_Signal_Generation(self, Array_Signal):
        Array_Signal_Impulse = numpy.zeros(Array_Signal.shape)
        Array_Signal_Impulse[Array_Signal_Impulse.size // 2] = 1
        Array_Signal_Impulse = (Array_Signal_Impulse - 1 / Array_Signal.size) \
                                / Array_Signal_Impulse.std() \
                                *  Array_Signal.std()
        Array_Signal_Impulse = Array_Signal_Impulse + Array_Signal.mean()
        return Array_Signal_Impulse

    def Function_Impulse_Signal_Generation_Oscilate_Delta(self, \
                    Array_Time_Predict, \
                    Array_Signal_Predict, \
                    Int_Length_Prediction):
        Array_Signal \
            = Array_Signal_Predict\
                [Int_Length_Prediction \
                    : Array_Signal_Predict.size - Int_Length_Prediction]
        Array_Impulse_Signal_Predict \
            = numpy.zeros(Array_Time_Predict.size)
        Array_Impulse_Signal_Predict[Array_Time_Predict.size // 2] = 1
        Array_Impulse_Signal_Predict[Array_Time_Predict.size // 2 - 1] = - 1
        Array_Impulse_Signal_Predict \
            = Array_Impulse_Signal_Predict / numpy.sqrt(2) \
                * numpy.sqrt(numpy.sum((Array_Signal - Array_Signal.mean())**2))
        Array_Impulse_Signal_Predict \
            = Array_Impulse_Signal_Predict + Array_Signal.mean()
        Array_Impulse_Signal \
            = Array_Impulse_Signal_Predict\
                [Int_Length_Prediction \
                    : Array_Time_Predict.size - Int_Length_Prediction]       
        return Array_Impulse_Signal, Array_Impulse_Signal_Predict

    def Function_Remove_NAN_Value(self, Array_Time, Array_Signal):
        Array_Index_NAN = numpy.argwhere(~numpy.isnan(Array_Signal)).reshape(-1)
        if Array_Index_NAN[0] == 0:
            Array_Signal[0] = 0
        if Array_Index_NAN[-1] == Array_Signal.size:
            Array_Signal[-1] = 0
        Function_interpolate_1d \
            = interpolate.interp1d(Array_Time[Array_Index_NAN], \
                                    Array_Signal[Array_Index_NAN], \
                                    kind = 'linear')
        self.Array_Signal_Remove_NAN = Function_interpolate_1d(Array_Time)
        return self.Array_Signal_Remove_NAN

    def Function_Extract_Data_of_Delong_Zuo(self, \
            i_Experiment_Time, i_Group, i_Col_Signal):
        """
        Load raw data from mat file:
            This part of code was rewritten to avoid the "exec" commande, 
            which caused multiple problems including the
            the problem of hidden in the main variable space and pylint error.
        ------------------------------------------------------------------------
        Input:
            i_Experiment_Time: 
                - Totally 10 experiments
            i_Group:
                - Each experiment has 4 group of sensors
            i_Col_Signal:
                - Groups 1-3 -> 30 Columns
                - Group 4 -> 10 Columns
        ------------------------------------------------------------------------
        Output:
            Array_Time
            Array_Signal
        """
        Str_Data_Path = PATH_Root_PK_Simulation + '/Data/DataBase_Measurements/Thunderstorm_Zuo_Delong.mat'
        RawData = io.loadmat(Str_Data_Path)
        Struct_List_File = RawData['Struct_List_File']
        Array_File_Name = []
        List_Data = []
        for i_File in range(Struct_List_File.size):
            Str_File_Name = Struct_List_File[i_File,0][0][0]
            if Str_File_Name[-3:] == 'csv':
                Array_File_Name \
                    = numpy.append(\
                            Array_File_Name, \
                            numpy.squeeze(Struct_List_File[i_File,0][0][0]))
                List_Data.append(numpy.squeeze(Struct_List_File[i_File,0][6]))
        # Extract specified signal from the file
        i_File = i_Experiment_Time * 4 + i_Group
        Value_Sampling_Interval = 0.02
        Str_File_Name = Array_File_Name[i_File]
        Array_Signal = List_Data[i_File][:,i_Col_Signal]
        Array_Time = numpy.arange(0, Value_Sampling_Interval * Array_Signal.size, Value_Sampling_Interval)
        return Array_Time, Array_Signal

    def Function_Load_Measurements(self, Str_File_Name):
        """
        Input:
            Str_File_Name
        """
        # Check the Str_File_Name
        if Str_File_Name[-4:] != '.mat':
            Str_File_Name = Str_File_Name + '.mat'
        RawData = io.loadmat(PATH_Root_PK_Simulation + '/Data/DataBase_Measurements/' + Str_File_Name)
        self.Dict_Keys = RawData.keys()
        if Str_File_Name == 'Haikui.mat':
            self.Array2D_Wind = RawData['SF040100']
            self.Value_Delta_Time = 3600 * 24 / self.Array2D_Wind.shape[0]
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
        if Str_File_Name == 'RFDdata.mat':
            self.Array2D_Wind = RawData['matrixtot']
            self.Value_Delta_Time = 1
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
            Int_Number_Interpolate_Times = 20
            self.Value_Delta_Time_New = self.Value_Delta_Time / Int_Number_Interpolate_Times

            Array_Time_New = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0],
                                        (self.Array2D_Wind.shape[0] - 1) * Int_Number_Interpolate_Times + 1)

            self.Array2D_Wind_New = numpy.zeros((Array_Time_New.size, self.Array2D_Wind.shape[1]))
            for i_Signal in range(self.Array2D_Wind.shape[1]):
                Array_Wind_Interp = self.Array2D_Wind[:, i_Signal]
                Function_interpolate_1d = interpolate.interp1d(Array_Time, Array_Wind_Interp, kind='linear')
                self.Array2D_Wind_New[:, i_Signal] = Function_interpolate_1d(Array_Time_New)
            Array_Time = Array_Time_New.copy()
            self.Array2D_Wind = self.Array2D_Wind_New.copy()
        if Str_File_Name == 'Thunderstorm_GS_SP_2_20120411.mat':
            self.Array2D_Wind = numpy.zeros((36000,4))
            self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
            self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
            self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
            self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
            self.Value_Delta_Time = 0.1
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
        if Str_File_Name == 'Thunderstorm_GS_SP_2_20110605_1450.mat':
            self.Array2D_Wind = numpy.zeros((36000,4))
            self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
            self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
            self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
            self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
            self.Value_Delta_Time = 0.1
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
        if Str_File_Name == 'Thunderstorm_GS_SP_2_20120411_0720.mat':
            self.Array2D_Wind = numpy.zeros((36000,4))
            self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
            self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
            self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
            self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
            self.Value_Delta_Time = 0.1
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
        if Str_File_Name == 'Thunderstorm_GS_SP_20120419_1250.mat':
            self.Array2D_Wind = numpy.zeros((36000,4))
            self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
            self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
            self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
            self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
            self.Value_Delta_Time = 0.1
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
        if Str_File_Name == 'Thunderstorm_GS_SP_3_20111025_1540.mat':
            self.Array2D_Wind = numpy.zeros((36000,4))
            self.Array2D_Wind[:, 0] = numpy.squeeze(RawData['Ufin'])
            self.Array2D_Wind[:, 1] = numpy.squeeze(RawData['Vfin'])
            self.Array2D_Wind[:, 2] = numpy.squeeze(RawData['modulo'])
            self.Array2D_Wind[:, 3] = numpy.squeeze(RawData['dir'])
            self.Value_Delta_Time = 0.1
            Array_Time = numpy.linspace(self.Value_Delta_Time, self.Value_Delta_Time * self.Array2D_Wind.shape[0], self.Array2D_Wind.shape[0])
        for i_Col in range(self.Array2D_Wind.shape[1]):
            self.Array2D_Wind[:,i_Col] = self.Function_Remove_NAN_Value(Array_Time, self.Array2D_Wind[:,i_Col])
        return self.Array2D_Wind, self.Value_Delta_Time, self.Dict_Keys

    def Function_Measurment_Description(self, Str_File_Name):
        Array2D_Wind, Value_Delta_Time, Dict_Keys = self.Function_Load_Measurements(Str_File_Name)
        Array_Time = numpy.linspace(Value_Delta_Time, Value_Delta_Time * Array2D_Wind.shape[0], Array2D_Wind.shape[0])
        if Str_File_Name == 'Haikui.mat':
            pyplot.figure(figsize = (12,8), dpi = 100, facecolor = 'white')
            # for i_Level in range(Array2D_Signal_Empirical_Mode_Decomposition.shape[1]):
            pyplot.subplot(221)
            i_Col = 0
            pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 0')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel('Wind speed (m/s)')
            pyplot.xticks(numpy.arange(0,25,2))
            pyplot.grid('on')
            pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.tight_layout()
            pyplot.subplot(222)
            i_Col = 1
            pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 1')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel(r'Wind direction ($^\circ$)')
            pyplot.xticks(numpy.arange(0,25,2))
            pyplot.grid('on')
            pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.tight_layout()
            pyplot.subplot(223)
            i_Col = 2
            pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 2')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel('Unknown Property')
            pyplot.xticks(numpy.arange(0,25,2))
            pyplot.grid('on')
            pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.tight_layout()
            pyplot.subplot(224)
            i_Col = 3
            pyplot.plot(Array_Time / 3600, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 3')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel('Unknown Property')
            pyplot.xticks(numpy.arange(0,25,2))
            pyplot.grid('on')
            pyplot.axis([Array_Time.min() / 3600, Array_Time.max() / 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.tight_layout()
            pyplot.show()
        if Str_File_Name == 'RFDdata.mat':
            List_Keys = list(Dict_Keys)
            pyplot.figure(figsize = ( 6 * 2, 2 * 5), dpi = 100, facecolor = 'white')
            for i_Keys in range(Array2D_Wind.shape[1]):
                pyplot.subplot2grid((5,2), (i_Keys // 2, i_Keys % 2))
                pyplot.plot(Array_Time, Array2D_Wind[:,1], color = 'black')
                pyplot.grid('on')
                pyplot.title(List_Keys[i_Keys + 4])
                pyplot.xlabel('Time (s)')
                pyplot.ylabel('Wind speed (s)')
                pyplot.axis([0, 1800, 0, 45])
                pyplot.tight_layout()
            pyplot.show()
        if Str_File_Name == 'Thunderstorm_GS_SP_2_20120411.mat':
            List_Keys = list(Dict_Keys)
            pyplot.figure(figsize = (12,8), dpi = 100, facecolor = 'white')
            # for i_Level in range(Array2D_Signal_Empirical_Mode_Decomposition.shape[1]):
            pyplot.subplot(221)
            i_Col = 0
            pyplot.plot(Array_Time , Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 0')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel('Wind speed (m/s)')
            pyplot.xticks(numpy.arange(0, 3800, 600))
            pyplot.grid('on')
            pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.title(List_Keys[i_Col + 3])
            pyplot.tight_layout()
            pyplot.subplot(222)
            i_Col = 1
            pyplot.plot(Array_Time , Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 1')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel(r'Wind direction ($^\circ$)')
            pyplot.xticks(numpy.arange(0, 3800, 600))
            pyplot.grid('on')
            pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.title(List_Keys[i_Col + 3])
            pyplot.tight_layout()
            pyplot.subplot(223)
            i_Col = 2
            pyplot.plot(Array_Time , Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 2')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel('Unknown Property')
            pyplot.xticks(numpy.arange(0, 3800, 600))
            pyplot.grid('on')
            pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.title(List_Keys[i_Col + 3])
            pyplot.tight_layout()
            pyplot.subplot(224)
            i_Col = 3
            pyplot.plot(Array_Time, Array2D_Wind[:,i_Col], linestyle = '-', color = 'black', linewidth = 0.5, label = 'Wind Speed 3')
            pyplot.xlabel('Time (h)')
            pyplot.ylabel('Unknown Property')
            pyplot.xticks(numpy.arange(0, 3800, 600))
            pyplot.grid('on')
            pyplot.axis([0, 3600, Array2D_Wind[:,i_Col].min(), Array2D_Wind[:,i_Col].max()])
            pyplot.title(List_Keys[i_Col + 3])
            pyplot.tight_layout()
            pyplot.show()

    def Function_Load_Wind_Speed(self, Str_Signal_Name):
        """
        This function loads the measurments and converts them with specified 
        parameters
        ------------------------------------------------------------------------
        Input:
            Str_Signal_Name
                1. Measured_Haikui_0
                2. hunderstorm_GS_SP_2_20120411
        ------------------------------------------------------------------------
        Output:
            Array_Time
            Array_Signal
            Value_Sampling_Frequency
            Value_Delta_Time
        """
        Array_Time = numpy.array([])
        Array_Signal = numpy.array([])
        if Str_Signal_Name == 'Measured_Haikui_0':
            Str_File_Name = 'Haikui.mat'
            Tuple_Function_Return \
                = self.Function_Load_Measurements(Str_File_Name)
            Array2D_Wind, Value_Delta_Time \
                = Tuple_Function_Return[0:2]
            Data_Matrix_Signal_Raw = Array2D_Wind[:,0]
            Temp_Array_x5 \
                = Data_Matrix_Signal_Raw\
                    [51200 - 1:Data_Matrix_Signal_Raw.shape[0]]
            Array_Signal \
                = Temp_Array_x5[1001 - 1: 1001 - 1 + 1800]
            Value_Sampling_Frequency = int(1)
            Array_Time \
                = numpy.linspace\
                    (1, Array_Signal.shape[0] * 1 / Value_Sampling_Frequency, \
                        Array_Signal.shape[0])
        elif Str_Signal_Name == 'Thunderstorm_GS_SP_2_20120411':
            Str_File_Name = 'Thunderstorm_GS_SP_2_20120411.mat'
            Tuple_Function_Return \
                = self\
                    .Function_Load_Measurements(Str_File_Name)
            Array2D_Wind, Value_Delta_Time = Tuple_Function_Return[0:2]
            Array_Signal = Array2D_Wind[:,0]
            Array_Time \
                = numpy.linspace(\
                            0, \
                            Value_Delta_Time * (Array_Signal.size - 1), \
                            Array_Signal.size)
            Value_Sampling_Frequency = int(1 / Value_Delta_Time)
        if Array_Time.size == 0:
            print('Error: The input signal name is not defined, returning NULL')
        return Array_Time, Array_Signal, Value_Sampling_Frequency, Value_Delta_Time

    def Function_Signal_Generation(self, Str_Signal_Name):
        """
        Output:
            Array_Time
            Array_Signal
        """
        Array_Time = numpy.array([])
        Array_Signal = numpy.array([])
        if Str_Signal_Name == 'Amplitude_Modulated_Signal_0':
            Array_Time = numpy.linspace(0, 273, 2731)
            Value_M = 1
            Value_f_m = 0.005  # Hz
            Value_Main_Frequency = 2  # Hz
            Array_Signal = numpy.sin(
                Array_Time * Value_Main_Frequency * 2 * numpy.pi)
            Array_Signal[1000:2000] = numpy.sin(Array_Time[1000:2000] * Value_Main_Frequency * 2 * numpy.pi) \
                                        * (1 + 0 * Value_M * numpy.sin((Array_Time[1000:2000] - 100) * 2 * numpy.pi * Value_f_m))
            Array_Signal = Array_Signal - Array_Signal.mean()
        elif Str_Signal_Name == 'Composite_Signal_1':
            Value_f11 = 1
            Value_f12 = 2
            Value_f2 = 3
            Value_omega_11 = 2 * numpy.pi * Value_f11
            Value_omega_12 = 2 * numpy.pi * Value_f12
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 30:
                    Array_Signal[i_Time] = numpy.sin(Array_Time[i_Time] * Value_omega_11) + numpy.sin(Array_Time[i_Time] \
                                            * Value_omega_2)
                elif Array_Time[i_Time] <=70:
                    Array_Signal[i_Time] = numpy.sin(Array_Time[i_Time] * Value_omega_11 \
                                            + (Array_Time[i_Time] - 30)**2 / 80 ) \
                                            + numpy.sin(Array_Time[i_Time] * Value_omega_2)
                else:
                    Array_Signal[i_Time] = numpy.sin(Array_Time[i_Time] * Value_omega_12) \
                                            + numpy.sin(Array_Time[i_Time] * Value_omega_2)
            Array_Signal = Array_Signal - Array_Signal.mean()
        elif Str_Signal_Name == 'Composite_Signal_AM_cos_and_cos':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.cos(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.cos(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 2 - numpy.abs(Array_Time[i_Time] - 50) / 5 
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 2 - numpy.abs(Array_Time[i_Time] - 50) / 5 
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a0':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 1
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a1':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 1 + 1 - numpy.abs(Array_Time[i_Time] - 50) / 5 
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a2':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 1 + 2 * (1 - numpy.abs(Array_Time[i_Time] - 50) / 5)
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a3':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 1 + 3 * (1 - numpy.abs(Array_Time[i_Time] - 50) / 5)
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Composite_Signal_AM_sin_and_sin_a4':
            Value_f1 = 1
            Value_f2 = 0.2
            Value_omega_1 = 2 * numpy.pi * Value_f1
            Value_omega_2 = 2 * numpy.pi * Value_f2
            Array_Time = numpy.linspace(0, 100, 1001)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Base_Signal_1 = numpy.sin(Array_Time * Value_omega_1)
            Array_Base_Signal_2 = numpy.sin(Array_Time * Value_omega_2)
            Array_Amplitude_Signal_1 = numpy.zeros(Array_Time.size)
            for i_Time in range(Array_Time.size):
                if Array_Time[i_Time] <= 55 and Array_Time[i_Time] >= 45:
                    Array_Amplitude_Signal_1[i_Time] \
                        = 1 + 4 * (1 - numpy.abs(Array_Time[i_Time] - 50) / 5)
                else:
                    Array_Amplitude_Signal_1[i_Time] = 1
            Temp_Array_Sub_Signal_1 \
                = Array_Base_Signal_1 * Array_Amplitude_Signal_1
            Temp_Array_Sub_Signal_2 = Array_Base_Signal_2
            Array_Signal = Temp_Array_Sub_Signal_1 + Temp_Array_Sub_Signal_2
            Array2D_Accessory_Signal = numpy.zeros([Array_Time.size, 2])
            Array2D_Accessory_Signal[:,0] = Temp_Array_Sub_Signal_1
            Array2D_Accessory_Signal[:,1] = Temp_Array_Sub_Signal_2
        elif Str_Signal_Name == 'Duffin_Type_Wave':
            Value_omega = numpy.pi * 2 / 100
            Array_Time = numpy.linspace(0,1000,1001)
            Array_Signal = numpy.sin(Array_Time * Value_omega + 0.3 * numpy.sin(2 * Value_omega * Array_Time) )
            Array_Signal = Array_Signal - Array_Signal.mean()
        elif Str_Signal_Name == 'Frequency_Modulated_Signal':
            Array_Time = numpy.linspace(0,273, 2731)
            Value_f_Delta = 0.1
            Value_f_m = 0.05
            Array_Signal = numpy.sin(Array_Time * 3 * 2 * numpy.pi)
            Array_Signal[1000:2000] = numpy.sin(Array_Time[1000:2000] * 3 * 2 * numpy.pi + Value_f_Delta / Value_f_m * numpy.sin(2 * numpy.pi * Value_f_m * (Array_Time[1000:2000]-100)))
        elif Str_Signal_Name == 'Impulse_Signal':
            Array_Time = numpy.linspace(0,273, 2731)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Signal[int(Array_Signal.size / 2)] = 1
        elif Str_Signal_Name == 'Impulse_Signal_0': 
            # The suffix 0 means that it has the same property with the original impulse signal generation function
            Array_Time = numpy.linspace(0,273, 2731)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Signal[int(Array_Signal.size / 2)] = 1
        elif Str_Signal_Name == 'Swept_Sine_Signal':
            Array_Time = numpy.linspace(0,273, 273 * 10 + 1)
            Array_Signal = numpy.sin(Array_Time * 1 * 2 * numpy.pi + 1 / 273 * Array_Time**2 * 2 * numpy.pi)
        elif Str_Signal_Name == 'Swept_Sine_Signal_Low_Frequency':
            Array_Time = numpy.linspace(0,273, 273 * 10 + 1)
            Value_Frequency = 0.4
            Array_Signal = numpy.sin(Array_Time * 1 * Value_Frequency * numpy.pi + 1 / 273 * Array_Time**2 * Value_Frequency * numpy.pi)
        elif Str_Signal_Name == 'Swept_Sine_Signal_Multiple_Trend':
            Array_Time = numpy.linspace(-100, 400, 500 * 10 + 1)
            Array_Signal = numpy.zeros(Array_Time.shape)
            Array_Signal[0:1000] = numpy.sin(Array_Time[0:1000] * 1 * 2 * numpy.pi - 1 / 273 * Array_Time[0:1000]**2 * 2 * numpy.pi + (200 + 1 / 273 * 100**2) * 2 * numpy.pi)
            Array_Signal[1000:1000 + 273 * 10 + 1] = numpy.sin(Array_Time[1000:1000 + 273 * 10 + 1] * 1 * 2 * numpy.pi + 1 / 273 * Array_Time[1000:1000 + 273 * 10 + 1]**2 * 2 * numpy.pi + (200 + 100**2 / 273 ) * 2 * numpy.pi)
            Array_Signal[1000 + 273 * 10 + 1:] = numpy.sin(Array_Time[1000 + 273 * 10 + 1:] * 2 * 3 * numpy.pi - 1 / 273 * (Array_Time[1000 + 273 * 10 + 1:] - 273)**2 * 2 * numpy.pi + (200 + 100**2 / 273 - 273) * 2 * numpy.pi)
        elif Str_Signal_Name == 'Stationary_Signal_0':
            Array_Time = numpy.linspace(0,273, 2731)
            Array_Signal = numpy.sin(Array_Time * 2.5 * 2 * numpy.pi) #+ numpy.sin(Array_Time * 3.2 * 2 * numpy.pi) + numpy.sin(Array_Time * 4.2 * 2 * numpy.pi)
        elif Str_Signal_Name == 'White_Noise_Signal_0': # White gaussian noise with unity std and zero mean
            Array_Time = numpy.linspace(0, 273, 2731)
            Array_Signal = numpy.random.randn(Array_Time.size)
            Array_Signal = Array_Signal / Array_Signal.std()
            Array_Signal = Array_Signal  - Array_Signal.mean()
            Array_Signal = Array_Signal + 0
        elif Str_Signal_Name == 'White_Noise_Signal_Cut_0': # White gaussian noise with unity std and zero mean
            Array_Time = numpy.linspace(0, 273, 2731)
            Array_Signal = numpy.random.randn(Array_Time.size)
            Array_Signal = Array_Signal / Array_Signal.std()
            Array_Signal = Array_Signal  - Array_Signal.mean()
            Array_Signal = Array_Signal
        # Error report
        if Array_Time.size == 0:
            print('Error: The input signal name is not defined, returning NULL')
        self.Array_Time = Array_Time
        self.Array_Signal = Array_Signal
        if 'Array2D_Accessory_Signal' not in locals():
            Array2D_Accessory_Signal = 0
        return self.Array_Time, self.Array_Signal, Array2D_Accessory_Signal

    def Function_Vertical_Profile(self, Array_Z_Coordinate):
        Array_Vertical_Profile = numpy.zeros(Array_Z_Coordinate.shape)
        Parameter_V_max = 10
        Parameter_b1 = -0.22
        Parameter_b2 = -2.75
        Parameter_z_max = 60.35
        for i_Height in range(Array_Z_Coordinate.size):
            Value_Height = Array_Z_Coordinate[i_Height]
            Array_Vertical_Profile[i_Height] \
                = 1.354 * Parameter_V_max \
                    * ((numpy.exp(Value_Height * Parameter_b1 / Parameter_z_max) \
                        - numpy.exp(Value_Height * Parameter_b2 / Parameter_z_max)))
        Array_Vertical_Profile = Array_Vertical_Profile / Array_Vertical_Profile[0]
        return Array_Vertical_Profile