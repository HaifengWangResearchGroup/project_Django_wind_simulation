"""
Creation date: 20190127:
    Extracted from the "Sub_Class" file for further use

Updates:
    20190325: 
        - Changed from Array_Z_Coordinate to Array2D_Locations
        - Rename class: 
            From: Class_Simulation_Hilbert_wavelet_POD_Only 
            To: Class_Simulation_Hilbert_wavelet_TV_POD
        - Rename class: 
            From: Class_Simulation_Hilbert_wavelet_POD 
            To: Class_Simulation_Hilbert_wavelet_TV_2DSVD
        - Rename function:
            From: Function_CDU_2DSVD_Eigendecomposition
            To: Function_CDU_Corr_Decomposition
            From: Function_Simulation_FM_Part_POD_Based_TV_Corr_2DSVD
            To: Function_Simulation_FM_Part_POD_Based
        - Rename Variable:
            From: Array3D_2DSVD_Eigenvector
            To: Array3D_Eigen_Vector
            From: Array3D_2DSVD_Sigma
            To: Array3D_Eigen_Sigma
        - Commented out Class:
            Class_Simulation_Hilbert_wavelet_CT_SVD
                Reason: Cannot remember why it is here since CT_SVD is CT_POD
    2019_03_28:
        - Input parameter:
            From: Verticall locations and vertical height
            To: Array2D_Locations
        - Input of Function_CDU_Target_Spatial_Correlation:
            From: Array2D_Locations, Str_Correlation_Model
            To: Str_Correlation_Model
"""

import numpy
import pywt
import time

from multiprocessing import cpu_count
from multiprocessing import Pool

from .Sub_Class_Signal_Processing import Class_Signal_Processing
from .Sub_Class_Wind_Field_Information import Class_Wind_Field_Information

Object_Signal_Processing = Class_Signal_Processing()
Object_Wind_Field_Information = Class_Wind_Field_Information()

class Class_Simulation_Hilbert_wavelet_TV_2DSVD():
    """
    Single point simulation:
        Hilbert-based method
    Multi point simulation:
        POD-based method
    ----------------------------------------------------------------------------
    !!! Parameter_Int_SVD_Truncation = 0!!!
        The automatic selection or manual selection of SVD truncation need to
        be achieved and validated.
    """
    def __init__(self, \
            Array2D_Locations, 
            Parameter_Str_IF_Method, \
            Parameter_Str_Wavelet_Name, \
            Parameter_Int_SVD_Truncation, \
            Parameter_Int_Max_WPT_Level, \
            Parameter_Str_Method_for_Obtaining_Std):
        #--Info-----------------------------------------------------------------
        self.Class_Name = 'Class of Hilbert-wavelet-based simulation'
        self.Bool_Flag_Debug = False
        #--Parameter------------------------------------------------------------
        self.Parameter_Total_Height \
            = Array2D_Locations[:,2].max() - Array2D_Locations[:,2].min() # Unit: m
        self.Parameter_Int_Number_Points = Array2D_Locations.shape[0]
        self.Array2D_Locations = Array2D_Locations
        # IF generation method
        self.Parameter_Str_IF_Method = Parameter_Str_IF_Method
        self.Parameter_Str_Wavelet_Name = Parameter_Str_Wavelet_Name
        self.Parameter_Int_Max_WPT_Level = Parameter_Int_Max_WPT_Level
        self.Parameter_Str_Method_for_Obtaining_Std \
                = Parameter_Str_Method_for_Obtaining_Std
        if Parameter_Int_SVD_Truncation == 0:
            self.Parameter_Int_SVD_Truncation = self.Parameter_Int_Number_Points
        else:
            self.Parameter_Int_SVD_Truncation = Parameter_Int_SVD_Truncation
        #--Data-Initiate--------------------------------------------------------
        self.Array_Signal = None
        self.Array_Time = None
        self.Array2D_Signal_Decomposed = None
        self.Array2D_Signal_Decomposed_Amplitude = None
        self.Array2D_Signal_Decomposed_Phase = None
        self.Array2D_Signal_Decomposed_Frequency = None
        self.Array_Center_Frequency = None
        self.Array2D_Simulation = None # [i_Time, i_Point]
        self.Array3D_Simulation_Decomposed = None # [i_Time, i_Point, i_Scale]
        self.Array3D_Eigen_Vector = None # [i_Point, i_SVD_Mode, i_Scale]
        self.Array3D_Eigen_Sigma = None # [i_Time, i_SVD_Mode, i_Scale]
        self.Array_Time_Varying_Standard_Deviation = None
        self.Array4D_Correlation = None  # [i_Point, i_Point, i_Scale, i_Time]

    #--Class-Data-Update--------------------------------------------------------
    def Function_CDU_Input_Data(self, Array_Time, Array_Signal):
        """
        Initiate the object with input measurement data
        """
        self.Array_Signal = Array_Signal
        self.Array_Time = Array_Time

    def Function_CDU_Decomposition_IA_IF(self):
        """
        Decompose the original data to:
            self.Array2D_Signal_Decomposed
            self.Array2D_Signal_Decomposed_Amplitude
            self.Array2D_Signal_Decomposed_Phase
            self.Array2D_Signal_Decomposed_Frequency
            self.Array_Center_Frequency
        ------------------------------------------------------------------------
        Data source;
            self.Array_Time
            self.Array_Signal
        ------------------------------------------------------------------------
        Data update:
        """
        Tuple_Function_Return\
            = Object_Signal_Processing\
                .Function_WPT_HT_Decomposition\
                    (self.Array_Time, \
                    self.Array_Signal, \
                    self.Parameter_Str_Wavelet_Name, \
                    self.Parameter_Int_Max_WPT_Level, 'All')
        self.Array2D_Signal_Decomposed = Tuple_Function_Return[0]
        self.Array2D_Signal_Decomposed_Amplitude = Tuple_Function_Return[1]
        self.Array2D_Signal_Decomposed_Phase = Tuple_Function_Return[2]
        self.Array2D_Signal_Decomposed_Frequency = Tuple_Function_Return[3]
        self.Array_Center_Frequency = Tuple_Function_Return[4]

    def Function_CDU_Target_Spatial_Correlation(self, \
            Str_Correlation_Model):
        """
        Description:
            2D Array: Time-invariant spatial correlation for all scales
            3D Array: Time-invariant for multi-scale correlation
            4D Array: Time-varying spatial corrrlation for various scales
        ------------------------------------------------------------------------
        Input:
            Str_Correlation_Model:
                - 'Jiang_2017'
        """
        if self.Parameter_Int_Max_WPT_Level == 0:
            WPT_Signal \
                = pywt.WaveletPacket(\
                    self.Array_Signal, \
                    self.Parameter_Str_Wavelet_Name, \
                    'symmetric')
            Int_Max_WPT_Level = WPT_Signal.maxlevel
        else:
            Int_Max_WPT_Level = self.Parameter_Int_Max_WPT_Level
        Array_Signal_Fluctuation \
                = self.Array2D_Signal_Decomposed[:,1:].sum(axis = 1)
        Array_Time_Varying_Standard_Deviation \
            = Object_Signal_Processing\
                .Function_Time_Varying_Standard_Deviation\
                    (self.Array_Time, \
                    Array_Signal_Fluctuation, \
                    2**Int_Max_WPT_Level, \
                    self.Parameter_Str_Method_for_Obtaining_Std)
        self.Array_Time_Varying_Standard_Deviation \
            = Object_Signal_Processing\
                .Function_Remove_Trend(Array_Time_Varying_Standard_Deviation, \
                                        self.Parameter_Str_Wavelet_Name)[0]
        if Str_Correlation_Model == 'Jiang_2017':
            Tuple_Function_Return = Object_Wind_Field_Information\
                .Function_Correlation_Generation_Jiang_2017(\
                    self.Array2D_Locations, \
                    self.Array_Center_Frequency, \
                    self.Array_Time, \
                    self.Array_Time_Varying_Standard_Deviation, \
                    self.Array_Time_Varying_Standard_Deviation.mean())
        else:
            print('Error: Function_CDU_Target_Spatial_Correlation')
            print('\t Specified correlaton model not encoded')
        Array4D_TV_Correlation = Tuple_Function_Return[0]
        self.Array4D_Correlation = Array4D_TV_Correlation

    def Function_CDU_Corr_Decomposition(self):
        """
        Description:
            Decompose all time-varying spatial correlation
        """
        if self.Bool_Flag_Debug:
            Value_Start_Time = time.time()
        Array3D_Eigen_Vector \
            = numpy.zeros([self.Parameter_Int_SVD_Truncation, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size])
        Array3D_Eigen_Sigma \
            = numpy.zeros([self.Array_Time.size, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size])
        for i_Scale in range(self.Array_Center_Frequency.size):
            Tuple_Function_Return\
                = self.Function_2DSVD_of_TV_Corr_Matrix(i_Scale)
            Array2D_Eig_Vecto, Array2D_Sigma = Tuple_Function_Return
            Array3D_Eigen_Vector[:,:,i_Scale] = Array2D_Eig_Vecto
            Array3D_Eigen_Sigma[:,:,i_Scale] = Array2D_Sigma
        self.Array3D_Eigen_Vector = Array3D_Eigen_Vector
        self.Array3D_Eigen_Sigma = Array3D_Eigen_Sigma
        if self.Bool_Flag_Debug:
            Value_Time_Comsumption = time.time() - Value_Start_Time
            print('Debug:')
            print('\tClass:', self.Class_Name)
            print('\tFunction: Function_CDU_Corr_Decomposition - SVD')
            print('\tTime consumption: {:.4f}'.format(Value_Time_Comsumption))

    def Function_CDU_Corr_Decomposition_Savememory(self, \
            Array2D_Locations):
        """
        Description:
            Combines 
                - Function_CDU_Target_Spatial_Correlation
                - Function_CDU_Corr_Decomposition
                - Function_2DSVD_of_TV_Corr_Matrix
            to reduce memory usage
        """
        if self.Parameter_Int_Max_WPT_Level == 0:
            WPT_Signal \
                = pywt.WaveletPacket(\
                    self.Array_Signal, \
                    self.Parameter_Str_Wavelet_Name, \
                    'symmetric')
            Int_Max_WPT_Level = WPT_Signal.maxlevel
        else:
            Int_Max_WPT_Level = self.Parameter_Int_Max_WPT_Level
        Array_Signal_Fluctuation \
                = self.Array2D_Signal_Decomposed[:,1:].sum(axis = 1)
        Array_Time_Varying_Standard_Deviation \
            = Object_Signal_Processing\
                .Function_Time_Varying_Standard_Deviation\
                    (self.Array_Time, \
                    Array_Signal_Fluctuation, \
                    2**Int_Max_WPT_Level, \
                    self.Parameter_Str_Method_for_Obtaining_Std)
        self.Array_Time_Varying_Standard_Deviation \
            = Object_Signal_Processing\
                .Function_Remove_Trend(Array_Time_Varying_Standard_Deviation, \
                                        self.Parameter_Str_Wavelet_Name)[0]
        Temp_Value_Mean = self.Array_Time_Varying_Standard_Deviation.mean()
        def Sub_Function_Get_Current_Time_Corr(i_Time,i_Scale):                                        
            Tuple_Function_Return = Object_Wind_Field_Information\
                .Function_Correlation_Generation_Jiang_2017\
                    (Array2D_Locations, \
                    self.Array_Center_Frequency[i_Scale:i_Scale+1], \
                    self.Array_Time[i_Time: i_Time + 1], \
                    self.Array_Time_Varying_Standard_Deviation\
                            [i_Time: i_Time + 1], \
                    Temp_Value_Mean)
            Array2D_Curr_Correlation = Tuple_Function_Return[0][:,:,0,0]
            return Array2D_Curr_Correlation

        if self.Bool_Flag_Debug:
            Value_Start_Time = time.time()
        Array3D_Eigen_Vector \
            = numpy.zeros([self.Parameter_Int_SVD_Truncation, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size])
        Array3D_Eigen_Sigma \
            = numpy.zeros([self.Array_Time.size, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size])
        for i_Scale in range(self.Array_Center_Frequency.size):
            Array2D_U_SUM \
                = numpy.zeros([self.Parameter_Int_Number_Points, \
                                self.Parameter_Int_Number_Points])
            # Array2D_V_Sum is not calculated since U and V are the tranpose of
            # each other.
            for i_Time in range(self.Array_Time.size):
                Array2D_Snapshot_i \
                    = Sub_Function_Get_Current_Time_Corr(i_Time, i_Scale)
                Array2D_U_SUM \
                    += Array2D_Snapshot_i.T.dot(Array2D_Snapshot_i)
            Array2D_Eig_Vecto = numpy.linalg.eig(Array2D_U_SUM)[1]

            Array3D_Sigma \
                    = numpy.zeros([self.Parameter_Int_Number_Points, \
                                    self.Parameter_Int_Number_Points, \
                                    self.Array_Time.size])
            Array2D_Sigma = numpy.zeros([self.Parameter_Int_Number_Points, \
                                            self.Array_Time.size])
            for i_Time in range(self.Array_Time.size):
                Array2D_Snapshot_i \
                    = Sub_Function_Get_Current_Time_Corr(i_Time, i_Scale)
                Array3D_Sigma[:,:,i_Time] \
                    = Array2D_Eig_Vecto.T\
                        .dot(Array2D_Snapshot_i)\
                        .dot(Array2D_Eig_Vecto)
                Array2D_Sigma[:, i_Time] = numpy.diag(Array3D_Sigma[:,:,i_Time])
            Array2D_Sigma \
                = Array2D_Sigma[:self.Parameter_Int_SVD_Truncation,:].T

            Array3D_Eigen_Vector[:,:,i_Scale] = Array2D_Eig_Vecto
            Array3D_Eigen_Sigma[:,:,i_Scale] = Array2D_Sigma
        self.Array3D_Eigen_Vector = Array3D_Eigen_Vector
        self.Array3D_Eigen_Sigma = Array3D_Eigen_Sigma
        if self.Bool_Flag_Debug:
            Value_Time_Comsumption = time.time() - Value_Start_Time
            print('Debug:')
            print('\tClass:', self.Class_Name)
            print('\tFunction: Function_CDU_Corr_Decomposition - SVD')
            print('\tTime consumption: {:.4f}'.format(Value_Time_Comsumption))

    def Function_CDU_Simulation_Single_Scale(self, i_Scale):
        if i_Scale == 0:
            for i_Point in range(self.Parameter_Int_Number_Points):
                self.Array3D_Simulation_Decomposed[:,i_Point,i_Scale] \
                    = self.Array2D_Signal_Decomposed[:,i_Scale]
        else:
            Tuple_Function_Return \
                = self.Function_Simulation_FM_Part_POD_Based(\
                        i_Scale)
            self.Array3D_Simulation_Decomposed[:,:,i_Scale] \
                = Tuple_Function_Return[0]
                
    def Function_CDU_Simulation_All_Scales(self):
        self.Array3D_Simulation_Decomposed \
            = numpy.zeros([self.Array_Time.size, \
                            self.Parameter_Int_Number_Points, \
                            self.Array_Center_Frequency.size])
        for i_Scale in range(self.Array_Center_Frequency.size):
            self.Function_CDU_Simulation_Single_Scale(i_Scale)
        self.Array2D_Simulation \
            = self.Array3D_Simulation_Decomposed.sum(axis = 2)

    #--Calculation-Function-----------------------------------------------------
    def Function_Instantaneous_Phase_Generation(self, \
            Array_Instantaneous_Phase_Measurement):
        """
        Return Generated instantaneous phase
        Three methods can be used:
        1. DWT_based
        2. FFT_based
        3. Original : Direct return original instantaneous phase
        """
        Str_Method = self.Parameter_Str_IF_Method
        if Str_Method == 'DWT_based':
            Array_Instantaneous_Frequency_Measurement \
                = numpy.diff(Array_Instantaneous_Phase_Measurement)
            WPT_Instantaneous_Frequency \
                = pywt.WaveletPacket(\
                    data = Array_Instantaneous_Frequency_Measurement, \
                    wavelet = self.Parameter_Str_Wavelet_Name, \
                    mode = 'symmetric')
            Value_Maximum_Level = WPT_Instantaneous_Frequency.maxlevel
            WPT_Instantaneous_Frequency_Reconstruct \
                = pywt.WaveletPacket\
                        (data = numpy.zeros\
                                (Array_Instantaneous_Phase_Measurement.shape), \
                        wavelet = self.Parameter_Str_Wavelet_Name, \
                        mode='symmetric')
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
            Array_Instantaneous_Phase_Simulation[0] \
                = numpy.random.rand(1) * numpy.pi * 2
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
            i_Scale):
        """
        Single point, Single subcomponent simulation
        ------------------------------------------------------------------------
        Input:
            i_Scale
        ------------------------------------------------------------------------
        Output:
            Array_Simulation_Current_Node
            Array_Instantaneous_Phase_Simulation
        """
        Array_Instantaneous_Phase_Measurement =\
            self.Array2D_Signal_Decomposed_Phase[:,i_Scale]
        Array_Instantaneous_Amplitude_Measurement =\
            self.Array2D_Signal_Decomposed_Amplitude[:,i_Scale]
        Array_Instantaneous_Phase_Simulation \
            = self.Function_Instantaneous_Phase_Generation\
                    (Array_Instantaneous_Phase_Measurement)
        Array_Simulation_Current_Node \
            = numpy.cos(Array_Instantaneous_Phase_Simulation) \
                * Array_Instantaneous_Amplitude_Measurement
        return Array_Simulation_Current_Node, \
                Array_Instantaneous_Phase_Simulation

    def Function_Simulation_FM_Part_SVD_Based_TV_Corr(self, \
            Array2D_Eigen_Vector, Temp_Array2D_2D_SVE_Sigma, i_Scale):
        """
        POD-based simulation with time-invariant correlation and unity standard
        deviation
        ------------------------------------------------------------------------
        Input:
            Array2D_Eigen_Vector
                - Eigenvecgtor of target correlation coefficient matrix
            Array_Eigen_Value
                - Eigenvalue of target corrrelation coefficient matrix
            i_Scale
        ------------------------------------------------------------------------
        Output:
            Array2D_Simulation_Unit_Std
        """
        Array2D_FM_Part_Simulation \
            = numpy.zeros([self.Array_Time.size, \
                            Temp_Array2D_2D_SVE_Sigma.shape[1]])
        # Generate uncorrelated signals for further use
        for i_Random_Signal in range(self.Parameter_Int_Number_Points):
            Array2D_FM_Part_Simulation[:,i_Random_Signal] = self.\
                Function_Current_Node_Simulation_Single_Point(i_Scale)[0]
        # Generated correlated simulation
        Array2D_Simulation \
            = Array2D_Eigen_Vector\
                .dot((Array2D_FM_Part_Simulation \
                        * numpy.sqrt(Temp_Array2D_2D_SVE_Sigma)).T)
        Array2D_Simulation = Array2D_Simulation.T
        return Array2D_Simulation

    def Function_2DSVD_of_TV_Corr_Matrix(self, i_Scale):
        Array3D_TV_Correlation_Scale \
            = self.Array4D_Correlation[:,:,i_Scale,:]
        Array2D_U_SUM \
            = numpy.zeros([self.Parameter_Int_Number_Points, \
                            self.Parameter_Int_Number_Points])
        # Array2D_V_Sum is not calculated since U and V are the tranpose of
        # each other.
        for i_Time in range(self.Array_Time.size):
            Array2D_Snapshot_i \
                = Array3D_TV_Correlation_Scale[:,:,i_Time]
            Array2D_U_SUM \
                += Array2D_Snapshot_i.T.dot(Array2D_Snapshot_i)
        Array2D_Eig_Vecto = numpy.linalg.eig(Array2D_U_SUM)[1]

        Array3D_Sigma = numpy.zeros(Array3D_TV_Correlation_Scale.shape)
        Array2D_Sigma = numpy.zeros([self.Parameter_Int_Number_Points, \
                                        self.Array_Time.size])
        for i_Time in range(self.Array_Time.size):
            Array3D_Sigma[:,:,i_Time] \
                = Array2D_Eig_Vecto.T\
                    .dot(Array3D_TV_Correlation_Scale[:,:,i_Time])\
                    .dot(Array2D_Eig_Vecto)
            Array2D_Sigma[:, i_Time] = numpy.diag(Array3D_Sigma[:,:,i_Time])
        Array2D_Sigma = Array2D_Sigma[:self.Parameter_Int_SVD_Truncation,:].T

        if self.Bool_Flag_Debug == True:
            Array3D_Recons = numpy.zeros(Array3D_TV_Correlation_Scale.shape)
            for i_Time in range(self.Array_Time.size):
                Temp_Array2D_Recons = numpy.zeros(Array2D_U_SUM.shape)
                for i_Eig in range(Array2D_Eig_Vecto.shape[1]):
                    Temp_Array_U_i_Eig\
                         = Array2D_Eig_Vecto[:,i_Eig]
                    Temp_Array2D_Single_SVD_Mode \
                        = Temp_Array_U_i_Eig.reshape(-1,1)\
                            .dot(Temp_Array_U_i_Eig.reshape(1,-1))
                    Temp_Array2D_Recons \
                        += Temp_Array2D_Single_SVD_Mode \
                            * Array2D_Sigma[i_Time, i_Eig]
                Array3D_Recons[:,:,i_Time] = Temp_Array2D_Recons
                Temp_Value_Maximum_Error \
                    = numpy.max(numpy.abs(Array3D_Recons - \
                                            Array3D_TV_Correlation_Scale)\
                                )
            Temp_Str_Output \
                = 'SVD Reconstruction error for scale {} is {}'\
                    .format(i_Scale, Temp_Value_Maximum_Error)
            print(Temp_Str_Output)
        return Array2D_Eig_Vecto, Array2D_Sigma

    def Function_Simulation_FM_Part_POD_Based(self, i_Scale):
        Time_Start = time.time()
        if self.Bool_Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = 'Debug: \n\tScale {} Started\n\t Time: {} \n'\
                    .format(i_Scale, Str_DateTime_Start)
            print(Str_Output)
        # Here, the TV_Correlation determineds that the correlation coffficient
        # matrix is a 4D arrray
        Temp_Array2D_2DSVD_Eigenvector \
            = self.Array3D_Eigen_Vector[:,:,i_Scale]
        Temp_Array2D_2D_SVE_Sigma \
            = self.Array3D_Eigen_Sigma[:,:,i_Scale]

        Array2D_Simulation \
            = self.Function_Simulation_FM_Part_SVD_Based_TV_Corr\
                    (Temp_Array2D_2DSVD_Eigenvector, \
                        Temp_Array2D_2D_SVE_Sigma, \
                        i_Scale)
        Time_Ended = time.time()
        if self.Bool_Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = '\tScale {} Ended\n\t Time: {} \n'\
                    .format(i_Scale, Str_DateTime_Start)
            print(Str_Output)
            Str_Output \
                = '\tFinished the {}th scale, time comsumption: {:.3f} s'\
                            .format(i_Scale, Time_Ended - Time_Start)
            print(Str_Output)
        return Array2D_Simulation, Temp_Array2D_2D_SVE_Sigma

class Class_Simulation_Hilbert_wavelet_TV_POD(\
        Class_Simulation_Hilbert_wavelet_TV_2DSVD):
    """
    Description:
        Time-variant (TV) correlation with POD method
    ----------------------------------------------------------------------------
    Functions:
        - Function_CDU_Corr_Decomposition
        - Function_CDU_Corr_Decomposition_Savememory
        - Function_Simulation_FM_Part_POD_Based_TV_Corr
    """
    def __init__(self, \
        Array2D_Locations,
        Parameter_Str_IF_Method, \
        Parameter_Str_Wavelet_Name, \
        Parameter_Int_SVD_Truncation, \
        Parameter_Int_Max_WPT_Level, \
        Parameter_Str_Method_for_Obtaining_Std):
        Class_Simulation_Hilbert_wavelet_TV_2DSVD.__init__(self, \
            Array2D_Locations, \
            Parameter_Str_IF_Method, \
            Parameter_Str_Wavelet_Name, \
            Parameter_Int_SVD_Truncation, \
            Parameter_Int_Max_WPT_Level, \
            Parameter_Str_Method_for_Obtaining_Std)
        self.Class_Name = 'POD only method for the simulation speed comparison'
        self.Array4D_Eigen_Vector = None # [i_Point, i_Mode, i_Scale, i_Time]
        self.Array3D_POD_Sigma = None # [i_Mode, i_Scale, i_Time]

    def Function_CDU_Corr_Decomposition(self):
        """
        Description:
            Decompose all time-varying spatial correlation
        """
        if self.Bool_Flag_Debug:
            Value_Start_Time = time.time()
        Array4D_Eigen_Vector \
            = numpy.zeros([self.Parameter_Int_Number_Points, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size, \
                            self.Array_Time.size])
        Array3D_POD_Sigma \
            = numpy.zeros([self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size, \
                            self.Array_Time.size])
        for i_Scale in range(self.Array_Center_Frequency.size):
            for i_Time in range(self.Array_Time.size):
                Tuple_Function_Return\
                    = numpy.linalg.eig\
                        (self.Array4D_Correlation[:,:,i_Scale,i_Time])
                Array_Sigma, Array2D_Eig_Vecto = Tuple_Function_Return
                Array4D_Eigen_Vector[:,:,i_Scale, i_Time] = Array2D_Eig_Vecto
                Array3D_POD_Sigma[:, i_Scale, i_Time] = Array_Sigma

        for i_Scale in range(self.Array_Center_Frequency.size):
            for i_Time in range(self.Array_Time.size):
                for i_Mode in range(self.Parameter_Int_Number_Points):
                    if Array4D_Eigen_Vector[0,i_Mode,i_Scale, i_Time] < 0:
                        Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time] \
                            = - Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time]
                        Array3D_POD_Sigma[i_Mode, i_Scale, i_Time] \
                                = - Array3D_POD_Sigma[i_Mode, i_Scale, i_Time]
        for i_Scale in range(self.Array_Center_Frequency.size):
            for i_Time in range(self.Array_Time.size):
                for i_Mode in range(self.Parameter_Int_Number_Points):
                    if Array3D_POD_Sigma[i_Mode,i_Scale, i_Time] < 0:
                        Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time] \
                            = - Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time]
                        Array3D_POD_Sigma[i_Mode, i_Scale, i_Time] \
                                = - Array3D_POD_Sigma[i_Mode, i_Scale, i_Time]
        self.Array4D_Eigen_Vector = Array4D_Eigen_Vector
        self.Array3D_POD_Sigma = Array3D_POD_Sigma
        if self.Bool_Flag_Debug:
            Value_Time_Comsumption = time.time() - Value_Start_Time
            print('Debug:')
            print('\tClass:', self.Class_Name)
            print('\tFunction: Function_CDU_Corr_Decomposition - SVD')
            print('\tTime consumption: {:.4f}'.format(Value_Time_Comsumption))

    def Function_CDU_Corr_Decomposition_Savememory(self):
        """
        Description:
            Decompose all time-varying spatial correlation
        """
        if self.Bool_Flag_Debug:
            Value_Start_Time = time.time()
        Array4D_Eigen_Vector \
            = numpy.zeros([self.Parameter_Int_Number_Points, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size, \
                            self.Array_Time.size])
        Array3D_POD_Sigma \
            = numpy.zeros([self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size, \
                            self.Array_Time.size])
        for i_Scale in range(self.Array_Center_Frequency.size):
            for i_Time in range(self.Array_Time.size):
                Tuple_Function_Return\
                    = numpy.linalg.eig\
                        (self.Array4D_Correlation[:,:,i_Scale,i_Time])
                Array_Sigma, Array2D_Eig_Vecto = Tuple_Function_Return
                Array4D_Eigen_Vector[:,:,i_Scale, i_Time] = Array2D_Eig_Vecto
                Array3D_POD_Sigma[:, i_Scale, i_Time] = Array_Sigma

        for i_Scale in range(self.Array_Center_Frequency.size):
            for i_Time in range(self.Array_Time.size):
                for i_Mode in range(self.Parameter_Int_Number_Points):
                    if Array4D_Eigen_Vector[0,i_Mode,i_Scale, i_Time] < 0:
                        Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time] \
                            = - Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time]
                        Array3D_POD_Sigma[i_Mode, i_Scale, i_Time] \
                                = - Array3D_POD_Sigma[i_Mode, i_Scale, i_Time]
        for i_Scale in range(self.Array_Center_Frequency.size):
            for i_Time in range(self.Array_Time.size):
                for i_Mode in range(self.Parameter_Int_Number_Points):
                    if Array3D_POD_Sigma[i_Mode,i_Scale, i_Time] < 0:
                        Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time] \
                            = - Array4D_Eigen_Vector[:,i_Mode,i_Scale, i_Time]
                        Array3D_POD_Sigma[i_Mode, i_Scale, i_Time] \
                                = - Array3D_POD_Sigma[i_Mode, i_Scale, i_Time]
        self.Array4D_Eigen_Vector = Array4D_Eigen_Vector
        self.Array3D_POD_Sigma = Array3D_POD_Sigma
        if self.Bool_Flag_Debug:
            Value_Time_Comsumption = time.time() - Value_Start_Time
            print('Debug:')
            print('\tClass:', self.Class_Name)
            print('\tFunction: Function_CDU_Corr_Decomposition - SVD')
            print('\tTime consumption: {:.4f}'.format(Value_Time_Comsumption))

    def Function_Simulation_FM_Part_POD_Based_TV_Corr(self, \
            Array3D_Eigen_Vector, Array2D_Eigen_Value, i_Scale):
        """
        POD-based simulation with time-varying correlation and unity standard
        deviation
        ------------------------------------------------------------------------
        Input:
            Array3D_Eigen_Vector
                - Eigenvecgtor of target correlation coefficient matrix
            Array2D_Eigen_Value
                - Eigenvalue of target corrrelation coefficient matrix
            i_Scale
        ------------------------------------------------------------------------
        Output:
            Array2D_Simulation_Unit_Std
        """
        Array2D_FM_Part_Simulation \
            = numpy.zeros([self.Array_Time.size, \
                            Array2D_Eigen_Value.shape[0]])
        # Generate uncorrelated signals for further use
        for i_Random_Signal in range(self.Parameter_Int_Number_Points):
            Tuple_Function_Return = self.\
                Function_Current_Node_Simulation_Single_Point(i_Scale)
            Array_Simulation_Current_Node \
                = Tuple_Function_Return[0]
            Array2D_FM_Part_Simulation[:,i_Random_Signal] \
                = Array_Simulation_Current_Node
        # Generated correlated simulation
        Array2D_Simulation \
            = numpy.zeros([self.Array_Time.size, \
                            self.Parameter_Int_Number_Points])
        for i_Time in range(Array2D_Eigen_Value.shape[1]):
            Array2D_Simulation[i_Time,:] \
                = Array3D_Eigen_Vector[:,:,i_Time]\
                    .dot(numpy.diag(numpy.sqrt(Array2D_Eigen_Value[:,i_Time])))\
                    .dot(Array2D_FM_Part_Simulation[i_Time,:].reshape(-1,1))\
                    .reshape(-1)
        return Array2D_Simulation

    def Function_Simulation_FM_Part_POD_Based(self, i_Scale):
        Time_Start = time.time()
        if self.Bool_Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = 'Scale {} Started\n\t Time: {} \n'\
                    .format(i_Scale, Str_DateTime_Start)
            print(Str_Output)
        # Here, the TV_Correlation determineds that the correlation coffficient
        # matrix is a 4D arrray
        Temp_Array3D_Eigenvector \
            = self.Array4D_Eigen_Vector[:,:,i_Scale, :]
        Temp_Array2D_Sigma \
            = self.Array3D_POD_Sigma[:,i_Scale,:]

        Array2D_Simulation \
            = self.Function_Simulation_FM_Part_POD_Based_TV_Corr\
                    (Temp_Array3D_Eigenvector, \
                        Temp_Array2D_Sigma, \
                        i_Scale)
        Time_Ended = time.time()
        if self.Bool_Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = 'Scale {} Ended\n\t Time: {} \n'\
                    .format(i_Scale, Str_DateTime_Start)
            print(Str_Output)
            Str_Output \
                = 'Finished the {}th scale, time comsumption: {:.3f} s'\
                            .format(i_Scale, Time_Ended - Time_Start)
            print(Str_Output)
        return Array2D_Simulation, Array2D_Simulation

# class Class_Simulation_Hilbert_wavelet_CT_SVD(\
#         Class_Simulation_Hilbert_wavelet_TV_2DSVD):
#     def __init__(self, \
#         Parameter_Total_Height, \
#         Parameter_Int_Number_Points, \
#         Parameter_Str_IF_Method, \
#         Parameter_Str_Wavelet_Name, \
#         Parameter_Int_SVD_Truncation, \
#         Parameter_Int_Max_WPT_Level, \
#         Parameter_Str_Method_for_Obtaining_Std):
#         Class_Simulation_Hilbert_wavelet_TV_2DSVD.__init__(self, \
#             Parameter_Total_Height, \
#             Parameter_Int_Number_Points, \
#             Parameter_Str_IF_Method, \
#             Parameter_Str_Wavelet_Name, \
#             Parameter_Int_SVD_Truncation, \
#             Parameter_Int_Max_WPT_Level, \
#             Parameter_Str_Method_for_Obtaining_Std)
#         self.Class_Name = 'POD only method for the simulation speed comparison'
#         self.Array3D_Eigen_Vector = None # [i_Point, i_Mode, i_Scale, i_Time]
#         self.Array2D_POD_Sigma = None # [i_Mode, i_Scale, i_Time]

#     def Function_CDU_Target_Spatial_Correlation(self, \
#             Array2D_Locations, Str_Correlation_Model):
#         """
#         Description:
#             2D Array: Time-invariant spatial correlation for all scales
#             3D Array: Time-invariant for multi-scale correlation
#             4D Array: Time-varying spatial corrrlation for various scales
#         ------------------------------------------------------------------------
#         Input:
#             Array2D_Locations
#             Str_Correlation_Model:
#                 - 'Hanse_1999' (Constant_Correlation)
#                 - 'Jiang_2017' (Constant_Correlation)
#         """
#         if self.Parameter_Int_Max_WPT_Level == 0:
#             WPT_Signal \
#                 = pywt.WaveletPacket(\
#                     self.Array_Signal, \
#                     self.Parameter_Str_Wavelet_Name, \
#                     'symmetric')
#             Int_Max_WPT_Level = WPT_Signal.maxlevel
#         else:
#             Int_Max_WPT_Level = self.Parameter_Int_Max_WPT_Level
#         Array_Signal_Fluctuation \
#                 = self.Array2D_Signal_Decomposed[:,1:].sum(axis = 1)
#         Array_Time_Varying_Standard_Deviation \
#             = Object_Signal_Processing\
#                 .Function_Time_Varying_Standard_Deviation\
#                     (self.Array_Time, \
#                     Array_Signal_Fluctuation, \
#                     2**Int_Max_WPT_Level, \
#                     self.Parameter_Str_Method_for_Obtaining_Std)
#         self.Array_Time_Varying_Standard_Deviation \
#             = Object_Signal_Processing\
#                 .Function_Remove_Trend(Array_Time_Varying_Standard_Deviation, \
#                                         self.Parameter_Str_Wavelet_Name)[0]
#         if Str_Correlation_Model == 'Jiang_2017':
#             Tuple_Function_Return = Object_Wind_Field_Information\
#                 .Function_Correlation_Generation_Jiang_2017\
#                     (Array2D_Locations, \
#                     self.Array_Center_Frequency, \
#                     self.Array_Time[:1], \
#                     self.Array_Time_Varying_Standard_Deviation[:1], \
#                     self.Array_Time_Varying_Standard_Deviation.mean())
#         else:
#             print('Error: Function_CDU_Target_Spatial_Correlation')
#             print('\t Specified correlaton model not encoded')
#         Array4D_CT_Correlation = Tuple_Function_Return[1]
#         self.Array3D_Correlation = Array4D_CT_Correlation[:,:,:,0]

#     def Function_CDU_Corr_Decomposition(self):
#         """
#         Description:
#             Decompose all time-varying spatial correlation
#         """
#         if self.Bool_Flag_Debug:
#             Value_Start_Time = time.time()
#         Array3D_Eigen_Vector \
#             = numpy.zeros([self.Parameter_Int_Number_Points, \
#                             self.Parameter_Int_SVD_Truncation, \
#                             self.Array_Center_Frequency.size])
#         Array2D_POD_Sigma \
#             = numpy.zeros([self.Parameter_Int_SVD_Truncation, \
#                             self.Array_Center_Frequency.size])
#         for i_Scale in range(self.Array_Center_Frequency.size):
#             Tuple_Function_Return\
#                 = numpy.linalg.eig\
#                     (self.Array3D_Correlation[:,:,i_Scale])
#             Array_Sigma, Array2D_Eig_Vecto = Tuple_Function_Return
#             Array3D_Eigen_Vector[:,:,i_Scale] = Array2D_Eig_Vecto
#             Array2D_POD_Sigma[:, i_Scale] = Array_Sigma

#         self.Array3D_Eigen_Vector = Array3D_Eigen_Vector
#         self.Array2D_POD_Sigma = Array2D_POD_Sigma
#         if self.Bool_Flag_Debug:
#             Value_Time_Comsumption = time.time() - Value_Start_Time
#             print('Debug:')
#             print('\tClass:', self.Class_Name)
#             print('\tFunction: Function_CDU_Corr_Decomposition - SVD')
#             print('\tTime consumption: {:.4f}'.format(Value_Time_Comsumption))

#     def Function_Simulation_FM_Part_POD_Based_CT_Corr(self, \
#             Array3D_Eigen_Vector, Array2D_Eigen_Value, i_Scale):
#         """
#         POD-based simulation with time-varying correlation and unity standard
#         devaition
#         ------------------------------------------------------------------------
#         Input:
#             Array3D_Eigen_Vector
#                 - Eigenvecgtor of target correlation coefficient matrix
#             Array2D_Eigen_Value
#                 - Eigenvalue of target corrrelation coefficient matrix
#             i_Scale
#         ------------------------------------------------------------------------
#         Output:
#             Array2D_Simulation_Unit_Std
#         """
#         Array2D_FM_Part_Simulaiton \
#             = numpy.zeros([self.Array_Time.size, \
#                             Array2D_Eigen_Value.shape[0]])
#         # Generate uncorrelated signals for further use
#         for i_Random_Signal in range(self.Parameter_Int_Number_Points):
#             Tuple_Function_Return = self.\
#                 Function_Current_Node_Simulation_Single_Point(i_Scale)
#             Array_Simulation_Current_Node \
#                 = Tuple_Function_Return[0]
#             Array2D_FM_Part_Simulaiton[:,i_Random_Signal] \
#                 = Array_Simulation_Current_Node
#         # Generated correlated simulation
#         Array2D_Simulation \
#             = numpy.zeros([self.Array_Time.size, \
#                             self.Parameter_Int_Number_Points])
#         Array2D_Simulation \
#             = Array3D_Eigen_Vector[:,:,i_Scale]\
#                 .dot(numpy.diag(numpy.sqrt(Array2D_Eigen_Value[:,i_Scale])))\
#                 .dot(Array2D_FM_Part_Simulaiton[:,:].T)\
#                     .reshape(-1)
#         return Array2D_Simulation

#     def Function_Simulation_FM_Part_POD_Based(self, i_Scale):
#         Time_Start = time.time()
#         if self.Bool_Flag_Debug:
#             Str_DateTime_Start \
#                 = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
#             Str_Output \
#                 = 'Scale {} Started\n\t Time: {} \n'\
#                     .format(i_Scale, Str_DateTime_Start)
#             print(Str_Output)
#         # Here, the TV_Correlation determineds that the correlation coffficient
#         # matrix is a 4D arrray

#         Array2D_Simulation \
#             = self.Function_Simulation_FM_Part_POD_Based_CT_Corr\
#                     (self.Array3D_Eigen_Vector, \
#                         self.Array2D_POD_Sigma, \
#                         i_Scale)
#         Time_Ended = time.time()
#         if self.Bool_Flag_Debug:
#             Str_DateTime_Start \
#                 = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
#             Str_Output \
#                 = 'Scale {} Ended\n\t Time: {} \n'\
#                     .format(i_Scale, Str_DateTime_Start)
#             print(Str_Output)
#             Str_Output \
#                 = 'Finished the {}th scale, time comsumption: {:.3f} s'\
#                             .format(i_Scale, Time_Ended - Time_Start)
#             print(Str_Output)

#         return Array2D_Simulation, Array2D_Simulation

class Class_Simulation_Hilbert_wavelet_CT_POD(\
        Class_Simulation_Hilbert_wavelet_TV_2DSVD):
    """
    Deacription:
        Constant (CT) correlation
    """
    def __init__(self, \
        Array2D_Locations, \
        Parameter_Str_IF_Method, \
        Parameter_Str_Wavelet_Name, \
        Parameter_Int_SVD_Truncation, \
        Parameter_Int_Max_WPT_Level, \
        Parameter_Str_Method_for_Obtaining_Std):
        Class_Simulation_Hilbert_wavelet_TV_2DSVD.__init__(self, \
            Array2D_Locations, \
            Parameter_Str_IF_Method, \
            Parameter_Str_Wavelet_Name, \
            Parameter_Int_SVD_Truncation, \
            Parameter_Int_Max_WPT_Level, \
            Parameter_Str_Method_for_Obtaining_Std)
        self.Class_Name = 'POD only method for the simulation speed comparison'
        self.Array2D_POD_Sigma = None # [i_SVD_Mode, i_Scale]
 
    def Function_CDU_Target_Spatial_Correlation(self, \
            Str_Correlation_Model):
        """
        Description:
            2D Array: Time-invariant spatial correlation for all scales
            3D Array: Time-invariant for multi-scale correlation
            4D Array: Time-varying spatial corrrlation for various scales
        ------------------------------------------------------------------------
        Input:
            Array2D_Locations
            Str_Correlation_Model:
                - 'Hanse_1999' (Constant_Correlation)
                - 'Jiang_2017' (Constant_Correlation)
        """
        if self.Parameter_Int_Max_WPT_Level == 0:
            WPT_Signal \
                = pywt.WaveletPacket(\
                    self.Array_Signal, \
                    self.Parameter_Str_Wavelet_Name, \
                    'symmetric')
            Int_Max_WPT_Level = WPT_Signal.maxlevel
        else:
            Int_Max_WPT_Level = self.Parameter_Int_Max_WPT_Level
        Array_Signal_Fluctuation \
                = self.Array2D_Signal_Decomposed[:,1:].sum(axis = 1)
        Array_Time_Varying_Standard_Deviation \
            = Object_Signal_Processing\
                .Function_Time_Varying_Standard_Deviation\
                    (self.Array_Time, \
                    Array_Signal_Fluctuation, \
                    2**Int_Max_WPT_Level, \
                    self.Parameter_Str_Method_for_Obtaining_Std)
        self.Array_Time_Varying_Standard_Deviation \
            = Object_Signal_Processing\
                .Function_Remove_Trend(Array_Time_Varying_Standard_Deviation, \
                                        self.Parameter_Str_Wavelet_Name)[0]
        if Str_Correlation_Model == 'Jiang_2017':
            Tuple_Function_Return = Object_Wind_Field_Information\
                .Function_Correlation_Generation_Jiang_2017(\
                    self.Array2D_Locations, \
                    self.Array_Center_Frequency, \
                    self.Array_Time[:1], \
                    self.Array_Time_Varying_Standard_Deviation[:1], \
                    self.Array_Time_Varying_Standard_Deviation.mean())
            Array4D_CT_Correlation = Tuple_Function_Return[1]
            self.Array3D_Correlation = Array4D_CT_Correlation[:,:,:,0]
        elif Str_Correlation_Model == 'Hanse_1999':
            Tuple_Function_Return = Object_Wind_Field_Information\
                .Function_HVD_HC_Coherence_Hanse_1999(
                    self.Array2D_Locations, \
                    self.Array_Center_Frequency, \
                    self.Array_Signal.mean(), \
                    100)
            self.Array3D_Correlation = Tuple_Function_Return[0]
        else:
            print('Error: Function_CDU_Target_Spatial_Correlation')
            print('\t Specified correlaton model not encoded')
        
        

    def Function_CDU_Corr_Decomposition(self):
        """
        Description:
            Decompose all time-varying spatial correlation
        """
        if self.Bool_Flag_Debug:
            Value_Start_Time = time.time()
        Array3D_Eigen_Vector \
            = numpy.zeros([self.Parameter_Int_Number_Points, \
                            self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size])
        Array2D_POD_Sigma \
            = numpy.zeros([self.Parameter_Int_SVD_Truncation, \
                            self.Array_Center_Frequency.size])
        for i_Scale in range(self.Array_Center_Frequency.size):
            Tuple_Function_Return\
                = numpy.linalg.eig(\
                    self.Array3D_Correlation[:,:,i_Scale])
            Array_Sigma, Array2D_Eig_Vecto = Tuple_Function_Return
            Array3D_Eigen_Vector[:,:,i_Scale] = Array2D_Eig_Vecto
            Array2D_POD_Sigma[:, i_Scale] = Array_Sigma

        self.Array3D_Eigen_Vector = Array3D_Eigen_Vector
        self.Array2D_POD_Sigma = Array2D_POD_Sigma
        if self.Bool_Flag_Debug:
            Value_Time_Comsumption = time.time() - Value_Start_Time
            print('Debug:')
            print('\tClass:', self.Class_Name)
            print('\tFunction: Function_CDU_CTPOD_Eigendecomposition - SVD')
            print('\tTime consumption: {:.4f}'.format(Value_Time_Comsumption))

    def Function_Simulation_FM_Part_POD_Based_CT_Corr(self, \
            Array3D_Eigen_Vector, Array2D_Eigen_Value, i_Scale):
        """
        POD-based simulation with time-varying correlation and unity standard
        devaition
        ------------------------------------------------------------------------
        Input:
            Array3D_Eigen_Vector
                - Eigenvecgtor of target correlation coefficient matrix
            Array2D_Eigen_Value
                - Eigenvalue of target corrrelation coefficient matrix
            i_Scale
        ------------------------------------------------------------------------
        Output:
            Array2D_Simulation_Unit_Std
        """
        Array2D_FM_Part_Simulaiton \
            = numpy.zeros([self.Array_Time.size, \
                            Array2D_Eigen_Value.shape[0]])
        # Generate uncorrelated signals for further use
        for i_Random_Signal in range(self.Parameter_Int_Number_Points):
            Tuple_Function_Return = self.\
                Function_Current_Node_Simulation_Single_Point(i_Scale)
            Array_Simulation_Current_Node \
                = Tuple_Function_Return[0]
            Array2D_FM_Part_Simulaiton[:,i_Random_Signal] \
                = Array_Simulation_Current_Node
        # Generated correlated simulation
        Array2D_Simulation \
            = numpy.zeros([self.Array_Time.size, \
                            self.Parameter_Int_Number_Points])
        Array2D_Simulation \
            = Array3D_Eigen_Vector[:,:,i_Scale]\
                .dot(numpy.diag(numpy.sqrt(Array2D_Eigen_Value[:,i_Scale])))\
                .dot(Array2D_FM_Part_Simulaiton.T).T
        return Array2D_Simulation

    def Function_Simulation_FM_Part_POD_Based(self, i_Scale):
        Time_Start = time.time()
        if self.Bool_Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = 'Scale {} Started\n\t Time: {} \n'\
                    .format(i_Scale, Str_DateTime_Start)
            print(Str_Output)
        # Here, the CT_Correlation determineds that the correlation coffficient
        # matrix is a 3D arrray
        Array2D_Simulation \
            = self.Function_Simulation_FM_Part_POD_Based_CT_Corr\
                    (self.Array3D_Eigen_Vector, \
                        self.Array2D_POD_Sigma, \
                        i_Scale)
        Time_Ended = time.time()
        if self.Bool_Flag_Debug:
            Str_DateTime_Start \
                = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(Time_Start))
            Str_Output \
                = 'Scale {} Ended\n\t Time: {} \n'\
                    .format(i_Scale, Str_DateTime_Start)
            print(Str_Output)
            Str_Output \
                = 'Finished the {}th scale, time comsumption: {:.3f} s'\
                            .format(i_Scale, Time_Ended - Time_Start)
            print(Str_Output)
        return Array2D_Simulation, self.Array2D_POD_Sigma
