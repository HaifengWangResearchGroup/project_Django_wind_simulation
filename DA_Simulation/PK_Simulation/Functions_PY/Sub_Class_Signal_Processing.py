"""
Creation data:
    20190318
Modification records:
    20190318
        - Moved from Sub_Class
        - Commited out nonstationary index functions

To be done:
    - Creat nonstationary index class
"""

import numpy
import pywt

from scipy import signal
from scipy import interpolate

class Class_Signal_Processing():
    """
    Function List:
        - Function_Wavlelet_Max_Level
        - Function_Low_Pass_Filter
        - Function_Time_Varying_Correlation_Coefficient
        - Function_Time_Varying_Mean
        - Function_Time_Varying_Standard_Deviation
        - Function_WPT_HT_Decomposition
        - Function_WPT_NHT_Decomposition
        - Fundtion_Surrogate_Data_Generation
        - Fundtion_Surrogate_Data_Generation_With_Padding
        - Function_Signal_Padding
        - Function_Mirror_Signal_Padding
        - Function_Signal_Depedding_2D
        - Function_SVR_Prediction
        - Function_SVR_2D_Training_Whole_Prediction
        - Function_Nonstationary_Ratio
        # - Function_Local_Nonstationary_Index
        # - Function_Global_Nonstationary_Index
        # - Function_Global_Nonstationary_Index_With_FRF
        # - Function_Relative_Nonstationary_Index
        # - Function_Relative_Nonstationary_Index_With_FRF
        # - Function_Relative_Nonstationary_Index_With_Distribution
        # - Function_Relative_Nonstationary_Index_With_Distribution_FRF
        # - Function_Nonstationary_Index_Pad_Depad_Remove_Prediction
        - Function_Normalized_Hilbert_Transform
        - Function_Moving_Average
        - Function_Remove_Trend
        - Function_Remove_Amplitude_Modulation
        - Function_Make_Unit_Length
        - Function_Zero_Padding_Both_Side
    """
    def __init__(self):
        self.Class_Name = 'Class of self defined signal processing functions'
    
    def Function_Wavlelet_Max_Level(self, \
            Array_Signal, Value_Delta_T, Str_Wavelet_Name):
        Int_Max_WPT_Level = pywt.dwt_max_level(Array_Signal.size, \
                    pywt.Wavelet(Str_Wavelet_Name))
        Value_Window_Width = Value_Delta_T * 2**Int_Max_WPT_Level
        return Int_Max_WPT_Level, Value_Window_Width

    def Function_Low_Pass_Filter(self, Array_Signal):
        Array_Filter = numpy.zeros(Array_Signal.shape)
        Int_Length_Filter = 1
        Array_Filter[0:Int_Length_Filter] = 1 / Int_Length_Filter
        self.Array_Filtered_Signal = numpy.convolve(Array_Signal, Array_Filter) 
        self.Array_Filtered_Signal \
            = self.Array_Filtered_Signal[:Array_Signal.size]
        return self.Array_Filtered_Signal    
    
    def Function_Time_Varying_Correlation_Coefficient(self, \
            Array_Time, Array_Signal_1, Array_Signal_2, Value_Averaging_Time):
        Value_Sampling_Interval = numpy.diff(Array_Time).mean()
        Int_Averaging_Points \
            = int(Value_Averaging_Time / Value_Sampling_Interval)
        Array_Correlation_Coefficient = numpy.zeros(Array_Time.size)
        i_Start_Index \
            = int(Int_Averaging_Points / 2)
        i_Ended_Index \
            = int(Array_Time.size - Int_Averaging_Points / 2)
        for i_Time in numpy.arange(i_Start_Index, i_Ended_Index, 1):
            Temp_i_Window_Start \
                = int(i_Time - Int_Averaging_Points / 2)
            Temp_i_Window_Ended \
                = int(i_Time + Int_Averaging_Points / 2)
            Temp_Array_Selected_Index \
                = numpy.arange(Temp_i_Window_Start, Temp_i_Window_Ended, 1)
            Temp_Array2D_Corrcoef \
                = numpy.corrcoef(Array_Signal_1\
                                        [Temp_Array_Selected_Index], \
                                Array_Signal_2\
                                        [Temp_Array_Selected_Index])
            Array_Correlation_Coefficient[i_Time] \
                = Temp_Array2D_Corrcoef[0,1]
        return Array_Correlation_Coefficient
    
    def Function_Time_Varying_Mean(self, \
            Array_Time, Array_Signal, Value_Averaging_Time):
        self.Array_Time = Array_Time
        self.Array_Signal = Array_Signal
        self.Value_Averaging_Time = Value_Averaging_Time
        self.Array_Time_Varying_Mean = numpy.zeros(self.Array_Time.size)
        self.Value_Sampling_Interval = numpy.diff(Array_Time).mean()
        self.Int_Averaging_Points \
            = int(self.Value_Averaging_Time / self.Value_Sampling_Interval)
        Temp_Index_Start \
            = int(self.Int_Averaging_Points / 2)
        Temp_Index_Ended \
            = int(self.Array_Time.size - self.Int_Averaging_Points / 2)
        for i_Time in numpy.arange(Temp_Index_Start, Temp_Index_Ended, 1):
            Temp_Array_Selected_Index \
                = numpy.arange(int(i_Time - self.Int_Averaging_Points / 2), \
                                int(i_Time + self.Int_Averaging_Points / 2), \
                                1)
            self.Array_Time_Varying_Mean[i_Time] \
                    = numpy.mean(self.Array_Signal[Temp_Array_Selected_Index])
        return self.Array_Time_Varying_Mean
    
    def Function_Time_Varying_Standard_Deviation(self, \
            Array_Time, Array_Signal_Fluctuation, Value_Averaging_Time,
            Str_Method):
        """
        Description:
            Calculate the time-vayring standard deviation by moving average
            method
        ------------------------------------------------------------------------
        Input:
            Array_Time
            Array_Signal_Fluctuation
            Value_Averaging_Time
            Str_Method:
                1. Moving_Average
                2. Wavelet (TBD)
        ------------------------------------------------------------------------
        Output:
            Array_Time_Varying_Standard_Deviation
        """
        if Str_Method == 'Moving_Average':
            self.Array_Time = Array_Time
            self.Array_Signal_Fluctuation = Array_Signal_Fluctuation
            self.Value_Averaging_Time = Value_Averaging_Time
            self.Array_Time_Varying_Standard_Deviation \
                = numpy.zeros(self.Array_Time.size)
            self.Value_Sampling_Interval = numpy.diff(Array_Time).mean()
            self.Int_Averaging_Points \
                = int(self.Value_Averaging_Time / self.Value_Sampling_Interval)
            Temp_Index_Start \
                = int(self.Int_Averaging_Points / 2)
            Temp_Index_Ended \
                = int(self.Array_Time.size - self.Int_Averaging_Points / 2)
            for i_Time in numpy.arange(Temp_Index_Start, Temp_Index_Ended, 1):
                Temp_Array_Selected_Index \
                    = numpy.arange(\
                            int(i_Time - self.Int_Averaging_Points / 2), \
                            int(i_Time + self.Int_Averaging_Points / 2), \
                            1)
                self.Array_Time_Varying_Standard_Deviation[i_Time] \
                    = numpy.std(self.Array_Signal_Fluctuation\
                                        [Temp_Array_Selected_Index])
            self.Array_Time_Varying_Standard_Deviation\
                    [: int(self.Int_Averaging_Points / 2) + 1] \
                = numpy.linspace\
                        (numpy.abs(Array_Signal_Fluctuation[0]), \
                        self.Array_Time_Varying_Standard_Deviation\
                        [int(self.Int_Averaging_Points / 2) + 1], \
                        int(self.Int_Averaging_Points / 2) + 1)
            self.Array_Time_Varying_Standard_Deviation\
                    [-int(self.Int_Averaging_Points / 2) - 2 : ] \
                = numpy.linspace\
                        (self.Array_Time_Varying_Standard_Deviation\
                        [- int(self.Int_Averaging_Points / 2) - 2], \
                        numpy.abs(Array_Signal_Fluctuation[-1]), \
                        int(self.Int_Averaging_Points / 2) + 2)
        else:
            print("Error! Secified method not found!")
        return self.Array_Time_Varying_Standard_Deviation
    
    def Function_WPT_HT_Decomposition(self, \
            Array_Time, Array_Signal, \
            Str_Wavelet_Name, Parameter_Max_WPT_Level, \
            Array_Index_Selected_Component):
        """
        Wavelet packet decompostion nested with Hilbert transform
        ------------------------------------------------------------------------
        Input:
            Array_Time, Array_Signal,Str_Wavelet_Name, Parameter_Max_WPT_Level
            Array_Index_Selected_Component:
                if == 'All' -> Highest decomposition level
                if != 'All' -> Return selected component
        ------------------------------------------------------------------------
        Output:
            Array2D_Signal_Decomposed
            Array2D_Signal_Decomposed_Amplitude
            Array2D_Signal_Decomposed_Phase
            Array2D_Signal_Decomposed_Frequency
            Array_Center_Frequency[Array_Index_Selected_Component]
        """
        Value_Signal_Mean = Array_Signal.mean()
        Array_Signal = Array_Signal - Value_Signal_Mean
        WPT_Signal \
            = pywt.WaveletPacket(Array_Signal, Str_Wavelet_Name, 'symmetric')
        if Parameter_Max_WPT_Level == 0:
            Value_Maximum_Level = WPT_Signal.maxlevel
        else:
            Value_Maximum_Level = Parameter_Max_WPT_Level # WPT_Signal.maxlevel
        if str(Array_Index_Selected_Component).lower() == 'all':
            Array_Index_Selected_Component \
                = numpy.arange(2**Value_Maximum_Level)
        Array2D_Signal_Decomposed \
            = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
        i_Current_Node = 0
        Array_Flag_Component_Selection \
            = numpy.zeros(2**Value_Maximum_Level, dtype= bool)
        Array_Flag_Component_Selection[Array_Index_Selected_Component] = True
        for node in WPT_Signal.get_level(Value_Maximum_Level, 'freq'):
            if Array_Flag_Component_Selection[i_Current_Node] == True:
                WPT_Reconstruct \
                    = pywt.WaveletPacket(numpy.zeros(Array_Signal.shape), \
                                            Str_Wavelet_Name, 'symmetric')
                WPT_Reconstruct[node.path] = WPT_Signal[node.path].data
                Array_Signal_Decomposed = WPT_Reconstruct.reconstruct()
                Array2D_Signal_Decomposed[:,i_Current_Node] \
                    = Array_Signal_Decomposed
            i_Current_Node = i_Current_Node + 1
        Array2D_Signal_Decomposed[:,0] += Value_Signal_Mean
        Array2D_Signal_Decomposed \
            = Array2D_Signal_Decomposed[:, Array_Index_Selected_Component]
        Array2D_Signal_Decomposed_Amplitude \
            = numpy.zeros((Array_Signal.size, \
                            Array_Index_Selected_Component.size))
        Array2D_Signal_Decomposed_Phase \
            = numpy.zeros((Array_Signal.size, \
                            Array_Index_Selected_Component.size))
        for i_Current_Node in range(Array_Index_Selected_Component.size):
            Array_Temp_Signal_Subcomponents \
                = signal.hilbert(Array2D_Signal_Decomposed[:,i_Current_Node])
            Array_Temp_Signal_Subcomponent_Amplitude \
                = numpy.abs(Array_Temp_Signal_Subcomponents)
            Array_Temp_Signal_Subcomponent_Phase \
                = numpy.angle(Array_Temp_Signal_Subcomponents)
            Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node] \
                = Array_Temp_Signal_Subcomponent_Amplitude
            Array2D_Signal_Decomposed_Phase[:,i_Current_Node] \
                = numpy.unwrap(Array_Temp_Signal_Subcomponent_Phase)
        Array2D_Signal_Decomposed_Frequency \
            = numpy.diff(Array2D_Signal_Decomposed_Phase, axis = 0) \
                    / numpy.diff(Array_Time).mean() \
                    / 2 / numpy.pi
        Array2D_Signal_Decomposed_Frequency \
            = numpy.append\
                        (numpy.zeros\
                                ((1, \
                                Array2D_Signal_Decomposed_Phase.shape[1])),\
                        Array2D_Signal_Decomposed_Frequency, \
                        axis = 0)
        Value_Default_Sample_Frequency = 1 / numpy.diff(Array_Time).mean()
        Value_Default_Low_Frequency_Threshold \
            = 1 / (Array_Time.max() - Array_Time.min())
        Value_Frequency_Band_Width \
            = (Value_Default_Sample_Frequency / 2 \
                    - Value_Default_Low_Frequency_Threshold) \
                / 2**Value_Maximum_Level
        Array_Center_Frequency \
            = numpy.linspace(Value_Default_Low_Frequency_Threshold, \
                                Value_Default_Sample_Frequency / 2 \
                                    - Value_Frequency_Band_Width, \
                                2**Value_Maximum_Level) \
                + Value_Frequency_Band_Width / 2
        Temp_Int_Number_Nodes = Array2D_Signal_Decomposed_Amplitude.shape[1]
        for i_Current_Node in range(Temp_Int_Number_Nodes):
            Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node] \
                = self.Function_Low_Pass_Filter\
                        (Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node])
        return Array2D_Signal_Decomposed, \
                Array2D_Signal_Decomposed_Amplitude, \
                Array2D_Signal_Decomposed_Phase, \
                Array2D_Signal_Decomposed_Frequency, \
                Array_Center_Frequency[Array_Index_Selected_Component]

    def Function_WPT_NHT_Decomposition(self, \
            Array_Time, Array_Signal, \
            Str_Wavelet_Name, Parameter_Max_WPT_Level = 0, \
            Array_Index_Selected_Component = 'All'):
        # Wavelet packet decomposition with normalized Hilbert transform
        Array_Time = numpy.copy(Array_Time, order = 'C')
        Array_Signal = numpy.copy(Array_Signal, order = 'C')
        Value_Signal_Mean = Array_Signal.mean()
        Array_Signal = Array_Signal - Value_Signal_Mean
        Value_Default_Sample_Frequency = 1 / numpy.diff(Array_Time).mean()
        WPT_Signal \
            = pywt.WaveletPacket(Array_Signal, Str_Wavelet_Name, 'symmetric')
        if Parameter_Max_WPT_Level == 0:
            Value_Maximum_Level = WPT_Signal.maxlevel
        else:
            Value_Maximum_Level \
                = Parameter_Max_WPT_Level # WPT_Signal.maxlevel
        Array2D_Signal_Decomposed \
            = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
        i_Current_Node = 0
        for node in WPT_Signal.get_level(Value_Maximum_Level, 'freq'):
            WPT_Reconstruct \
                = pywt.WaveletPacket(numpy.zeros(Array_Signal.shape), \
                                        Str_Wavelet_Name, \
                                        'symmetric')
            WPT_Reconstruct[node.path] = WPT_Signal[node.path].data
            Array_Signal_Decomposed = WPT_Reconstruct.reconstruct()
            Array2D_Signal_Decomposed[:,i_Current_Node] \
                = Array_Signal_Decomposed
            i_Current_Node = i_Current_Node + 1
        Array2D_Signal_Decomposed[:,0] += Value_Signal_Mean 
        Array2D_Signal_Decomposed_Amplitude \
            = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
        Array2D_Signal_Decomposed_Phase \
            = numpy.zeros((Array_Signal.size,2**Value_Maximum_Level))
        for i_Current_Node in range(2**Value_Maximum_Level):
            Tuple_Function_Return \
                = self.Function_Normalized_Hilbert_Transform\
                        (Array2D_Signal_Decomposed[:,i_Current_Node])
            Array_Normalized_Hilbert_Transform, Array_Normalize_Envelop \
                = Tuple_Function_Return[0:2]
            Array_Temp_Signal_Subcomponent_Amplitude \
                = Array_Normalize_Envelop
            Array_Temp_Signal_Subcomponent_Phase \
                = numpy.angle(Array_Normalized_Hilbert_Transform)
            Array2D_Signal_Decomposed_Amplitude[:,i_Current_Node] \
                = Array_Temp_Signal_Subcomponent_Amplitude
            Array2D_Signal_Decomposed_Phase[:,i_Current_Node] \
                = numpy.unwrap(Array_Temp_Signal_Subcomponent_Phase)
        Array2D_Signal_Decomposed_Frequency \
            = numpy.diff(Array2D_Signal_Decomposed_Phase, axis = 0) \
                / numpy.diff(Array_Time).mean() / 2 / numpy.pi
        Array2D_Signal_Decomposed_Frequency \
            = numpy.append(\
                    numpy.zeros((1, Array2D_Signal_Decomposed_Phase.shape[1])),\
                    Array2D_Signal_Decomposed_Frequency, axis = 0)
        Value_Frequency_Band_Width \
            = Value_Default_Sample_Frequency \
                / Array2D_Signal_Decomposed.shape[1] / 2
        Array_Center_Frequency \
            = numpy.linspace(Value_Frequency_Band_Width, \
                                Value_Default_Sample_Frequency / 2, \
                                Array2D_Signal_Decomposed.shape[1])
        Temp_Int_Number_Nodes = Array2D_Signal_Decomposed_Amplitude.shape[1]
        for i_Current_Node in range(Temp_Int_Number_Nodes):
            Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node] \
                = self.Function_Low_Pass_Filter(\
                        Array2D_Signal_Decomposed_Amplitude[:, i_Current_Node])  
        return Array2D_Signal_Decomposed, \
                Array2D_Signal_Decomposed_Amplitude, \
                Array2D_Signal_Decomposed_Phase, \
                Array2D_Signal_Decomposed_Frequency, \
                Array_Center_Frequency

    def Fundtion_Surrogate_Data_Generation(self, \
            Array_Time, Array_Signal, Int_Number_of_Surrogates):
        Array_Signal_FFT = numpy.fft.fft(Array_Signal)
        Array_Signal_FFT_Amplitude = numpy.abs(Array_Signal_FFT)
        Array_Signal_FFT_Phase = numpy.angle(Array_Signal_FFT)
        Array2D_Signal_Surrogate \
            = numpy.zeros((Array_Signal.size, Int_Number_of_Surrogates), \
                            dtype = numpy.complex_)
        for i_Surrogate in range(Int_Number_of_Surrogates):
            Array_Signal_FFT_Phase_Generated \
                = numpy.random.rand(Array_Signal_FFT_Phase.size) \
                        * 2 * numpy.pi \
                    - numpy.pi
            Array_Signal_FFT_Phase_Generated[0] = 0
            if Array_Signal_FFT_Phase_Generated.size % 2 == 0:
                Array_Signal_FFT_Phase_Generated\
                [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
                = -numpy\
                    .flipud(\
                        Array_Signal_FFT_Phase_Generated\
                        [1: int(Array_Signal_FFT_Phase_Generated.size / 2) + 0])
            else:
                Array_Signal_FFT_Phase_Generated\
                    [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
                    = -numpy.flipud(Array_Signal_FFT_Phase_Generated\
                        [1: int(Array_Signal_FFT_Phase_Generated.size / 2) + 1])
            Array2D_Signal_Surrogate[:,i_Surrogate] \
                = numpy\
                    .fft.ifft(Array_Signal_FFT_Amplitude \
                            * numpy.exp(1j * Array_Signal_FFT_Phase_Generated))
        Array2D_Signal_Surrogate = Array2D_Signal_Surrogate.real
        return Array2D_Signal_Surrogate

    def Fundtion_Surrogate_Data_Generation_With_Padding(self, \
            Array_Time, Array_Signal, Int_Number_of_Surrogates, \
            Int_Number_Prediction):
        Array_Signal_FFT = numpy.fft.fft(Array_Signal)
        Array_Signal_FFT_Amplitude = numpy.abs(Array_Signal_FFT)
        Array_Signal_FFT_Phase = numpy.angle(Array_Signal_FFT)
        Array2D_Signal_Surrogate \
            = numpy.zeros((Array_Signal.size, Int_Number_of_Surrogates), \
                            dtype = numpy.complex_)
        for i_Surrogate in range(Int_Number_of_Surrogates):
            Array_Signal_FFT_Phase_Generated \
                = numpy.random.rand(Array_Signal_FFT_Phase.size) \
                    * 2 * numpy.pi - numpy.pi
            Array_Signal_FFT_Phase_Generated[0] = 0
            if Array_Signal_FFT_Phase_Generated.size % 2 == 0:
                Array_Signal_FFT_Phase_Generated\
                    [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
                        = -numpy.flipud(Array_Signal_FFT_Phase_Generated\
                            [1: int(Array_Signal_FFT_Phase_Generated.size / 2) \
                                    + 0])
            else:
                Array_Signal_FFT_Phase_Generated\
                    [int(Array_Signal_FFT_Phase_Generated.size / 2) + 1:] \
                        = -numpy.flipud(Array_Signal_FFT_Phase_Generated\
                            [1: int(Array_Signal_FFT_Phase_Generated.size / 2) \
                                    + 1])
            Array2D_Signal_Surrogate[:,i_Surrogate] \
                = numpy.fft.ifft(Array_Signal_FFT_Amplitude * numpy.exp( 1j \
                    * Array_Signal_FFT_Phase_Generated))
        Array2D_Signal_Surrogate = Array2D_Signal_Surrogate.real
        if Int_Number_Prediction == 0:
            pass
        else:
            Array2D_Signal_Surrogate \
                = numpy.append(Array2D_Signal_Surrogate, \
                                Array2D_Signal_Surrogate\
                                    [:Int_Number_Prediction * 2,:], \
                                axis = 0)
        return Array2D_Signal_Surrogate

    def Function_Signal_Padding(self, \
            Array_Time, Array_Signal, Int_Number_Prediction):
        """
        Have minor bug
        Should be replaced with "Function_Mirror_Signal_Padding"
        Kept for the compatibility of old functions
        ------------------------------------------------------------------------
        Input:
            Array_Time
            Array_Signal
            Int_Number_Prediction
        ------------------------------------------------------------------------
        Output:
            Array_Time_With_Pad
            Array_Signal_With_Pad
        """
        Array_Signal_With_Pad \
            = numpy.zeros(Array_Signal.size + Int_Number_Prediction * 2)
        Temp_Value_Mean = Array_Signal[0:Int_Number_Prediction].mean()
        Array_Signal_With_Pad[0 :Int_Number_Prediction - 1] \
            = - numpy.flipud(Array_Signal[0:Int_Number_Prediction - 1]) \
                + Temp_Value_Mean * 2
        Array_Signal_With_Pad[Int_Number_Prediction - 1] = Temp_Value_Mean
        Array_Signal_With_Pad\
            [Int_Number_Prediction:Int_Number_Prediction + Array_Signal.size] \
                = Array_Signal
        Temp_Value_Mean = Array_Signal[- Int_Number_Prediction:].mean()
        Array_Signal_With_Pad\
            [Int_Number_Prediction + Array_Signal.size + 1\
            : Array_Signal_With_Pad.size] \
                = - numpy.flipud(Array_Signal[- Int_Number_Prediction + 1:]) \
                    + Temp_Value_Mean * 2
        Array_Signal_With_Pad[Int_Number_Prediction + Array_Signal.size] \
            = Temp_Value_Mean
        Array_Time_With_Pad \
            = numpy.zeros(Array_Time.size + Int_Number_Prediction * 2)
        Value_Delta_t = numpy.diff(Array_Time).mean()
        Array_Time_With_Pad\
            = numpy.linspace\
                (Array_Time.min() - Value_Delta_t * Int_Number_Prediction, \
                Array_Time.max() + Value_Delta_t * Int_Number_Prediction, \
                Array_Time.size + Int_Number_Prediction * 2)
        return Array_Time_With_Pad, Array_Signal_With_Pad
    
    def Function_Mirror_Signal_Padding(self, \
            Array_Time, Array_Signal, Int_Number_Prediction):
        """
        Mirror padding of the original signal:
            if Prediction == 0, then return original arrays
            if Prediction >=1,  then do mirror padding
        ------------------------------------------------------------------------
        Input:
            Array_Time
            Array_Signal
            Int_Number_Prediction
        ------------------------------------------------------------------------
        Output:
            Array_Time_With_Pad
            Array_Signal_With_Pad
        """
        if Int_Number_Prediction <= 0:
            Array_Time_With_Pad, Array_Signal_With_Pad \
                = Array_Time, Array_Signal
        else:
            Array_Signal_With_Pad \
                = numpy.zeros(Array_Signal.size + Int_Number_Prediction * 2)
            Temp_Value_End = Array_Signal[0]
            Array_Signal_With_Pad[0:Int_Number_Prediction] \
                = - numpy.flipud(Array_Signal[1:Int_Number_Prediction + 1]) \
                    + Temp_Value_End * 2
            Array_Signal_With_Pad\
                [Int_Number_Prediction \
                :Int_Number_Prediction + Array_Signal.size] \
                    = Array_Signal
            Array_Signal_With_Pad\
                [Int_Number_Prediction + Array_Signal.size\
                : Array_Signal_With_Pad.size] \
                    = - numpy.flipud\
                        (Array_Signal[- Int_Number_Prediction - 1: -1]) \
                        + Temp_Value_End * 2
            Value_Delta_t = Array_Time[1] - Array_Time[0]
            Array_Time_With_Pad\
                = numpy.linspace(\
                    Array_Time.min() - Value_Delta_t * Int_Number_Prediction, \
                    Array_Time.max() + Value_Delta_t * Int_Number_Prediction, \
                    Array_Time.size + Int_Number_Prediction * 2)
        return Array_Time_With_Pad, Array_Signal_With_Pad

    def Function_Signal_Depedding_2D(self, Array_Time_With_Pad, \
        Array2D_Signal_With_Pad, Int_Number_Prediction):
        """
        Depadding 2d Array
        ------------------------------------------------------------------------
        Output:
            Array_Time
            Array2D_Signal
        """
        Array_Time \
            = Array_Time_With_Pad\
                [Int_Number_Prediction \
                    : Array_Time_With_Pad.size - Int_Number_Prediction]
        Array2D_Signal \
            = Array2D_Signal_With_Pad\
                [Int_Number_Prediction \
                    : Array_Time_With_Pad.size - Int_Number_Prediction,:]
        return Array_Time, Array2D_Signal

    # def Function_Nonstationary_Ratio(self, Array_Time, Array_Signal, \
    #                                 Str_Wavelet_Name, \
    #                                 Int_Number_of_Surrogates, \
    #                                 Parameter_Max_WPT_Level):
    #     """
    #     Calculates the ratio of the local energy to that of the averaged one 
    #     """
    #     # Signal decompostiion
    #     Tuple_Functin_Return \
    #         = self.Function_WPT_HT_Decomposition\
    #                 (Array_Time, Array_Signal, Str_Wavelet_Name, \
    #                     Parameter_Max_WPT_Level, 'All')
    #     Array2D_Signal_Decomposed = Tuple_Functin_Return[0]
    #     Array2D_Signal_Decomposed_Amplitude = Tuple_Functin_Return[1]
    #     Array2D_Signal_Decomposed_Frequency = Tuple_Functin_Return[3]
    #     Array_Center_Frequency = Tuple_Functin_Return[4]
            
    #     # Surrogate data generation
    #     Array2D_Signal_Surrogate \
    #         = self.Fundtion_Surrogate_Data_Generation\
    #                     (Array_Time, Array_Signal, Int_Number_of_Surrogates)
    #     Array3D_Surrogate_Decomposed_Amplitude \
    #         = numpy.zeros(Array_Time.size, \
    #                         Array_Center_Frequency.size, \
    #                         Int_Number_of_Surrogates)
    #     for i_Surrogate in range(Int_Number_of_Surrogates):
    #         Array_Surrogate = Array2D_Signal_Surrogate[:,i_Surrogate]
    #         Tuple_Return_WPT_Decomposition \
    #             = self.Function_WPT_HT_Decomposition\
    #                     (Array_Time, Array_Surrogate, Str_Wavelet_Name, \
    #                     Parameter_Max_WPT_Level, "all")
    #         Array2D_Surrogate_Decomposed_Amplitude \
    #             = Tuple_Return_WPT_Decomposition[1]
    #         Array_Center_Frequency \
    #             = Tuple_Return_WPT_Decomposition[4]
    #         Array3D_Surrogate_Decomposed_Amplitude[:,:,i_Surrogate] \
    #             = Array2D_Surrogate_Decomposed_Amplitude
    #     Array2D_Surrogate_Decomposed_Amplitude \
    #         = Array3D_Surrogate_Decomposed_Amplitude.mean(axis = 2)
    #     Array_Surrogate_Amplitude \
    #         = Array2D_Surrogate_Decomposed_Amplitude.mean(axis = 0)
    #     Array2D_Surrogate_Decomposed_Amplitude_Mean \
    #         = numpy.dot(numpy.ones((Array_Time.size, 1)), \
    #                     Array_Surrogate_Amplitude\
    #                         .reshape(1, Array_Center_Frequency.size))
    #     Array2D_Ratio \
    #         = Array2D_Signal_Decomposed_Amplitude \
    #             / Array2D_Surrogate_Decomposed_Amplitude_Mean
    #     return Array2D_Ratio, Array2D_Signal_Decomposed, \
    #             Array2D_Signal_Decomposed_Amplitude, \
    #             Array2D_Signal_Decomposed_Frequency, \
    #             Array_Center_Frequency, \
    #             Array3D_Surrogate_Decomposed_Amplitude, \
    #             Array2D_Signal_Surrogate

    # def Function_Local_Nonstationary_Index(self, \
    #         Array_Time_Predict, Array_Signal_Predict, \
    #         Int_Length_Prediction, \
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level):
    #     """
    #     Calculates the ratio of the local energy to that of the averaged one 
    #     ------------------------------------------------------------------------
    #     Input:
    #         Int_Length_Prediction: Single side prediction length
    #     ------------------------------------------------------------------------
    #     Output:
    #         Array2D_Local_Nonstationary_Index
    #         Array2D_Ratio
    #         Array2D_Signal_Decomposed
    #         Array2D_Signal_Decomposed_Amplitude
    #         Array2D_Signal_Decomposed_Frequency
    #         Array_Center_Frequency
    #     """
    #     # Signal decompostiion
    #     Tuple_Functin_Return \
    #         = self.Function_WPT_HT_Decomposition(\
    #                 Array_Time_Predict, Array_Signal_Predict, \
    #                 Str_Wavelet_Name, Parameter_Max_WPT_Level, 'all')
    #     Array2D_Signal_Predict_Decomposed, \
    #     Array2D_Signal_Predict_Decomposed_Amplitude, \
    #         = Tuple_Functin_Return[0:2]
    #     Array2D_Signal_Predict_Decomposed_Frequency, Array_Center_Frequency \
    #         = Tuple_Functin_Return[3:5]
    #     Array_Time = Array_Time_Predict[Int_Length_Prediction \
    #                 :Array_Time_Predict.size - Int_Length_Prediction]
    #     Array2D_Signal_Decomposed \
    #         = Array2D_Signal_Predict_Decomposed\
    #             [Int_Length_Prediction \
    #                 :Array_Time_Predict.size - Int_Length_Prediction, :]
    #     Array2D_Signal_Decomposed_Amplitude \
    #         = Array2D_Signal_Predict_Decomposed_Amplitude\
    #             [Int_Length_Prediction \
    #                 :Array_Time_Predict.size - Int_Length_Prediction, :]
    #     Array2D_Signal_Decomposed_Frequency \
    #         = Array2D_Signal_Predict_Decomposed_Frequency\
    #             [Int_Length_Prediction \
    #                 :Array_Time_Predict.size - Int_Length_Prediction, :]
    #     Array_Signal_Decomposed_Amplitude_Mean \
    #         = Array2D_Signal_Decomposed_Amplitude.mean(axis = 0)
    #     Array2D_Signal_Decomposed_Amplitude_Mean\
    #         = numpy.dot(numpy.ones((Array_Time.size, 1)), \
    #                     Array_Signal_Decomposed_Amplitude_Mean\
    #                         .reshape(1, Array_Center_Frequency.size))
    #     Array2D_Ratio \
    #         = Array2D_Signal_Decomposed_Amplitude \
    #             / Array2D_Signal_Decomposed_Amplitude_Mean
    #     Array2D_Local_Nonstationary_Index = numpy.abs(Array2D_Ratio - 1)
    #     return Array2D_Local_Nonstationary_Index, \
    #             Array2D_Ratio, \
    #             Array2D_Signal_Decomposed, \
    #             Array2D_Signal_Decomposed_Amplitude, \
    #             Array2D_Signal_Decomposed_Frequency, \
    #             Array_Center_Frequency

    # def Function_Global_Nonstationary_Index(self, \
    #         Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level):
    #     """
    #     Calculate the global nonstationary index
    #     ------------------------------------------------------------------------
    #     Output:
    #         Value_Global_Nonstationary_Index
    #         Array2D_Signal_Decomposed_Amplitude
    #     """
    #     Tuple_Function_Return \
    #         = self.Function_Local_Nonstationary_Index(\
    #                 Array_Time_Predict, Array_Signal_Predict, \
    #                 Int_Length_Prediction, Str_Wavelet_Name, \
    #                 Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
    #     Array2D_Local_Nonstationary_Index = Tuple_Function_Return[0]
    #     Array2D_Signal_Decomposed_Amplitude = Tuple_Function_Return[3]
    #     # Calculate the global nonstationary index
    #     Value_Global_Nonstationary_Index \
    #         = numpy.sum(Array2D_Local_Nonstationary_Index \
    #             * Array2D_Signal_Decomposed_Amplitude) \
    #             / numpy.sum(Array2D_Signal_Decomposed_Amplitude)
    #     return Value_Global_Nonstationary_Index, \
    #             Array2D_Signal_Decomposed_Amplitude

    # def Function_Global_Nonstationary_Index_With_FRF(self, \
    #         Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level, \
    #         Value_Damping_Ratio, Value_Mass, Value_Stiffness):
    #     """
    #     Calculate the global nonstationary index
    #     ------------------------------------------------------------------------
    #     Output:
    #         Value_Global_Nonstationary_Index
    #         Array2D_Signal_Decomposed_Amplitude
    #     """
    #     # Create structural response object
    #     Object_Structural_Response = Class_Structural_Response()
    #     # Calculation
    #     Tuple_Function_Return \
    #         = self.Function_Local_Nonstationary_Index(\
    #                 Array_Time_Predict, Array_Signal_Predict, \
    #                 Int_Length_Prediction, Str_Wavelet_Name, \
    #                 Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
    #     Array2D_Local_Nonstationary_Index = Tuple_Function_Return[0]
    #     Array2D_Signal_Decomposed_Amplitude = Tuple_Function_Return[3]
    #     Array_Center_Frequency = Tuple_Function_Return[5]
    #     # FRF
    #     Array_Frequency_Response \
    #         = Object_Structural_Response\
    #             .Function_SDOF_Frequency_Response\
    #                 (Array_Center_Frequency, \
    #                 Value_Damping_Ratio, Value_Mass, Value_Stiffness)
    #     Array2D_Frequency_Response \
    #         = numpy.repeat(Array_Frequency_Response.reshape(1, -1), \
    #                         Array2D_Local_Nonstationary_Index.shape[0], \
    #                         axis = 0)
    #     # Calculate the global nonstationary index
    #     Value_Global_Nonstationary_Index \
    #         = numpy.sum(Array2D_Local_Nonstationary_Index \
    #             * Array2D_Signal_Decomposed_Amplitude \
    #             * Array2D_Frequency_Response) \
    #             / numpy.sum(Array2D_Signal_Decomposed_Amplitude \
    #                         * Array2D_Frequency_Response)
    #     return Value_Global_Nonstationary_Index, \
    #             Array2D_Signal_Decomposed_Amplitude

    # def Function_Relative_Nonstationary_Index(self, \
    #         Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level):
    #     """
    #     Calculate the global nonstationary index
    #     """
    #     # Get original data
    #     # Array_Time \
    #     #     = Array_Time_Predict\
    #     #         [Int_Length_Prediction \
    #     #             : Array_Time_Predict.size - Int_Length_Prediction]
    #     # Array_Signal \
    #     #     = Array_Signal_Predict\
    #     #         [Int_Length_Prediction \
    #     #             : Array_Time_Predict.size - Int_Length_Prediction]
    #     # Get the corresponding impulse data
    #     # Array_Impulse = numpy.zeros(Array_Time_Predict.size)
    #     # Value_Energy = numpy.sum(Array_Signal**2)
    #     # Int_Mid_Index = int(Array_Time.size / 2)
    #     # Array_Impulse[Int_Mid_Index] \
    #     #     = numpy.sqrt(Value_Energy / 2)
    #     # Array_Impulse[Int_Mid_Index + 1] \
    #     #     = - Array_Impulse[Int_Mid_Index]
    #     Object_Signal_Generation = Class_Signal_Generation()
    #     Tuple_Function_Return \
    #         = Object_Signal_Generation\
    #             .Function_Impulse_Signal_Generation_Oscilate_Delta\
    #                 (Array_Time_Predict, \
    #                 Array_Signal_Predict, \
    #                 Int_Length_Prediction)
    #     Array_Impulse_Predict = Tuple_Function_Return[1]
    #     # Calculate the global nonstationary index of the analyzed signal
    #     Tuple_Function_Return \
    #         = self.Function_Global_Nonstationary_Index(\
    #                 Array_Time_Predict, Array_Signal_Predict, \
    #                 Int_Length_Prediction, \
    #                 Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #                 Parameter_Max_WPT_Level)
    #     Value_Global_Nonstationary_Index_Signal = Tuple_Function_Return[0]
    #     # Calculate the global nonstationary index of the impulse signal
    #     Tuple_Function_Return \
    #         = self.Function_Global_Nonstationary_Index(\
    #                 Array_Time_Predict, Array_Impulse_Predict, \
    #                 Int_Length_Prediction, \
    #                 Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #                 Parameter_Max_WPT_Level)
    #     Value_Global_Nonstationary_Index_Impulse = Tuple_Function_Return[0]
    #     Value_Relative_Nonstationary_Index \
    #         = Value_Global_Nonstationary_Index_Signal \
    #             / Value_Global_Nonstationary_Index_Impulse
    #     return Value_Relative_Nonstationary_Index, Array_Impulse_Predict

    # def Function_Relative_Nonstationary_Index_With_FRF(self, \
    #         Array_Time_Predict, Array_Signal_Predict, Int_Length_Prediction, \
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level, \
    #         Value_Damping_Ratio, Value_Mass, Value_Stiffness):
    #     """
    #     Calculate the global nonstationary index
    #     """
    #     # Get original data
    #     # Array_Time \
    #     #     = Array_Time_Predict\
    #     #         [Int_Length_Prediction \
    #     #             : Array_Time_Predict.size - Int_Length_Prediction]
    #     # Array_Signal \
    #     #     = Array_Signal_Predict\
    #     #         [Int_Length_Prediction \
    #     #             : Array_Time_Predict.size - Int_Length_Prediction]
    #     # Get the corresponding impulse data
    #     # Array_Impulse = numpy.zeros(Array_Time_Predict.size)
    #     # Value_Energy = numpy.sum(Array_Signal**2)
    #     # Int_Mid_Index = int(Array_Time.size / 2)
    #     # Array_Impulse[Int_Mid_Index] \
    #     #     = numpy.sqrt(Value_Energy / 2)
    #     # Array_Impulse[Int_Mid_Index + 1] \
    #     #     = - Array_Impulse[Int_Mid_Index]
    #     Object_Signal_Generation = Class_Signal_Generation()
    #     Tuple_Function_Return \
    #         = Object_Signal_Generation\
    #             .Function_Impulse_Signal_Generation_Oscilate_Delta\
    #                 (Array_Time_Predict, \
    #                 Array_Signal_Predict, \
    #                 Int_Length_Prediction)
    #     Array_Impulse_Predict = Tuple_Function_Return[1]
    #     # Calculate the global nonstationary index of the analyzed signal
    #     Tuple_Function_Return \
    #         = self.Function_Global_Nonstationary_Index_With_FRF(\
    #                 Array_Time_Predict, Array_Signal_Predict, \
    #                 Int_Length_Prediction, \
    #                 Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #                 Parameter_Max_WPT_Level, \
    #                 Value_Damping_Ratio, Value_Mass, Value_Stiffness)
    #     Value_Global_Nonstationary_Index_Signal = Tuple_Function_Return[0]
    #     # Calculate the global nonstationary index of the impulse signal
    #     Tuple_Function_Return \
    #         = self.Function_Global_Nonstationary_Index_With_FRF(\
    #                 Array_Time_Predict, Array_Impulse_Predict, \
    #                 Int_Length_Prediction, \
    #                 Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #                 Parameter_Max_WPT_Level, \
    #                 Value_Damping_Ratio, Value_Mass, Value_Stiffness)
    #     Value_Global_Nonstationary_Index_Impulse = Tuple_Function_Return[0]
    #     Value_Relative_Nonstationary_Index \
    #         = Value_Global_Nonstationary_Index_Signal \
    #             / Value_Global_Nonstationary_Index_Impulse
    #     return Value_Relative_Nonstationary_Index, Array_Impulse_Predict

    # def Function_Relative_Nonstationary_Index_With_Distribution(self, \
    #         Array_Time_Predict, Array_Signal_Predict, \
    #         Int_Length_Prediction, 
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level):
    #     Array_Time \
    #         = Array_Time_Predict\
    #             [Int_Length_Prediction \
    #                 : Array_Time_Predict.size - Int_Length_Prediction]
    #     Array_Signal \
    #         = Array_Signal_Predict\
    #             [Int_Length_Prediction \
    #                 : Array_Time_Predict.size - Int_Length_Prediction]
    #     Tuple_Function_Return \
    #         = self.Function_Relative_Nonstationary_Index(\
    #             Array_Time_Predict, Array_Signal_Predict, \
    #             Int_Length_Prediction, \
    #             Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #             Parameter_Max_WPT_Level)
    #     Value_Nonstationary_Index = Tuple_Function_Return[0]
    #     if Int_Number_of_Surrogates == 0:
    #         Array_Surrogate_Ratio_Sorted = 0
    #         Array2D_Signal_Surrogate_Prediction = 0
    #     else:
    #         Array2D_Signal_Surrogate_Prediction \
    #             = self.Fundtion_Surrogate_Data_Generation_With_Padding(\
    #                     Array_Time, Array_Signal, Int_Number_of_Surrogates, \
    #                     Int_Length_Prediction)
    #         Array_Relative_Nonstationary_Index_Surrogate \
    #             = numpy.zeros(Int_Number_of_Surrogates)
    #         for i_Surrogate in range(Int_Number_of_Surrogates):
    #             Array_Surrogate_Prediction \
    #                 = Array2D_Signal_Surrogate_Prediction[:,i_Surrogate]
    #             # Tuple_Return_WPT_Decomposition \
    #             #     = self.Function_WPT_HT_Decomposition(\
    #             #             Array_Time, Array_Surrogate, Str_Wavelet_Name, \
    #             #             Parameter_Max_WPT_Level)
    #             Tuple_Function_Return \
    #                 = self.Function_Relative_Nonstationary_Index(\
    #                     Array_Time_Predict, Array_Surrogate_Prediction, \
    #                     Int_Length_Prediction, \
    #                     Str_Wavelet_Name,  Int_Number_of_Surrogates , \
    #                     Parameter_Max_WPT_Level)
    #             Array_Relative_Nonstationary_Index_Surrogate[i_Surrogate] \
    #                 = Tuple_Function_Return[0]
    #         Array_Surrogate_Ratio_Sorted \
    #             = numpy.sort(Array_Relative_Nonstationary_Index_Surrogate)
    #     return Value_Nonstationary_Index, Array_Surrogate_Ratio_Sorted, \
    #             Array2D_Signal_Surrogate_Prediction

    # def Function_Relative_Nonstationary_Index_With_Distribution_FRF(self, \
    #         Array_Time_Predict, Array_Signal_Predict, \
    #         Int_Length_Prediction, 
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level, \
    #         Value_Damping_Ratio, Value_Mass, Value_Stiffness):
    #     Array_Time \
    #         = Array_Time_Predict\
    #             [Int_Length_Prediction \
    #                 : Array_Time_Predict.size - Int_Length_Prediction]
    #     Array_Signal \
    #         = Array_Signal_Predict\
    #             [Int_Length_Prediction \
    #                 : Array_Time_Predict.size - Int_Length_Prediction]
    #     Tuple_Function_Return \
    #         = self.Function_Relative_Nonstationary_Index_With_FRF(\
    #             Array_Time_Predict, Array_Signal_Predict, \
    #             Int_Length_Prediction, \
    #             Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #             Parameter_Max_WPT_Level, \
    #             Value_Damping_Ratio, Value_Mass, Value_Stiffness)
    #     Value_Nonstationary_Index = Tuple_Function_Return[0]
    #     if Int_Number_of_Surrogates == 0:
    #         Array_Surrogate_Ratio_Sorted = 0
    #         Array2D_Signal_Surrogate_Prediction = 0
    #     else:
    #         Array2D_Signal_Surrogate_Prediction \
    #             = self.Fundtion_Surrogate_Data_Generation_With_Padding(\
    #                     Array_Time, Array_Signal, Int_Number_of_Surrogates, \
    #                     Int_Length_Prediction)
    #         Array_Relative_Nonstationary_Index_Surrogate \
    #             = numpy.zeros(Int_Number_of_Surrogates)
    #         for i_Surrogate in range(Int_Number_of_Surrogates):
    #             Array_Surrogate_Prediction \
    #                 = Array2D_Signal_Surrogate_Prediction[:,i_Surrogate]
    #             # Tuple_Return_WPT_Decomposition \
    #             #     = self.Function_WPT_HT_Decomposition(\
    #             #             Array_Time, Array_Surrogate, Str_Wavelet_Name, \
    #             #             Parameter_Max_WPT_Level)
    #             Tuple_Function_Return \
    #                 = self.Function_Relative_Nonstationary_Index_With_FRF(\
    #                     Array_Time_Predict, Array_Surrogate_Prediction, \
    #                     Int_Length_Prediction, \
    #                     Str_Wavelet_Name,  Int_Number_of_Surrogates , \
    #                     Parameter_Max_WPT_Level, \
    #                     Value_Damping_Ratio, Value_Mass, Value_Stiffness)
    #             Array_Relative_Nonstationary_Index_Surrogate[i_Surrogate] \
    #                 = Tuple_Function_Return[0]
    #         Array_Surrogate_Ratio_Sorted \
    #             = numpy.sort(Array_Relative_Nonstationary_Index_Surrogate)
    #     return Value_Nonstationary_Index, Array_Surrogate_Ratio_Sorted, \
    #             Array2D_Signal_Surrogate_Prediction

    # def Function_Nonstationary_Index_Pad_Depad_Remove_Prediction(self, \
    #         Array_Time_Predict, Array_Signal_Predict, \
    #         Int_Number_Prediction, 
    #         Str_Wavelet_Name, Int_Number_of_Surrogates, \
    #         Parameter_Max_WPT_Level):
    #     Object_Signal_Generation = Class_Signal_Generation()
    #     class Signal_Info:
    #         # Collect the information of the signal including the decomposed 
    #         # signal, amplitudes, surrogates etc.
    #         def __init__(self, \
    #                     Array2D_Signal_Surrogate, \
    #                     Array2D_Signal_Decomposed_Amplitude, \
    #                     Array2D_Signal_Decomposed, \
    #                     Array2D_Surrogate_Decomposed_Amplitude_Mean):
    #             self.Array2D_Surrogate = Array2D_Signal_Surrogate
    #             self.Array2D_Decomposed_Amplitude \
    #                     = Array2D_Signal_Decomposed_Amplitude
    #             self.Array2D_Decomposed = Array2D_Signal_Decomposed
    #             self.Array2D_Surrogate_Decomposed_Amplitude_Mean \
    #                     = Array2D_Surrogate_Decomposed_Amplitude_Mean

    #     # Generate corresponding impulse signal data
    #     Array_Signal_Impulse \
    #         = Object_Signal_Generation\
    #             .Function_Impulse_Signal_Generation(Array_Signal_Predict)
    #     Tuple_Function_Return \
    #         = self.Function_Nonstationary_Ratio\
    #                     (Array_Time_Predict, Array_Signal_Predict, \
    #                     Str_Wavelet_Name, \
    #                     Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
    #     Array2D_With_Pad_Ratio = Tuple_Function_Return[0]
    #     Array2D_With_Pad_Signal_Decomposed = Tuple_Function_Return[1]
    #     Array2D_With_Pad_Signal_Decomposed_Amplitude = Tuple_Function_Return[3]
    #     Array2D_With_Pad_Signal_Decomposed_Frequency = Tuple_Function_Return[4]
    #     Array_With_Pad_Center_Frequency = Tuple_Function_Return[5]
    #     Array3D_Signal_Surrogate_Decomposed_Amplitude = Tuple_Function_Return[6]
    #     Array2D_Signal_Surrogate = Tuple_Function_Return[7]

    #     Array2D_With_Pad_Signal_Decomposed, \
    #     Array2D_With_Pad_Signal_Decomposed_Amplitude, \
    #     Array2D_With_Pad_Signal_Decomposed_Phase, \
    #     Array2D_With_Pad_Signal_Decomposed_Frequency, \
    #     Array_With_Pad_Center_Frequency \
    #         = self.Function_WPT_HT_Decomposition\
    #                     (Array_Time_Predict, Array_Signal_Predict, \
    #                     Str_Wavelet_Name, Parameter_Max_WPT_Level)
    #     # Mirror padding impulse and obtain the Hilbert spectrum
    #     Array2D_With_Pad_Impulse_Ratio, \
    #     Array2D_With_Pad_Signal_Impulse_Decomposed, \
    #     Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude, \
    #     Array2D_With_Pad_Signal_Impulse_Decomposed_Frequency, \
    #     Array_With_Pad_Impulse_Center_Frequency, \
    #     Array3D_Impulse_Surrogate_Decomposed_Amplitude, \
    #     Array2D_Impulse_Surrogate \
    #         = self.Function_Nonstationary_Ratio\
    #                     (Array_Time_Predict, Array_Signal_Impulse, \
    #                     Str_Wavelet_Name, \
    #                     Int_Number_of_Surrogates, Parameter_Max_WPT_Level)
    #     Array2D_With_Pad_Signal_Impulse_Decomposed, \
    #     Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude, \
    #     Array2D_With_Pad_Signal_Impulse_Decomposed_Phase, \
    #     Array2D_With_Pad_Signal_Impulse_Decomposed_Frequency, \
    #     Array_With_Pad_Impulse_Center_Frequency \
    #         = self.Function_WPT_HT_Decomposition\
    #                 (Array_Time_Predict, Array_Signal_Impulse, \
    #                 Str_Wavelet_Name, Parameter_Max_WPT_Level)
    #     # Surrogate data generation for signal and the Hilbert spectrum 
    #     Array2D_Surrogate_Decomposed_Amplitude \
    #         = Array3D_Signal_Surrogate_Decomposed_Amplitude.mean(axis = 2)
    #     Array_Surrogate_Amplitude \
    #         = Array2D_Surrogate_Decomposed_Amplitude.mean(axis = 0)
    #     Array2D_Surrogate_Decomposed_Amplitude_Mean \
    #         = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
    #                     Array_Surrogate_Amplitude\
    #                         .reshape(1, Array_With_Pad_Center_Frequency.size))
    #     Array2D_With_Pad_Surrogate_Decomposed_Amplitude_Mean \
    #         = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
    #                     Array_Surrogate_Amplitude\
    #                         .reshape(1, Array_With_Pad_Center_Frequency.size))
    #     # Surrogate data generation for impulse and the Hilbert spectrum
    #     Array2D_Impulse_Surrogate_Decomposed_Amplitude \
    #         = Array3D_Impulse_Surrogate_Decomposed_Amplitude.mean(axis = 2)
    #     Array_Impulse_Surrogate_Amplitude \
    #         = Array2D_Impulse_Surrogate_Decomposed_Amplitude.mean(axis = 0)
    #     Array2D_Impulse_Surrogate_Decomposed_Amplitude_Mean \
    #         = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
    #                     Array_Impulse_Surrogate_Amplitude\
    #                         .reshape(1, Array_With_Pad_Center_Frequency.size))
    #     Array2D_With_Pad_Impulse_Surrogate_Decomposed_Amplitude_Mean \
    #         = numpy.dot(numpy.ones((Array_Time_Predict.size, 1)), \
    #                     Array_Impulse_Surrogate_Amplitude\
    #                         .reshape(1, Array_With_Pad_Center_Frequency.size))
    #     # Get raw ratio
    #     Array2D_With_Pad_Ratio \
    #         = Array2D_With_Pad_Signal_Decomposed_Amplitude \
    #             / Array2D_With_Pad_Surrogate_Decomposed_Amplitude_Mean
    #     Array2D_With_Pad_Impulse_Ratio \
    #         = Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude \
    #             / Array2D_With_Pad_Impulse_Surrogate_Decomposed_Amplitude_Mean
    #     # Depadding
    #     Int_Padded_Number = Int_Number_Prediction
    #     Array_With_Pad_Depadded_Time, \
    #     Array2D_With_Pad_Depadded_Signal_Decomposed \
    #         = self.Function_Signal_Depedding_2D\
    #                 (Array_Time_Predict, \
    #                 Array2D_With_Pad_Signal_Decomposed, \
    #                 Int_Padded_Number)
    #     Array_With_Pad_Depadded_Time, \
    #     Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude \
    #         = self.Function_Signal_Depedding_2D\
    #                 (Array_Time_Predict, \
    #                 Array2D_With_Pad_Signal_Decomposed_Amplitude, \
    #                 Int_Padded_Number)
    #     Array_With_Pad_Depadded_Time, Array2D_With_Pad_Depadded_Ratio \
    #         = self.Function_Signal_Depedding_2D\
    #                 (Array_Time_Predict, Array2D_With_Pad_Ratio, \
    #                 Int_Padded_Number)
    #     Array_With_Pad_Depadded_Time, \
    #     Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed \
    #         = self.Function_Signal_Depedding_2D\
    #                 (Array_Time_Predict, \
    #                 Array2D_With_Pad_Signal_Impulse_Decomposed, \
    #                 Int_Padded_Number)
    #     Array_With_Pad_Depadded_Time, \
    #     Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude \
    #         = self.Function_Signal_Depedding_2D\
    #                 (Array_Time_Predict, \
    #                 Array2D_With_Pad_Signal_Impulse_Decomposed_Amplitude, \
    #                 Int_Padded_Number)
    #     Array_With_Pad_Depadded_Time, \
    #     Array2D_With_Pad_Depadded_Impulse_Ratio \
    #         = self.Function_Signal_Depedding_2D\
    #                 (Array_Time_Predict, \
    #                 Array2D_With_Pad_Impulse_Ratio, \
    #                 Int_Padded_Number)
    #     # Normalization by minusing one and weighting by the spectrum
    #     Array2D_With_Pad_Depadded_Normalized_Ratio \
    #         = (Array2D_With_Pad_Depadded_Ratio - 1) \
    #             * Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude \
    #             / Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude\
    #                 .sum().sum()

    #     Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio \
    #         = (Array2D_With_Pad_Depadded_Impulse_Ratio - 1) \
    #         * Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude \
    #         / Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude\
    #             .sum().sum()
    #     # Remove negative value as they are treated as stationary part
    #     Temp_Int_Number_Cols \
    #         = Array2D_With_Pad_Depadded_Normalized_Ratio.shape[1]
    #     Temp_Int_Number_Rows \
    #         = Array2D_With_Pad_Depadded_Normalized_Ratio.shape[0]
    #     for i_Col in range(Temp_Int_Number_Cols):
    #         for i_Row in range(Temp_Int_Number_Rows):
    #             if Array2D_With_Pad_Depadded_Normalized_Ratio\
    #                     [i_Row, i_Col] <= 0:
    #                 Array2D_With_Pad_Depadded_Normalized_Ratio\
    #                     [i_Row, i_Col] = 0
    #             if Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio\
    #                     [i_Row, i_Col] <= 0:
    #                 Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio\
    #                     [i_Row, i_Col] = 0
    #     # Distribution and threshold value of the surrogate data
    #     Array3D_Normalized_Surrogate_Ratio \
    #         = numpy.zeros((Array_With_Pad_Depadded_Time.size, \
    #                         Array_With_Pad_Center_Frequency.size, \
    #                         Int_Number_of_Surrogates))
    #     for i_Surrogate in range(Int_Number_of_Surrogates):
    #         Temp_Array2D_Surrogate_Ratio \
    #             = Array3D_Signal_Surrogate_Decomposed_Amplitude\
    #                     [:,:,i_Surrogate] \
    #                 / Array2D_Surrogate_Decomposed_Amplitude_Mean
    #         Temp_Array2D_Surrogate_Normalized_Ratio \
    #             = (Temp_Array2D_Surrogate_Ratio - 1) \
    #                 * Array3D_Signal_Surrogate_Decomposed_Amplitude\
    #                     [:,:,i_Surrogate] \
    #                 / Array3D_Signal_Surrogate_Decomposed_Amplitude\
    #                     [:,:,i_Surrogate].sum().sum()
    #         Temp_Int_Number_Cols \
    #             = Temp_Array2D_Surrogate_Normalized_Ratio.shape[1]
    #         Temp_Int_Number_Rows \
    #             = Temp_Array2D_Surrogate_Normalized_Ratio.shape[0]
    #         for i_Col in range(Temp_Int_Number_Cols):
    #             for i_Row in range(Temp_Int_Number_Rows):
    #                 if Temp_Array2D_Surrogate_Normalized_Ratio\
    #                         [i_Row, i_Col] <= 0:
    #                     Temp_Array2D_Surrogate_Normalized_Ratio\
    #                         [i_Row, i_Col] = 0
    #         Array_With_Pad_Depadded_Time, \
    #         Temp_Array2D_Surrogate_Normalized_Ratio_Depadded \
    #             = self.Function_Signal_Depedding_2D\
    #                     (Array_Time_Predict, \
    #                     Temp_Array2D_Surrogate_Normalized_Ratio, \
    #                     Int_Padded_Number)
    #         Array3D_Normalized_Surrogate_Ratio[:,:,i_Surrogate] \
    #             = Temp_Array2D_Surrogate_Normalized_Ratio_Depadded.copy()
    #     Array_Surrogate_Ratio = Array3D_Normalized_Surrogate_Ratio.reshape(-1)
    #     Array_Surrogate_Ratio_Sorted = numpy.sort(Array_Surrogate_Ratio)
    #     Array_With_Pad_Depadded_Center_Frequency \
    #         = Array_With_Pad_Center_Frequency.copy()
    #     Int_Threshold_Index = int(Array_Surrogate_Ratio_Sorted.size * 0.95)
    #     Value_Threshold = Array_Surrogate_Ratio_Sorted[Int_Threshold_Index]
    #     Value_Mean = Array_Surrogate_Ratio_Sorted.mean()
    #     # Get the final ratio    
    #     Value_Raw_Nonstationary_Index_Signal \
    #         = Array2D_With_Pad_Depadded_Normalized_Ratio[:,:]\
    #             .sum().sum()
    #     Value_Raw_Nonstationary_Index_Impulse \
    #         = Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio[:,:]\
    #             .sum().sum()
    #     Value_Nonstationary_Index \
    #         = Value_Raw_Nonstationary_Index_Signal \
    #             / Value_Raw_Nonstationary_Index_Impulse
    #     Object_Singal \
    #         = Signal_Info\
    #             (Array2D_Signal_Surrogate, \
    #             Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude, \
    #             Array2D_With_Pad_Depadded_Signal_Decomposed, \
    #             Array2D_Surrogate_Decomposed_Amplitude_Mean)
    #     Object_Signal_Impulse \
    #         = Signal_Info(Array2D_Signal_Surrogate, \
    #             Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude, \
    #             Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed, \
    #             Array2D_Surrogate_Decomposed_Amplitude_Mean)
    #     return Value_Nonstationary_Index, \
    #             Value_Raw_Nonstationary_Index_Signal, \
    #             Value_Raw_Nonstationary_Index_Impulse, \
    #             Array2D_With_Pad_Depadded_Ratio, \
    #             Array2D_With_Pad_Depadded_Signal_Decomposed, \
    #             Array2D_With_Pad_Depadded_Signal_Decomposed_Amplitude, \
    #             Array2D_With_Pad_Depadded_Normalized_Ratio, \
    #             Array2D_With_Pad_Depadded_Impulse_Ratio, \
    #             Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed, \
    #             Array2D_With_Pad_Depadded_Signal_Impulse_Decomposed_Amplitude, \
    #             Array2D_With_Pad_Depadded_Impulse_Normalized_Ratio, \
    #             Array_With_Pad_Depadded_Center_Frequency, \
    #             Value_Threshold, \
    #             Value_Mean, \
    #             Object_Singal, \
    #             Object_Signal_Impulse, \
    #             Array_Surrogate_Ratio_Sorted, \
    #             Array3D_Signal_Surrogate_Decomposed_Amplitude, \
    #             Array3D_Normalized_Surrogate_Ratio

    def Function_Normalized_Hilbert_Transform(self, Array_Signal):
        """
        Output:
            - Array_Normalized_Hilbert_Transform
            - Array_Normalize_Envelop
        """
        Array_Signal_ABS = numpy.abs(Array_Signal)
        Array_Index_Local_Maxima_ABS, \
            = signal.argrelextrema(Array_Signal_ABS, numpy.greater)
        Function_interpolate_1d \
            = interpolate.CubicSpline\
                            (Array_Index_Local_Maxima_ABS, \
                                Array_Signal_ABS[Array_Index_Local_Maxima_ABS])
        Array_Fit_Envelope \
            = Function_interpolate_1d(numpy.arange(0, Array_Signal.size, 1))
        Array_Normalized_Signal \
            = Array_Signal / Array_Fit_Envelope
        Parameter_Allowed_Error = 0.000001   
        # This error threshold was add to avoid numerical issues, such as a 
        # very samll value larger than one will be ignored in the cubic spline 
        # interpolation
        while numpy.abs(Array_Normalized_Signal\
                            [1:Array_Normalized_Signal.size].max()) \
                        > 1 + Parameter_Allowed_Error:
            Array_Signal_ABS = numpy.abs(Array_Normalized_Signal)
            Array_Index_Local_Maxima_ABS, \
                = signal.argrelextrema(Array_Signal_ABS, numpy.greater_equal)
            Function_interpolate_1d \
                = interpolate.CubicSpline\
                                (Array_Index_Local_Maxima_ABS, \
                                Array_Signal_ABS[Array_Index_Local_Maxima_ABS])
            Array_Fit_Envelope \
                = Function_interpolate_1d(numpy.arange(0, Array_Signal.size, 1))
            Array_Normalized_Signal \
                = Array_Normalized_Signal / Array_Fit_Envelope
        Array_Normalize_Envelop \
            = Array_Signal / Array_Normalized_Signal
        Array_Normalized_Hilbert_Transform \
            = signal.hilbert(Array_Normalized_Signal)
        return Array_Normalized_Hilbert_Transform, Array_Normalize_Envelop

    def Function_Moving_Average(self, Array_Signal, Int_Averaging_Interval):
        ''' 
        Since the moving average should have an averaging interval, however, 
        the averaged number of data points varies
        with the change of the sampling interval, 
        so the averaged number of data points is directly used in this 
        function 
        '''
        Array_Signal_Moving_Average = numpy.zeros(Array_Signal.size)
        Int_Left_Width = Int_Averaging_Interval // 2
        Int_Right_Width = Int_Averaging_Interval - Int_Left_Width

        for i_Time in range(Array_Signal.size):
            if i_Time < Int_Left_Width:
                Array_Signal_Moving_Average[i_Time] \
                    = numpy.mean(Array_Signal[:i_Time + Int_Right_Width])            
            elif i_Time > Array_Signal.size - Int_Right_Width:
                Array_Signal_Moving_Average[i_Time] \
                    = numpy.mean(Array_Signal[i_Time - Int_Left_Width:])
            else:
                Array_Signal_Moving_Average[i_Time] \
                    = numpy.mean(Array_Signal[i_Time - Int_Left_Width \
                                                : i_Time + Int_Right_Width])
        return Array_Signal_Moving_Average

    def Function_Remove_Trend(self, \
            Array_Signal, Str_Wavelet_Name, Level = 0):
        '''
        Remove trend via discrete wavelet
        ------------------------------------------------------------------------
        Input:
            Array_Signal
            STr_Wavelet_Name
            Level = 0
        ------------------------------------------------------------------------
        Output:
            Array_Trend, Array_Fluctuation
        '''
        if Level == 0:
            # print(Array_Signal.size)
            # print(Str_Wavelet_Name)
            # print(pywt.Wavelet(Str_Wavelet_Name))
            # print(pywt.dwt_max_level(Array_Signal.size, \
            #        pywt.Wavelet(Str_Wavelet_Name)))
            Level = pywt.dwt_max_level(Array_Signal.size, \
                                        pywt.Wavelet(Str_Wavelet_Name))
        Tuple_DWT_Coefficients \
            = pywt.wavedec(Array_Signal, Str_Wavelet_Name,'symmetric', Level)
        Tuple_DWT_Coefficients[0][:] = 0
        Array_Signal_Fluctuation \
            = pywt.waverec(Tuple_DWT_Coefficients, \
                            Str_Wavelet_Name, \
                            'symmetric')[:Array_Signal.size]
        Array_Signal_Trend = Array_Signal - Array_Signal_Fluctuation
        return Array_Signal_Trend, Array_Signal_Fluctuation

    def Function_Remove_Amplitude_Modulation(self, \
            Array_Signal, Str_Wavelet_Name, Level = 0):
        '''
        Remove amplitude modulation
        ------------------------------------------------------------------------
        Input:
            Array_Signal
            STr_Wavelet_Name
            Level = 0
        ------------------------------------------------------------------------
        Output:
            Array_Signal_Trend
            Array_Fit_Envelope
            Array_Normalized_Fluctuation
        '''
        if Level == 0:
            Level = pywt.dwt_max_level(Array_Signal.size, \
                                        pywt.Wavelet(Str_Wavelet_Name))
        Tuple_DWT_Coefficients \
            = pywt.wavedec(Array_Signal, Str_Wavelet_Name,'symmetric', Level)
        Tuple_DWT_Coefficients[0][:] = 0
        Array_Signal_Fluctuation \
            = pywt.waverec(Tuple_DWT_Coefficients, Str_Wavelet_Name, \
                            'symmetric')[:Array_Signal.size]
        Array_Signal_Trend = Array_Signal - Array_Signal_Fluctuation
        Array_Signal_Fluctuation_ABS = numpy.abs(Array_Signal_Fluctuation)
        Array_Index_Local_Maxima_ABS, \
            = signal.argrelextrema(Array_Signal_Fluctuation_ABS, numpy.greater)
        Array_Moving_Average_Local_Maxima \
            = self.Function_Moving_Average\
                    (Array_Signal_Fluctuation_ABS\
                        [Array_Index_Local_Maxima_ABS], \
                    2**Level)
        Function_interpolate_1d \
            = interpolate.CubicSpline(Array_Index_Local_Maxima_ABS, \
                                        Array_Moving_Average_Local_Maxima)
        Array_Fit_Envelope \
            = Function_interpolate_1d(numpy.arange(0, Array_Signal.size, 1))
        Array_Normalized_Fluctuation \
            = Array_Signal_Fluctuation / Array_Fit_Envelope
        return Array_Signal_Trend, Array_Fit_Envelope, \
                Array_Normalized_Fluctuation

    def Function_Make_Unit_Length(self, \
            Array_Time, Array_Signal, Int_Unit_Length):
        """
        Description: Pad arbitray length signal with zero to unit lengths
        ------------------------------------------------------------------------
        Input: 
            Array_Time, Array_Signal, Int_Unitlength
        ------------------------------------------------------------------------
        Output: 
            Array_Time_Make_Int_Length, Array_Signal_Make_Int_Length
        """
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Array_Signal_Make_Int_Length \
            = numpy.append\
                    (Array_Signal, \
                        numpy.zeros\
                                ((int(Array_Signal.size / Int_Unit_Length) \
                                        + 1) \
                                    * Int_Unit_Length \
                                - Array_Signal.size))
        Array_Time_Make_Int_Length \
            = numpy.linspace\
                        (0, \
                        Value_Delta_T * Array_Signal_Make_Int_Length.size, \
                        Array_Signal_Make_Int_Length.size)
        return Array_Time_Make_Int_Length, Array_Signal_Make_Int_Length

    def Function_Zero_Padding_Both_Side(self, \
            Array_Time, Array_Signal, Int_Left_Padding, Int_Right_Padding):
        """
        Description: Zero-padding signal with specific lengths
        ------------------------------------------------------------------------
        Input: 
            Array_Time, Array_Signal, Int_Left_Padding, Int_Right_Padding
        ------------------------------------------------------------------------
        Output: 
            Array_Time_Both_Expanded 
            Array_Signal_Both_Expanded
        """
        Value_Delta_T = Array_Time[1] - Array_Time[0]
        Array_Signal_Right_Expanded \
            = numpy.append(Array_Signal, \
                            numpy.zeros(Int_Right_Padding), axis = 0)
        Array_Signal_Both_Expanded \
            = numpy.append(numpy.zeros(Int_Left_Padding), \
                            Array_Signal_Right_Expanded, axis = 0)
        Array_Time_Both_Expanded \
            = numpy.linspace\
                        (- Value_Delta_T * Int_Left_Padding, \
                        Value_Delta_T * Array_Signal_Right_Expanded.size, \
                        Array_Signal_Both_Expanded.size)
        return Array_Time_Both_Expanded, Array_Signal_Both_Expanded



class Class_Signal_Processing_DMD():
    """
    Calculation dynamic mode, amplitude and related info
    ----------------------------------------------------------------------------
    Functions list:
        - Function_Update_Data_Reshapping_to_2D_Snapshot
        - Function_Calculation_Snapshot_Reshapping
        - Function_Calculation_Data_Reshapping_to_Original_Shape
        - Function_SVD_Based_Denoising
        - Function_Update_Projected_Eigendecompositon_of_A
        - Function_Update_Amplitude_Calculation
        - Function_Fit_Data
    ----------------------------------------------------------------------------
    Reference Paper: 
        1. Jovanovic-2014-SDMD
        2. Steven-2016-CSAD
    Reference Link:
        1. https://github.com/mathLab/PyDMD/blob/master/pydmd/dmdbase.py
        2. https://github.com/aaren/sparse_dmd
    """
    # Initiation
    def __init__(self, Int_SVD_Rank):
        self.Class_Name \
            = 'Class: Signal Processing\n\tDynamic mode decomposition'
        self.Int_SVD_Rank = Int_SVD_Rank 
        self.ArrayND_Data_Original = None
        self.Array2D_Snapshots = None
        self.Array2D_Data_S = None
        self.Array2D_Data_S_Prime = None
        self.Array2D_Phi = None
        self.ArrayND_Phi_Reshapped = None # Array2D_Reshapped_Dynamic_Modes
        self.Array_Eigen_Value = None
        self.Array2D_Eigen_Vector = None
        self.Array_Alpha = None 
        
    ## Functions
    def Function_Update_Data_Reshapping_to_2D_Snapshot(self):
        """
        Reshape the original N-Dimensional array to 2D Array of snapshots 
        ------------------------------------------------------------------------
        Data source:
            self.ArrayND_Data_Original
        ------------------------------------------------------------------------
        Data update:
            self.Array2D_Snapshots
        """
        ArrayND_Data_Original = self.ArrayND_Data_Original
        if ArrayND_Data_Original.ndim == 2:
            self.Array2D_Snapshots = ArrayND_Data_Original
        else:
            Temp_Int_Col_Number \
                = ArrayND_Data_Original.shape[-1]
            Temp_Int_Row_Number \
                = ArrayND_Data_Original.size // Temp_Int_Col_Number
            Array2D_Snapshots \
                = numpy.zeros([Temp_Int_Row_Number, Temp_Int_Col_Number])
            for i_DMD_Time in range(Temp_Int_Col_Number):
                Array2D_Snapshots[:,i_DMD_Time] \
                    = ArrayND_Data_Original[...,i_DMD_Time]\
                        .reshape(-1, order = 'C')
            self.Array2D_Snapshots = Array2D_Snapshots

    def Function_Calculation_Snapshot_Reshapping(self,\
            Array_Snapshot, Array_Original_Shape):
        """
        Reshape a 1D snapshot to the ND_Array shape
        ------------------------------------------------------------------------
        Input:
            Array_Snapshot
            Array_Original_Shape
        ------------------------------------------------------------------------
        Output:
            ArrayND_Data_Original
        """
        if Array_Snapshot.shape == Array_Original_Shape:
            ArrayND_Snapshot_Original_Shape = Array_Snapshot
        else:
            ArrayND_Snapshot_Original_Shape \
                = Array_Snapshot\
                    .reshape(Array_Original_Shape[:-1], order = 'C')
        return ArrayND_Snapshot_Original_Shape

    def Function_Calculation_Data_Reshapping_to_Original_Shape(self, \
            Array2D_Snapshots, Array_Original_Shape):
        """
        Reshape the 2D snapshots to the shape of the original data
        ------------------------------------------------------------------------
        Input:
            Array2D_Snapshots
        ------------------------------------------------------------------------
        Output:
            ArrayND_Data_Original
        """
        if Array2D_Snapshots.shape == Array_Original_Shape:
            ArrayND_Data_Original_Shape = Array2D_Snapshots
        else:
            ArrayND_Data_Original_Shape = numpy.zeros(Array_Original_Shape)
            for i_DMD_Time in range(Array2D_Snapshots.shape[1]):
                ArrayND_Data_Original_Shape[...,i_DMD_Time] \
                    = self.Function_Calculation_Snapshot_Reshapping(\
                        Array2D_Snapshots[:,i_DMD_Time], \
                        Array_Original_Shape)
        return ArrayND_Data_Original_Shape

    def Function_SVD_Based_Denoising(self):
        pass
        return None

    def Function_Update_Projected_Eigendecompositon_of_A(self):
        """
        Calculate the dynamic mode and eigenvalue of matrix A
        ------------------------------------------------------------------------
        Data source:
            self.Array2D_S
            self.Array2D_S_Prime
        ------------------------------------------------------------------------
        Data update :
            self.Array2D_U
            self.Array_S
            self.Array2D_V
            self.Array2D_Phi
            self.Array_Lambda
        """
        Tuple_Function_Return = numpy.linalg.svd(self.Array2D_Data_S)
        self.Array2D_U = Tuple_Function_Return[0]
        self.Array_S = Tuple_Function_Return[1]
        Array2D_V_asterisk = Tuple_Function_Return[2]
        self.Array2D_V = Array2D_V_asterisk.conj().T
        
        self.Array2D_U_Selected = self.Array2D_U[:,:self.Int_SVD_Rank]
        self.Array_S_Selected = self.Array_S[:self.Int_SVD_Rank]
        self.Array2D_V_Selected = self.Array2D_V[:,:self.Int_SVD_Rank]
        Array_S_Selected_Inverse = 1 / self.Array_S_Selected

        Array2D_A_Tilde \
            = self.Array2D_U_Selected.conj().T\
                .dot(self.Array2D_Data_S_Prime)\
                .dot(self.Array2D_V_Selected)\
                .dot(numpy.diag(Array_S_Selected_Inverse))

        Tuple_Function_Return = numpy.linalg.eig(Array2D_A_Tilde)
        self.Array_Eigen_Value = Tuple_Function_Return[0]
        self.Array2D_Eigen_Vector = Tuple_Function_Return[1]
        self.Array2D_Phi \
            = self.Array2D_Data_S_Prime\
                .dot(self.Array2D_V_Selected)\
                .dot(numpy.diag(Array_S_Selected_Inverse))\
                .dot(self.Array2D_Eigen_Vector)

    def Function_Update_Amplitude_Calculation(self, Calculation_Method):
        """
        Calculate the amplitude of DMD modes
        ------------------------------------------------------------------------
        Data source (method 1):
            self.Array2D_Data_S
            self.Array2D_Phi
            ----
        Data source (method 2)
            self.
        ------------------------------------------------------------------------
        Output:
            Array_Alpha
        ------------------------------------------------------------------------
        Notes:
        There are two methods for the estimation of the amplitude array: 
            1. Optimized for first snapshot
            2. Optimized for all snapshots
        Curretnly, the function calculates using the first method. Something is 
        wrong with the equation of optimized amplitude. So the first snapshot 
        optimization is used for the calculation of amplitude.
        
        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document

        """
        if Calculation_Method == 1:
            # Method 1: First snapshot
            self.Array_Amplitude \
                = numpy.linalg.lstsq(self.Array2D_Phi, self.Array2D_Data_S[:, 0], \
                                        rcond=None)[0]
        elif Calculation_Method == 2:
            # Method 2: Based on Jovanovic-2004-SPDM
            Value_Delta_T = numpy.diff(self.Array_Time).mean()

            Tuple_Function_Return \
                = numpy.meshgrid(numpy.log(self.Array_Eigen_Value) \
                    / Value_Delta_T, self.Array_Time[:-1])
            Temp_Array2D_Grid_X = Tuple_Function_Return[0]
            Temp_Array2D_Grid_Y = Tuple_Function_Return[1]
            Array2D_Vander \
            = numpy.exp(Temp_Array2D_Grid_X * Temp_Array2D_Grid_Y).T

            Temp_Part_1 = self.Array2D_Phi.conj().T.dot(self.Array2D_Phi)
            Temp_Part_2 = Array2D_Vander.dot(Array2D_Vander.conj().T)
            Array2D_P = numpy.multiply(Temp_Part_1, Temp_Part_2)
            Temp_Part_q = Array2D_Vander\
                            .dot(self.Array2D_V_Selected)\
                            .dot(numpy.diag(self.Array_S_Selected.conj()))\
                            .dot(self.Array2D_Phi.T)
            Array_q = numpy.diag(Temp_Part_q)
            self.Array_Amplitude = numpy.linalg.solve(Array2D_P, Array_q)
        else:
            print('Error: Selected amplitude claculation method is invalid')

    def Function_Fit_Data(self, ArrayND_Data_Original, Array_Time):
        self.ArrayND_Data_Original = ArrayND_Data_Original
        self.Array_Time = Array_Time
        self.Function_Update_Data_Reshapping_to_2D_Snapshot()
        self.Array2D_Data_S = self.Array2D_Snapshots[:,:-1]
        self.Array2D_Data_S_Prime = self.Array2D_Snapshots[:,1:]
        self.Function_Update_Projected_Eigendecompositon_of_A()
        self.Function_Update_Amplitude_Calculation(2)
        return None
