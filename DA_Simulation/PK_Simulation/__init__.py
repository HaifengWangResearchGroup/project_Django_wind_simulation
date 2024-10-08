## Libraries
import os
import numpy
import time
from matplotlib import pyplot
from scipy import io


from .Functions_PY import Dict_Figure_Parameter 
from .Functions_PY import Dict_Simulation_Parameter

from .Functions_PY \
    import Class_Signal_Generation
from .Functions_PY \
    import Class_Signal_Processing
from .Functions_PY \
    import Class_Wind_Field_Information
from .Functions_PY \
    import Class_Simulation_Hilbert_wavelet_TV_2DSVD
from .Functions_PY \
    import Class_Simulation_Hilbert_wavelet_CT_POD

Parameter_Str_Wavelet_Name \
    = Dict_Simulation_Parameter['Parameter_Str_Wavelet_Name']
Parameter_Total_Height \
    = Dict_Simulation_Parameter['Parameter_Total_Height']
Parameter_Str_Wind_File_Name \
    = Dict_Simulation_Parameter['Parameter_Str_Wind_File_Name']
Parameter_Int_i_Start \
    = Dict_Simulation_Parameter['Parameter_Int_i_Start']
Parameter_Int_i_Ended \
    = Dict_Simulation_Parameter['Parameter_Int_i_Ended']
Parameter_Str_Wavelet_Name \
    = Dict_Simulation_Parameter['Parameter_Str_Wavelet_Name']
Parameter_Int_SVD_Truncation \
    = Dict_Simulation_Parameter['Parameter_Int_SVD_Truncation']
Parameter_Int_Max_WPT_Level \
    = Dict_Simulation_Parameter['Parameter_Int_Max_WPT_Level']
Parameter_Str_Method_for_Obtaining_Std \
    = Dict_Simulation_Parameter['Parameter_Str_Method_for_Obtaining_Std']
Parameter_Str_IF_Method \
    = Dict_Simulation_Parameter['Parameter_Str_IF_Method']

Parameter_Int_Selection_Interval \
    = Dict_Simulation_Parameter['Parameter_Int_Selection_Interval']
Parameter_Int_Number_Points \
    = Dict_Simulation_Parameter['Parameter_Int_Number_Points']

Parameter_Int_Selection_Interval = 20
Parameter_Int_Number_Points = 4

def Function_DJ_Simulation_TV_2DSVD(\
        Parameter_HS, \
        Parameter_HL, \
        Parameter_VS, \
        Parameter_VL,\
        Parameter_Int_Selection_Interval,
        str_file_name = ''):

    Array_Time_Record = numpy.array(time.time())
    Object_Signal_Generation = Class_Signal_Generation()
    Object_Wind_Field_Information = Class_Wind_Field_Information()

    if Parameter_HL != 1:
        print('Error! Cannot handle horizontal distribution in TV corr')
        return None

    Array2D_Locations \
        = Object_Wind_Field_Information.Functino_2D_Location_Generation(\
            1, 0, 0, \
            Parameter_HL, 0, Parameter_HS, \
            Parameter_VL, 0, Parameter_VS)

    Object_Simulation_Hilbert_wavelet_POD \
        = Class_Simulation_Hilbert_wavelet_TV_2DSVD(\
                Array2D_Locations, \
                Parameter_Str_IF_Method, \
                Parameter_Str_Wavelet_Name, \
                Parameter_Int_SVD_Truncation, \
                Parameter_Int_Max_WPT_Level, \
                Parameter_Str_Method_for_Obtaining_Std)

    Tuple_Function_Return \
        = Object_Signal_Generation\
            .Function_Load_Wind_Speed(Parameter_Str_Wind_File_Name)
    Array_Time = Tuple_Function_Return[0]
    Array_Signal = Tuple_Function_Return[1]

    Temp_Int_Index_Extension_to_Avoid_Edge_Effect = int(300)
    Array_Time \
        = Array_Time[Parameter_Int_i_Start \
                        - Temp_Int_Index_Extension_to_Avoid_Edge_Effect \
                    :Parameter_Int_i_Ended \
                        + Temp_Int_Index_Extension_to_Avoid_Edge_Effect] 
    Array_Signal \
        = Array_Signal[Parameter_Int_i_Start \
                            - Temp_Int_Index_Extension_to_Avoid_Edge_Effect \
                    :Parameter_Int_i_Ended \
                            + Temp_Int_Index_Extension_to_Avoid_Edge_Effect] 

    Array_Time = Array_Time[::Parameter_Int_Selection_Interval]
    Array_Signal = Array_Signal[::Parameter_Int_Selection_Interval]
    Array_Time = Array_Time - Array_Time.min()

    Array_Time_Record = numpy.array(time.time())
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Input_Data(Array_Time, Array_Signal)
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Decomposition_IA_IF()
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Target_Spatial_Correlation('Jiang_2017')
    Array_Time_Record = numpy.append(Array_Time_Record, time.time())
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Corr_Decomposition()
    Array_Time_Record = numpy.append(Array_Time_Record, time.time())
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Simulation_All_Scales()
    Array_Time_Record = numpy.append(Array_Time_Record, time.time())
    print('Time Records: {}'.format(numpy.diff(Array_Time_Record)))

    if str_file_name == '':
        str_file_name = 'Simu_X.mat'

    if os.path.isfile('static/' + str_file_name):
        print('Previous simulation file exist')
        print('Replaced with new file')

    io.savemat('Data_Simulation/' + str_file_name, \
                {'Array2D_Simulation'\
                    :Object_Simulation_Hilbert_wavelet_POD.Array2D_Simulation,\
                'Array_Time'\
                    :Object_Simulation_Hilbert_wavelet_POD.Array_Time,\
                'Array2D_Locations'\
                    :Array2D_Locations})

    Temp_Fig_Array_0 \
        = Object_Simulation_Hilbert_wavelet_POD.Array_Time
    Temp_Fig_Array2D_1 \
        = Object_Simulation_Hilbert_wavelet_POD.Array2D_Simulation
    pyplot.figure(figsize = (4,3), dpi = 100)
    pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array2D_1, linewidth = 0.2, alpha = 0.7)
    pyplot.xlabel('Time (s)', fontsize = 6)
    pyplot.ylabel('Wind speed (m/s)', fontsize = 6)
    pyplot.xticks(fontsize = 6)
    pyplot.yticks(fontsize = 6)
    pyplot.grid(True, alpha = 0.2)
    pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
    pyplot.savefig('static/Figure_5_All_Simu.png', transparent = True)
    pyplot.close('all')

    Temp_Fig_Array_0 \
        = Array_Time
    Temp_Fig_Array_1 \
        = Temp_Fig_Array2D_1[:,0]
    pyplot.figure(figsize = (4,3), dpi = 100)
    pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array_1, linewidth = 0.9, alpha = 0.99)
    pyplot.xlabel('Time (s)', fontsize = 5)
    pyplot.ylabel('Wind speed (m/s)', fontsize = 5)
    pyplot.grid(True, alpha = 0.2)
    pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
    pyplot.tight_layout()
    pyplot.savefig('static/Figure_3_History.png', transparent = True)
    pyplot.close('all')
    return True


def Function_DJ_Simulation_CT_POD(\
        Parameter_HS, \
        Parameter_HL, \
        Parameter_VS, \
        Parameter_VL,\
        Parameter_Int_Selection_Interval, 
        str_file_name = ''):

    Array_Time_Record = numpy.array(time.time())
    Object_Signal_Generation = Class_Signal_Generation()
    Object_Wind_Field_Information = Class_Wind_Field_Information()

    Array2D_Locations \
        = Object_Wind_Field_Information.Functino_2D_Location_Generation(\
            1, 0, 0, \
            Parameter_HL, 0, Parameter_HS, \
            Parameter_VL, 0, Parameter_VS)

    Object_Simulation_Hilbert_wavelet_POD \
        = Class_Simulation_Hilbert_wavelet_CT_POD(\
                Array2D_Locations, \
                Parameter_Str_IF_Method, \
                Parameter_Str_Wavelet_Name, \
                Parameter_Int_SVD_Truncation, \
                Parameter_Int_Max_WPT_Level, \
                Parameter_Str_Method_for_Obtaining_Std)

    Tuple_Function_Return \
        = Object_Signal_Generation\
            .Function_Load_Wind_Speed(Parameter_Str_Wind_File_Name)
    Array_Time = Tuple_Function_Return[0]
    Array_Signal = Tuple_Function_Return[1]

    Temp_Int_Index_Extension_to_Avoid_Edge_Effect = int(300)
    Array_Time \
        = Array_Time[Parameter_Int_i_Start \
                        - Temp_Int_Index_Extension_to_Avoid_Edge_Effect \
                    :Parameter_Int_i_Ended \
                        + Temp_Int_Index_Extension_to_Avoid_Edge_Effect] 
    Array_Signal \
        = Array_Signal[Parameter_Int_i_Start \
                            - Temp_Int_Index_Extension_to_Avoid_Edge_Effect \
                    :Parameter_Int_i_Ended \
                            + Temp_Int_Index_Extension_to_Avoid_Edge_Effect] 

    Array_Time = Array_Time[::Parameter_Int_Selection_Interval]
    Array_Signal = Array_Signal[::Parameter_Int_Selection_Interval]
    Array_Time = Array_Time - Array_Time.min()
    
    Array_Time_Record = numpy.array(time.time())
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Input_Data(Array_Time, Array_Signal)
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Decomposition_IA_IF()
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Target_Spatial_Correlation('Hanse_1999')
    Array_Time_Record = numpy.append(Array_Time_Record, time.time())
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Corr_Decomposition()
    Array_Time_Record = numpy.append(Array_Time_Record, time.time())
    Object_Simulation_Hilbert_wavelet_POD\
        .Function_CDU_Simulation_All_Scales()
    Array_Time_Record = numpy.append(Array_Time_Record, time.time())
    print('Time Records: {}'.format(numpy.diff(Array_Time_Record)))

    if str_file_name == '':
        str_file_name = 'Simu_X.mat'

    if os.path.isfile('static/' + str_file_name):
        print('Privious simulation file exist')
        print('Replaced with new file')
    io.savemat('Data_Simulation/' + str_file_name, \
                {'Array2D_Simulation'\
                    :Object_Simulation_Hilbert_wavelet_POD.Array2D_Simulation,\
                'Array_Time'\
                    :Object_Simulation_Hilbert_wavelet_POD.Array_Time,\
                'Array2D_Locations'\
                    :Array2D_Locations})

    Temp_Fig_Array_0 \
        = Object_Simulation_Hilbert_wavelet_POD.Array_Time
    Temp_Fig_Array2D_1 \
        = Object_Simulation_Hilbert_wavelet_POD.Array2D_Simulation

    pyplot.figure(figsize = (4,3), dpi = 100)
    pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array2D_1, linewidth = 0.2, alpha = 0.7)
    pyplot.xlabel('Time (s)', fontsize = 6)
    pyplot.ylabel('Wind speed (m/s)', fontsize = 6)
    pyplot.xticks(fontsize = 6)
    pyplot.yticks(fontsize = 6)
    pyplot.grid(True, alpha = 0.2)
    pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
    pyplot.savefig('static/Figure_5_All_Simu.png', transparent = True)
    pyplot.close('all')

    Temp_Fig_Array_0 \
        = Array_Time
    Temp_Fig_Array_1 \
        = Temp_Fig_Array2D_1[:,0]
    pyplot.figure(figsize = (4,3), dpi = 100)
    pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array_1, linewidth = 0.9, alpha = 0.99)
    pyplot.xlabel('Time (s)', fontsize = 5)
    pyplot.ylabel('Wind speed (m/s)', fontsize = 5)
    pyplot.grid(True, alpha = 0.2)
    pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
    pyplot.tight_layout()
    pyplot.savefig('static/Figure_3_History.png', transparent = True)
    pyplot.close('all')
    return True