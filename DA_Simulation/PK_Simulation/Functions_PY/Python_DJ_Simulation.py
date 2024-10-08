# ## Libraries
# import os
# import numpy
# import time
# from matplotlib import pyplot
# from scipy import io

# from .Parameter_Setting_App_Simulation import Dict_Figure_Parameter 

# from .Sub_Class_Signal_Generation \
#     import Class_Signal_Generation
# from .Sub_Class_Signal_Processing \
#     import Class_Signal_Processing
# from .Sub_Class_Wind_Field_Information \
#     import Class_Wind_Field_Information
# from .Sub_Class_Simulation_Hilbert_Wavelet_POD \
#     import Class_Simulation_Hilbert_wavelet_TV_2DSVD

# Parameter_Int_Selection_Interval = 20
# Parameter_Int_Number_Points = 4

# def Function_DJ_Simulation(\
#         Parameter_Total_Height,\
#         Parameter_Int_Selection_Interval, \
#         Parameter_Int_Number_Points):


#     Array_Time_Record = numpy.array(time.time())
#     Object_Simulation_Hilbert_wavelet_POD \
#         = Class_Simulation_Hilbert_wavelet_POD(Parameter_Total_Height, \
#                 Parameter_Int_Number_Points, \
#                 Parameter_Str_IF_Method, \
#                 Parameter_Str_Wavelet_Name, \
#                 Parameter_Int_SVD_Truncation, \
#                 Parameter_Int_Max_WPT_Level, \
#                 Parameter_Str_Method_for_Obtaining_Std)

#     Object_Signal_Generation = Class_Signal_Generation()
#     Object_Signal_Processing = Class_Signal_Processing()
#     # Object_Wind_Field_Information = Class_Wind_Field_Information()

#     Array_Time_Record = numpy.append(Array_Time_Record, time.time())

#     Value_Interval_Point \
#         = Parameter_Total_Height / Parameter_Int_Number_Points
#     Array_Z_Coordinate \
#         = numpy.linspace(Value_Interval_Point, \
#                         Value_Interval_Point * Parameter_Int_Number_Points, \
#                         Parameter_Int_Number_Points)
#     Tuple_Function_Return \
#         = Object_Signal_Generation\
#             .Function_Load_Wind_Speed(Parameter_Str_Wind_File_Name)
#     Array_Time = Tuple_Function_Return[0]
#     Array_Signal = Tuple_Function_Return[1]
#     Value_Sampling_Frequency = Tuple_Function_Return[2]
#     Value_Delta_T = Tuple_Function_Return[3]

#     Temp_Int_Index_Extension_to_Avoid_Edge_Effect = int(300)
#     Array_Time \
#         = Array_Time[Parameter_Int_i_Start \
#                         - Temp_Int_Index_Extension_to_Avoid_Edge_Effect \
#                     :Parameter_Int_i_Ended \
#                         + Temp_Int_Index_Extension_to_Avoid_Edge_Effect] 
#     Array_Signal \
#         = Array_Signal[Parameter_Int_i_Start \
#                             - Temp_Int_Index_Extension_to_Avoid_Edge_Effect \
#                     :Parameter_Int_i_Ended \
#                             + Temp_Int_Index_Extension_to_Avoid_Edge_Effect] 
#     # Array_Signal = Array_Signal + 15

#     Array_Time = Array_Time[::Parameter_Int_Selection_Interval]
#     Array_Signal = Array_Signal[::Parameter_Int_Selection_Interval]
#     Value_Delta_T = Array_Time[1] - Array_Time[0]
#     Value_Sampling_Frequency = 1 / Value_Delta_T
#     Array_Time = Array_Time - Array_Time.min()

#     Object_Simulation_Hilbert_wavelet_POD\
#         .Function_CDU_Input_Data(Array_Time, Array_Signal)
#     Object_Simulation_Hilbert_wavelet_POD\
#         .Function_CDU_Decomposition_IA_IF()
#     Object_Simulation_Hilbert_wavelet_POD\
#         .Function_CDU_Target_Spatial_Correlation(Array_Z_Coordinate)

#     Object_Simulation_Hilbert_wavelet_POD.Function_CDU_2DSVD_Eigendecomposition()
#     Array_Time_Record = numpy.append(Array_Time_Record, time.time())
#     print('Time consumption: {:.4f}s'.format(numpy.diff(Array_Time_Record)[-1]))
#     Array_Time_Record = numpy.append(Array_Time_Record, time.time())
#     Object_Simulation_Hilbert_wavelet_POD.Function_CDU_Simulation_All_Scales()
#     Array_Time_Record = numpy.append(Array_Time_Record, time.time())
#     print('Time consumption: {:.4f}s'.format(numpy.diff(Array_Time_Record)[-1]))
#     if os.path.isfile('static/Simu_X.mat'):
#         print('Privious simulation file exist')
#         print('Replaced with new file')
#     io.savemat('Data_Simulation/Simu_X.mat', \
#                 {'Array2D_Simulation'\
#                     :Object_Simulation_Hilbert_wavelet_POD.Array2D_Simulation,\
#                 'Array_Time'\
#                     :Object_Simulation_Hilbert_wavelet_POD.Array_Time,\
#                 'Array_Z_Coordinate'\
#                     :Array_Z_Coordinate})

#     Temp_Fig_Array_0 \
#         = Object_Simulation_Hilbert_wavelet_POD.Array_Time
#     Temp_Fig_Array2D_1 \
#         = Object_Simulation_Hilbert_wavelet_POD.Array2D_Simulation
#     pyplot.figure(figsize = (4,3), dpi = 100)
#     pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array2D_1, linewidth = 0.2, alpha = 0.7)
#     pyplot.xlabel('Time (s)', fontsize = 6)
#     pyplot.ylabel('Wind speed (m/s)', fontsize = 6)
#     pyplot.xticks(fontsize = 6)
#     pyplot.yticks(fontsize = 6)
#     pyplot.grid(True, alpha = 0.2)
#     pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
#     pyplot.savefig('static/Figure_5_All_Simu.png', transparent = True)
#     pyplot.close('all')

#     Temp_Fig_Array_0 \
#         = Array_Time
#     Temp_Fig_Array_1 \
#         = Temp_Fig_Array2D_1[:,0]
#     pyplot.figure(figsize = (4,3), dpi = 100)
#     pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array_1, linewidth = 0.9, alpha = 0.99)
#     pyplot.xlabel('Time (s)', fontsize = 5)
#     pyplot.ylabel('Wind speed (m/s)', fontsize = 5)
#     pyplot.grid(True, alpha = 0.2)
#     pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
#     pyplot.tight_layout()
#     pyplot.savefig('static/Figure_3_History.png', transparent = True)
#     pyplot.close('all')
#     return True