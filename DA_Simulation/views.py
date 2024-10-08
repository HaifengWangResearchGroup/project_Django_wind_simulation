# Create your views here.

## Import python built-in
import os
import numpy
from scipy import io
from matplotlib import pyplot
## Import django built-in
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.http import Http404
from django.template import loader
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.views.generic import TemplateView
# from django.

## Import self-defined code - Django
from .forms import HomeForm
from .models import Class_Data
# Import self-defined code - Python
from .PK_Simulation import Function_DJ_Simulation_TV_2DSVD
from .PK_Simulation import Function_DJ_Simulation_CT_POD

# def simple_upload(request):
#     if request.method == 'POST' and request.FILES['Input_Upload_File']:
#         Input_Upload_File = request.FILES['Input_Upload_File']
#         fs = FileSystemStorage()
#         filename = fs.save(Input_Upload_File.name, Input_Upload_File)
#         uploaded_file_url = fs.url(filename)
#         return render(request, 'core/simple_upload.html', {
#             'uploaded_file_url': uploaded_file_url
#         })
#     return render(request, 'core/simple_upload.html')

class Class_Simulation_Page(TemplateView):
    Parameter_Max_Locations = 30
    Str_Template_Name = 'Page_Simulation.html'
    Template = loader.get_template(Str_Template_Name)
    # print(request.scheme)
    # print(request.method)
    # print(request.GET)
    # print(request.POST)
    Dict_Args \
        = {\
            'Status_Select_CM': '',}
    Dict_Simulation_Parameters \
        = {}

    def get(self, request):
        # print('Get_Function_Print_Get', request.GET)
        # print('Get_Function_Print_POST', request.POST)
        # if request.GET.get('Btn_Simu'):
        #     print('Btn_Clicked')
        List_CM \
            = [["", "disabled selected hidden" ,"Select correlation model" ], \
                ["TIC", "" ,"Time-invariant correlation" ], \
                ["TVC", "" , "Time-variant correlation" ]]

        Dict_Args_Modify \
            = {'Status_Select_CM': 'enabled', \
                'List_CM':List_CM, \
                'Status_PS_Row_H': 'hidden', \
                'Status_PS_Row_H_Col_HS': '', \
                'Status_PS_Row_H_Col_HL': '', \
                'Status_PS_Row_V': 'hidden', \
                'Status_PS_Row_V_Col_VS': '', \
                'Status_PS_Row_V_Col_VL': '',\
                'Status_PS_Row_SF': 'hidden', \
                'Status_PS_Row_HSM': 'hidden', \
                'Status_PS_Row_TVMM': 'hidden', \
                'Status_PS_Row_SBTN': 'hidden', \
                'Status_Btn_PS_HVS': 'hidden', \
                'Status_Fig_SL': 'hidden', \
                'Status_Fig_TC': 'hidden', \
                'Status_Fig_TVM': 'hidden', \
                'Status_Fig_HSM': 'hidden', \
                'Status_Validation_Time_Domain': 'hidden', \
                'Status_Validation':'hidden', \
                'Status_Show_Simu_Sele': 'disabled', \
                'Str_Fig_6_Corr_Path': '/static/Figure_6_Corr_Tar.png', \
                'Parameter_Max_Locations': self.Parameter_Max_Locations}
        self.Dict_Args.update(Dict_Args_Modify)
        return render(request, self.Str_Template_Name, self.Dict_Args)
        # return render(request, self.Str_Template_Name, {'form':form})

    def post(self, request):
        print('Test-Request', request.user)
        print('Test-Request-Cookies', request.COOKIES)
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')
        else:
            ip = request.META.get('REMOTE_ADDR')
        print('Test-Request', ip)

        form = HomeForm(request.POST)
        print('POST_Function_Print_POST', request.POST)
        Dict_POST_Parameters = request.POST
        print(Dict_POST_Parameters)
        
        if Dict_POST_Parameters['Name_Submit_Source'] == 'Select_CM':
            Str_Selected_CM = Dict_POST_Parameters['Name_Select_CM']

            Value_HS = '100 m'
            Value_VS = '100 m'
            List_Select_HL = numpy.arange(1,20,1)
            List_Select_VL = numpy.arange(1,20,1)
            List_Value_SF = [0.5, 1, 2, 4]

            Dict_Args_Modify \
                = {'CM':Str_Selected_CM, \
                    'Value_HS': Value_HS, \
                    'List_Select_HL':List_Select_HL, \
                    'Value_VS': Value_VS, \
                    'List_Select_VL':List_Select_VL, \
                    'Status_PS_Row_SF': 'shown', \
                    'Status_Btn_PS_HVS': 'shown', \
                    'List_Value_SF':List_Value_SF, \
                    'Status_Select_SF': 'enabled', \
                    'Status_Fig_SL': 'shown', \
                    'Status_Fig_TC': 'shown'}
            self.Dict_Args.update(Dict_Args_Modify)

            if Str_Selected_CM == 'TIC':
                self.Dict_Simulation_Parameters['CM'] = 'TIC'
                List_CM \
                    = [["", "" ,"Time-invariant correlation" ],]
                Dict_Args_Modify \
                    = {'Status_Select_CM':'disabled', \
                        'List_CM':List_CM, \
                        'Status_PS_Row_H': '', \
                        'Status_PS_Row_V': '', \
                        'Str_Fig_6_Corr_Path': '/static/Figure_6_Corr_Tar_CT.png'}
            elif Str_Selected_CM == 'TVC':
                self.Dict_Simulation_Parameters['CM'] = 'TVC'
                List_CM \
                    = [["", "disabled" ,"Time-variant correlation" ],]
                Dict_Args_Modify \
                    = {'Status_Select_CM':'disabled', \
                        'List_CM':List_CM, \
                        'Status_PS_Row_H': 'hidden', \
                        'Status_PS_Row_V': '', \
                        'Str_Fig_6_Corr_Path': '/static/Figure_6_Corr_Tar.png'}
            self.Dict_Args.update(Dict_Args_Modify)
            return render(request, self.Str_Template_Name, self.Dict_Args)        

        elif Dict_POST_Parameters['Name_Submit_Source'] == 'Select_PS_HVS':
            Dict_Args_Modify \
                = {'Status_Btn_PS_HVS': 'hidden', \
                    'Status_Fig_TVM': 'shown', \
                    'Status_Fig_HSM': 'shown', \
                    'Status_PS_Row_HSM': 'shown', \
                    'Status_PS_Row_TVMM': 'shown', \
                    'Status_PS_Row_SBTN': 'shown', \
                    'Status_PS_Row_H_Col_HS': 'disabled', \
                    'Status_PS_Row_H_Col_HL': 'disabled', \
                    'Status_PS_Row_V_Col_VS': 'disabled', \
                    'Status_PS_Row_V_Col_VL': 'disabled', \
                    'Status_Select_SF': 'disabled' \
                    }
            self.Dict_Args.update(Dict_Args_Modify)

            Parameter_HS \
                = float(Dict_POST_Parameters['Name_Input_HS'])
            Parameter_HL \
                = int(Dict_POST_Parameters['Name_Select_HL'])            
            Parameter_VS \
                = float(Dict_POST_Parameters['Name_Input_VS'])
            Parameter_VL \
                = int(Dict_POST_Parameters['Name_Select_VL'])
            Parameter_Sampling_Frequency \
                = float(Dict_POST_Parameters['Name_Select_SF'])
            
            self.Dict_Simulation_Parameters['Parameter_HS'] \
                = Parameter_HS
            self.Dict_Simulation_Parameters['Parameter_HL'] \
                = Parameter_HL
            self.Dict_Simulation_Parameters['Parameter_VS'] \
                = Parameter_VS
            self.Dict_Simulation_Parameters['Parameter_VL'] \
                = Parameter_VL
            self.Dict_Simulation_Parameters['Parameter_Sampling_Frequency'] \
                = Parameter_Sampling_Frequency


            Value_HS = '{} m'.format(Parameter_HS)
            Value_VS = '{} m'.format(Parameter_VS)
            List_Select_HL = [Parameter_HL,]
            List_Select_VL = [Parameter_VL,]
            List_Value_SF = [Parameter_Sampling_Frequency,]

            Dict_Args_Modify \
                = {\
                    'Value_VS': Value_VS, \
                    'Value_HS': Value_HS, \
                    'List_Select_HL':List_Select_HL, \
                    'List_Select_VL':List_Select_VL, \
                    'List_Value_SF':List_Value_SF, \
                    }
            self.Dict_Args.update(Dict_Args_Modify)

            return render(request, self.Str_Template_Name, self.Dict_Args)     

        elif Dict_POST_Parameters['Name_Submit_Source'] == 'Click_BTN_Simu':
            Parameter_HS \
                = self.Dict_Simulation_Parameters['Parameter_HS']
            Parameter_HL \
                = self.Dict_Simulation_Parameters['Parameter_HL']
            Parameter_VS \
                = self.Dict_Simulation_Parameters['Parameter_VS']
            Parameter_VL \
                = self.Dict_Simulation_Parameters['Parameter_VL']
            Parameter_Sampling_Frequency \
                = self.Dict_Simulation_Parameters['Parameter_Sampling_Frequency']
            Parameter_Int_Selection_Interval \
                = int( 1 / Parameter_Sampling_Frequency * 10)
            Parameter_Str_CM \
                    = self.Dict_Simulation_Parameters['CM']
            if Parameter_Str_CM == 'TVC':
                Function_DJ_Simulation_TV_2DSVD(\
                    Parameter_HS, \
                    Parameter_HL, \
                    Parameter_VS, \
                    Parameter_VL,\
                    Parameter_Int_Selection_Interval)
            elif Parameter_Str_CM == 'TIC':
                Function_DJ_Simulation_CT_POD(\
                    Parameter_HS, \
                    Parameter_HL, \
                    Parameter_VS, \
                    Parameter_VL,\
                    Parameter_Int_Selection_Interval)

            List_Option_HI = range(Parameter_HL + 1)[1:]
            List_Option_VI = range(Parameter_VL + 1)[1:]       


            Dict_Args_Modify \
                = {\
                    'Status_Validation_Time_Domain': 'shown', \
                    'Status_Validation': 'hidden', \
                    'List_Option_HI': List_Option_HI, \
                    'List_Option_VI': List_Option_VI, \
                    'Status_Option_VI': 'disabled', \
                    }
            self.Dict_Args.update(Dict_Args_Modify)
            return render(request, self.Str_Template_Name, self.Dict_Args)          


        # if 'Name_Btn_Simu' in Dict_POST_Parameters:
        #     # Value_VS = '100 m'
        #     # 'Value_VS':Value_VS, \
        #     try:
        #         Dict_POST_Parameters['Name_Btn_Simu'] == 'Click_Simu'
        #         # Name_Input_HS \
        #         #     = int(Dict_POST_Parameters['Name_Input_HS'])
        #         Parameter_Total_Height \
        #             = float(Dict_POST_Parameters['Name_Input_VS'])
        #         Parameter_Int_Number_Points \
        #             = int(Dict_POST_Parameters['Name_Select_VL'])
        #         Parameter_Sampling_Frequency \
        #             = float(Dict_POST_Parameters['Name_Select_SF'])
        #         Dict_POST_Parameters['Name_Select_SF']   
        #         List_Option_HI = range(1 + 1)[1:]
        #         List_Option_VI = range(Parameter_Int_Number_Points + 1)[1:]
        #         Dict_Args \
        #             = {'List_Valeu_HL':[1], \
        #                 'Value_VS':Parameter_Total_Height, \
        #                 'Status_PS_Row_V_Col_VS': 'disabled', \
        #                 'List_Select_VL': [Parameter_Int_Number_Points],\
        #                 'Status_PS_Row_V_Col_VL': 'disabled',\
        #                 'List_Value_SF': [Parameter_Sampling_Frequency], \
        #                 'Status_Select_SF': 'disabled', \
        #                 'Hide_Validation':'', \
        #                 'List_Option_HI':List_Option_HI, \
        #                 'Status_Option_HI': '', 
        #                 'List_Option_VI':List_Option_VI, \
        #                 'Status_Option_VI': ''}
                
        #         Parameter_Int_Selection_Interval \
        #             = int( 1 / Parameter_Sampling_Frequency * 10)
        #         print(\
        #             Parameter_Total_Height, \
        #             Parameter_Int_Selection_Interval, \
        #             Parameter_Int_Number_Points)
        #         Function_DJ_Simulation(\
        #             Parameter_Total_Height, \
        #             Parameter_Int_Selection_Interval, \
        #             Parameter_Int_Number_Points)
        #         return render(request, self.Str_Template_Name, Dict_Args)             
        #     except:
        #         print('The wrong input value')
        #         Str_PH_Input_HS = 'Wrong Value'
        #         Dict_Args \
        #             = {'Str_PH_Input_HS':Str_PH_Input_HS}
        #         return render(request, self.Str_Template_Name, Dict_Args)
        
        # if 'Name_Btn_Show' in Dict_POST_Parameters:
        #     try:
        #         Dict_POST_Parameters['Name_Btn_Show'] \
        #             == 'Click_Show'
        #         Value_Index \
        #             = int(Dict_POST_Parameters['Name_Show_VI'])
        #         RawData \
        #             = io.loadmat('Data_Simulation/Simu_X.mat')
        #         Array2D_Simulation \
        #             = RawData['Array2D_Simulation']
        #         Array_Time \
        #             = numpy.squeeze(RawData['Array_Time'])
        #         Array_Z_Coordinate \
        #             = numpy.squeeze(RawData['Array_Z_Coordinate'])
        #         Temp_Fig_Array_0 \
        #             = Array_Time
        #         Temp_Fig_Array_1 \
        #             = Array2D_Simulation[:,Value_Index]
        #         pyplot.figure(figsize = (4,3), dpi = 100)
        #         pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array_1, \
        #                     linewidth = 0.9, alpha = 0.99)
        #         pyplot.xlabel('Time (s)', fontsize = 5)
        #         pyplot.ylabel('Wind speed (m/s)', fontsize = 5)
        #         pyplot.grid(True, alpha = 0.2)
        #         pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
        #         pyplot.tight_layout()
        #         pyplot.savefig('static/Figure_3_History.png', \
        #                         transparent = True)
        #         pyplot.close('all')
        #         print('Figure Saved')
        #         Parameter_Total_Height \
        #             = Array_Z_Coordinate.max()
        #         Parameter_Int_Number_Points \
        #             = Array_Z_Coordinate.size
        #         Parameter_Sampling_Frequency \
        #             = 1 / (Array_Time[1] - Array_Time[0])
        #         List_Option_HI = [1]
        #         List_Option_VI = [Value_Index]
        #         Dict_Args \
        #             = {'List_Valeu_HL':[1], \
        #                 'Value_VS':Parameter_Total_Height, \
        #                 'Status_PS_Row_V_Col_VS': 'disabled', \
        #                 'List_Select_VL': [Parameter_Int_Number_Points],\
        #                 'Status_PS_Row_V_Col_VL': 'disabled',\
        #                 'List_Value_SF': [Parameter_Sampling_Frequency], \
        #                 'Status_Select_SF': 'disabled', \
        #                 'Hide_Validation':'', \
        #                 'List_Option_HI':List_Option_HI, \
        #                 'Status_Option_HI': 'disabled', \
        #                 'List_Option_VI':List_Option_VI, \
        #                 'Status_Option_VI': 'disabled', \
        #                 'Status_Show_Simu_Sele': 'disabled'}                      
        #         return render(request, self.Str_Template_Name, Dict_Args)
        #     except:
        #         print('Something went wrong')


def Function_detail(request, User_ID):
    Str_HTTP_Response \
        = "This is the request from user{}".format(User_ID)
    return HttpResponse(Str_HTTP_Response)

def Function_Content(request, User_ID):
    List_Object_Latest_Data = Class_Data.objects.order_by('Data_Date')[:5]
    List_String_Latest = []
    for Object_Data in List_Object_Latest_Data:
        List_String_Latest.append(Object_Data.Data_Name)
    # Str_Output = ', '.join(List_String_Latest)
    Object_Template = loader.get_template('DA_Simulation/index.html')
    context = {
        'List_Latest_Data':List_Object_Latest_Data,
    }
    return HttpResponse(Object_Template.render(context, request))

def Function_Set_Col_1(request):
    Object_Data = get_object_or_404(Class_Data)
    try:
        selected_choice \
            = Object_Data.choice_set.get(pk = request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'DA_Simulation/detail.html', {
            'question': Object_Data,
            'error_message': "You didn't select a choice.",
        })
    return HttpResponse(Object_Template.render(context, request))

# def Function_Test_Template(request):
#     Str_Template_Name = 'Page_Simulation.html'
#     Template = loader.get_template(Str_Template_Name)
#     # print(request.scheme)
#     # print(request.method)
#     print(request.GET)
#     print(request.POST)

#     def get(self, request):
#         form = HomeForm()
#         if request.GET.get('Btn_Simu'):
#             print('Btn_Clicked')
#         return render(request, self.Str_Template_Name, {'form':form})

#     def post(self, request):
#         form = HomeForm(request.POST)
#         if form.is_valid():
#             text = form.cleaned_data
        
#         Dict_Args = {'form':form, 'text':text}
#         return render(request, self.Str_Template_Name, {'form':text})

#     return HttpResponse( render(request, Template.template.name))

# class Class_Test_Template(TemplateView):
#     Str_Template_Name = 'Page_Simulation.html'
#     Template = loader.get_template(Str_Template_Name)
#     # print(request.scheme)
#     # print(request.method)
#     # print(request.GET)
#     # print(request.POST)

#     def get(self, request):
#         form = HomeForm()
#         # print('Get_Function_Print_Get', request.GET)
#         # print('Get_Function_Print_POST', request.POST)
#         # if request.GET.get('Btn_Simu'):
#         #     print('Btn_Clicked')
#         Value_VS = '100 m'
#         List_Valeu_HL = [1]
#         List_Select_VL = [2,4,8,16]
#         List_Value_SF = [0.5, 1, 2, 4]
#         Dict_Args \
#             = {\
#                 'Value_VS':Value_VS, \
#                 'Status_PS_Row_V_Col_VS': '', \
#                 'List_Valeu_HL':List_Valeu_HL, \
#                 'List_Select_VL':List_Select_VL, \
#                 'Status_PS_Row_V_Col_VL': '',\
#                 'List_Value_SF':List_Value_SF, \
#                 'Status_Select_SF': '', \
#                 'Hide_Validation':'hidden'}
#         return render(request, self.Str_Template_Name, Dict_Args)
#         # return render(request, self.Str_Template_Name, {'form':form})

#     def post(self, request):
#         form = HomeForm(request.POST)
#         print('POST_Function_Print_POST', request.POST)
#         if form.is_valid():
#             text = form.cleaned_data
#         # text = '(m)'
#         Dict_POST_Parameters = request.POST
#         print(Dict_POST_Parameters)
 
#         if 'Name_Btn_Simu' in Dict_POST_Parameters:
#             try:
#                 Dict_POST_Parameters['Name_Btn_Simu'] == 'Click_Simu'
#                 # Name_Input_HS \
#                 #     = int(Dict_POST_Parameters['Name_Input_HS'])
#                 Parameter_Total_Height \
#                     = float(Dict_POST_Parameters['Name_Input_VS'])
#                 Parameter_Int_Number_Points \
#                     = int(Dict_POST_Parameters['Name_Select_VL'])
#                 Parameter_Sampling_Frequency \
#                     = float(Dict_POST_Parameters['Name_Select_SF'])
#                 Dict_POST_Parameters['Name_Select_SF']   
#                 List_Option_HI = range(1 + 1)[1:]
#                 List_Option_VI = range(Parameter_Int_Number_Points + 1)[1:]
#                 Dict_Args \
#                     = {'List_Valeu_HL':[1], \
#                         'Value_VS':Parameter_Total_Height, \
#                         'Status_PS_Row_V_Col_VS': 'disabled', \
#                         'List_Select_VL': [Parameter_Int_Number_Points],\
#                         'Status_PS_Row_V_Col_VL': 'disabled',\
#                         'List_Value_SF': [Parameter_Sampling_Frequency], \
#                         'Status_Select_SF': 'disabled', \
#                         'Hide_Validation':'', \
#                         'List_Option_HI':List_Option_HI, \
#                         'Status_Option_HI': '', 
#                         'List_Option_VI':List_Option_VI, \
#                         'Status_Option_VI': ''}
                
#                 Parameter_Int_Selection_Interval \
#                     = int( 1 / Parameter_Sampling_Frequency * 10)
#                 print(\
#                     Parameter_Total_Height, \
#                     Parameter_Int_Selection_Interval, \
#                     Parameter_Int_Number_Points)
#                 Function_DJ_Simulation(\
#                     Parameter_Total_Height, \
#                     Parameter_Int_Selection_Interval, \
#                     Parameter_Int_Number_Points)
#                 return render(request, self.Str_Template_Name, Dict_Args)             
#             except:
#                 print('The wrong input value')
#                 Str_PH_Input_HS = 'Wrong Value'
#                 Dict_Args \
#                     = {'Str_PH_Input_HS':Str_PH_Input_HS}
#                 return render(request, self.Str_Template_Name, Dict_Args)
        
#         if 'Name_Btn_Show' in Dict_POST_Parameters:
#             try:
#                 Dict_POST_Parameters['Name_Btn_Show'] == 'Click_Show'
#                 Value_Index = int(Dict_POST_Parameters['Name_Show_VI'])
#                 RawData = io.loadmat('Data_Simulation/Simu_X.mat')
#                 Array2D_Simulation = RawData['Array2D_Simulation']
#                 Array_Time = numpy.squeeze(RawData['Array_Time'])
#                 Array_Z_Coordinate = numpy.squeeze(RawData['Array_Z_Coordinate'])
#                 Temp_Fig_Array_0 \
#                     = Array_Time
#                 Temp_Fig_Array_1 \
#                     = Array2D_Simulation[:,Value_Index]
#                 pyplot.figure(figsize = (4,3), dpi = 100)
#                 pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array_1, \
#                             linewidth = 0.9, alpha = 0.99)
#                 pyplot.xlabel('Time (s)', fontsize = 5)
#                 pyplot.ylabel('Wind speed (m/s)', fontsize = 5)
#                 pyplot.grid(True, alpha = 0.2)
#                 pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
#                 pyplot.tight_layout()
#                 pyplot.savefig('static/Figure_3_History.png', \
#                                 transparent = True)
#                 pyplot.close('all')
#                 print('Figure Saved')
#                 Parameter_Total_Height \
#                     = Array_Z_Coordinate.max()
#                 Parameter_Int_Number_Points \
#                     = Array_Z_Coordinate.size
#                 Parameter_Sampling_Frequency \
#                     = 1 / (Array_Time[1] - Array_Time[0])
#                 List_Option_HI = [1]
#                 List_Option_VI = [Value_Index]
#                 Dict_Args \
#                     = {'List_Valeu_HL':[1], \
#                         'Value_VS':Parameter_Total_Height, \
#                         'Status_PS_Row_V_Col_VS': 'disabled', \
#                         'List_Select_VL': [Parameter_Int_Number_Points],\
#                         'Status_PS_Row_V_Col_VL': 'disabled',\
#                         'List_Value_SF': [Parameter_Sampling_Frequency], \
#                         'Status_Select_SF': 'disabled', \
#                         'Hide_Validation':'', \
#                         'List_Option_HI':List_Option_HI, \
#                         'Status_Option_HI': 'disabled', \
#                         'List_Option_VI':List_Option_VI, \
#                         'Status_Option_VI': 'disabled', \
#                         'Status_Show_Simu_Sele': 'disabled'}                      
#                 return render(request, self.Str_Template_Name, Dict_Args)
#             except:
#                 print('Something went wrong')
#     # return HttpResponse( render(request, Template.template.name))

def Function_Download_Simulation(request):
    Str_File_Path = 'Data_Simulation/Simu_X.mat'
    if os.path.exists(Str_File_Path):
        with open(Str_File_Path, 'rb') as file_handler:
            response \
                = HttpResponse\
                    (file_handler.read(), \
                    content_type = 'application/force-download') 
            response['Content-Disposition'] \
                = 'inline; filename=' + os.path.basename(Str_File_Path)
            return response                 
    return Http404

def Function_Click_Button_Simu(request):

    return HttpResponse('Test')


# class Class_Test_D3(TemplateView):
#     Str_Template_Name = 'Page_D3_Test.html'
#     Template = loader.get_template(Str_Template_Name)
#     # print(request.scheme)
#     # print(request.method)
#     # print(request.GET)
#     # print(request.POST)

#     def get(self, request):
#         form = HomeForm()
#         # print('Get_Function_Print_Get', request.GET)
#         # print('Get_Function_Print_POST', request.POST)
#         # if request.GET.get('Btn_Simu'):
#         #     print('Btn_Clicked')
#         Value_VS = '100 m'
#         List_Valeu_HL = [1]
#         List_Select_VL = numpy.arange(2,20,1)
#         List_Value_SF = [0.1, 0.2, 0.5, 1, 2]
#         Dict_Args \
#             = {\
#                 'Value_VS':Value_VS, \
#                 'Status_PS_Row_V_Col_VS': '', \
#                 'List_Valeu_HL':List_Valeu_HL, \
#                 'List_Select_VL':List_Select_VL, \
#                 'Status_PS_Row_V_Col_VL': '',\
#                 'List_Value_SF':List_Value_SF, \
#                 'Status_Select_SF': '', \
#                 'Hide_Validation':'hidden'}
#         return render(request, self.Str_Template_Name, Dict_Args)
#         # return render(request, self.Str_Template_Name, {'form':form})

#     def post(self, request):
#         form = HomeForm(request.POST)
#         print('POST_Function_Print_POST', request.POST)
#         if form.is_valid():
#             text = form.cleaned_data
#         # text = '(m)'
#         Dict_POST_Parameters = request.POST
#         print(Dict_POST_Parameters)
 
#         if 'Name_Btn_Simu' in Dict_POST_Parameters:
#             try:
#                 Dict_POST_Parameters['Name_Btn_Simu'] == 'Click_Simu'
#                 # Name_Input_HS = int(Dict_POST_Parameters['Name_Input_HS'])
#                 Parameter_Total_Height \
#                     = float(Dict_POST_Parameters['Name_Input_VS'])
#                 Parameter_Int_Number_Points \
#                     = int(Dict_POST_Parameters['Name_Select_VL'])
#                 Parameter_Sampling_Frequency \
#                     = float(Dict_POST_Parameters['Name_Select_SF'])
#                 Dict_POST_Parameters['Name_Select_SF']   
#                 List_Option_HI = range(1 + 1)[1:]
#                 List_Option_VI = range(Parameter_Int_Number_Points + 1)[1:]
#                 Dict_Args \
#                     = {'List_Valeu_HL':[1], \
#                         'Value_VS':Parameter_Total_Height, \
#                         'Status_PS_Row_V_Col_VS': 'disabled', \
#                         'List_Select_VL': [Parameter_Int_Number_Points],\
#                         'Status_PS_Row_V_Col_VL': 'disabled',\
#                         'List_Value_SF': [Parameter_Sampling_Frequency], \
#                         'Status_Select_SF': 'disabled', \
#                         'Hide_Validation':'', \
#                         'List_Option_HI':List_Option_HI, \
#                         'Status_Option_HI': '', 
#                         'List_Option_VI':List_Option_VI, \
#                         'Status_Option_VI': ''}
                
#                 Parameter_Int_Selection_Interval \
#                     = int( 1 / Parameter_Sampling_Frequency * 10)
#                 print(\
#                     Parameter_Total_Height, \
#                     Parameter_Int_Selection_Interval, \
#                     Parameter_Int_Number_Points)
#                 Function_DJ_Simulation(\
#                     Parameter_Total_Height, \
#                     Parameter_Int_Selection_Interval, \
#                     Parameter_Int_Number_Points)
#                 return render(request, self.Str_Template_Name, Dict_Args)             
#             except:
#                 print('The wrong input value')
#                 Str_PH_Input_HS = 'Wrong Value'
#                 Dict_Args \
#                     = {'Str_PH_Input_HS':Str_PH_Input_HS}
#                 return render(request, self.Str_Template_Name, Dict_Args)
        
#         if 'Name_Btn_Show' in Dict_POST_Parameters:
#             try:
#                 Dict_POST_Parameters['Name_Btn_Show'] == 'Click_Show'
#                 Value_Index = int(Dict_POST_Parameters['Name_Show_VI'])
#                 RawData = io.loadmat('Data_Simulation/Simu_X.mat')
#                 Array2D_Simulation = RawData['Array2D_Simulation']
#                 Array_Time = numpy.squeeze(RawData['Array_Time'])
#                 Array_Z_Coordinate = numpy.squeeze(RawData['Array_Z_Coordinate'])
#                 Temp_Fig_Array_0 \
#                     = Array_Time
#                 Temp_Fig_Array_1 \
#                     = Array2D_Simulation[:,Value_Index]
#                 pyplot.figure(figsize = (4,3), dpi = 100)
#                 pyplot.plot(Temp_Fig_Array_0, Temp_Fig_Array_1, linewidth = 0.9, alpha = 0.99)
#                 pyplot.xlabel('Time (s)', fontsize = 5)
#                 pyplot.ylabel('Wind speed (m/s)', fontsize = 5)
#                 pyplot.grid(True, alpha = 0.2)
#                 pyplot.xlim(Temp_Fig_Array_0.min(), Temp_Fig_Array_0.max())
#                 pyplot.tight_layout()
#                 pyplot.savefig('static/Figure_3_History.png', transparent = True)
#                 pyplot.close('all')
#                 print('Figure Saved')
#                 Parameter_Total_Height \
#                     = Array_Z_Coordinate.max()
#                 Parameter_Int_Number_Points \
#                     = Array_Z_Coordinate.size
#                 Parameter_Sampling_Frequency \
#                     = 1 / (Array_Time[1] - Array_Time[0])
#                 List_Option_HI = [1]
#                 List_Option_VI = [Value_Index]
#                 Dict_Args \
#                     = {'List_Valeu_HL':[1], \
#                         'Value_VS':Parameter_Total_Height, \
#                         'Status_PS_Row_V_Col_VS': 'disabled', \
#                         'List_Select_VL': [Parameter_Int_Number_Points],\
#                         'Status_PS_Row_V_Col_VL': 'disabled',\
#                         'List_Value_SF': [Parameter_Sampling_Frequency], \
#                         'Status_Select_SF': 'disabled', \
#                         'Hide_Validation':'', \
#                         'List_Option_HI':List_Option_HI, \
#                         'Status_Option_HI': 'disabled', \
#                         'List_Option_VI':List_Option_VI, \
#                         'Status_Option_VI': 'disabled', \
#                         'Status_Show_Simu_Sele': 'disabled'}                      
#                 return render(request, self.Str_Template_Name, Dict_Args)
#             except:
#                 print('Something went wrong')
#     # return HttpResponse( render(request, Template.template.name))


    