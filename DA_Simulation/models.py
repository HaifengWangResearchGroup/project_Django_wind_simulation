import datetime

from django.db import models
from django.utils import timezone

# Create your models here.

class Class_Data(models.Model):
    Data_Name \
        = models.CharField(max_length = 200, \
                            default = 'Default_Data_Name', \
                            editable = True)
    Data_Date \
        = models.DateTimeField('Date_Uploaded', \
                            default = '0000000000', \
                            editable = True)
    Data_Col_1 \
        = models.CharField(max_length = 300, \
                            default = 'Default_Data_Name', \
                            editable = True)
    Data_Col_2 \
        = models.CharField(max_length = 300, \
                            default = 'Default_Data_Name', \
                            editable = True)
    
    def Function_Duration_Since_Upload(self):
        Value_Duration_Since_Upload \
            = timezone.now() - self.Data_Date
        return Value_Duration_Since_Upload

    def Function_Set_Data_Col_1(self, Str_Date):
        Value_Duration_Since_Upload \
            = timezone.now() - self.Data_Date
        return Value_Duration_Since_Upload

class Class_Spectrum_Model(models.Model):
    Data_Name = models.CharField(max_length = 200)
    Data_Date = models.DateTimeField('Date_Operation')

