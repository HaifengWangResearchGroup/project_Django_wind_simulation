# Generated by Django 2.1.7 on 2019-03-08 16:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DA_Simulation', '0002_auto_20190308_1014'),
    ]

    operations = [
        migrations.AlterField(
            model_name='class_data',
            name='Data_Col_1',
            field=models.CharField(default='Default_Data_Name', max_length=300),
        ),
        migrations.AlterField(
            model_name='class_data',
            name='Data_Col_2',
            field=models.CharField(default='Default_Data_Name', max_length=300),
        ),
        migrations.AlterField(
            model_name='class_data',
            name='Data_Date',
            field=models.DateTimeField(default='0000000000', verbose_name='Date_Uploaded'),
        ),
        migrations.AlterField(
            model_name='class_data',
            name='Data_Name',
            field=models.CharField(default='Default_Data_Name', max_length=200),
        ),
    ]