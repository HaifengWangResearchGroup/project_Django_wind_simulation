var Valeu_Max_Locations = document.getElementById('ID_Para_Max_Locations').value;
var Value_Max_VL = Valeu_Max_Locations;
var Value_Max_HL = Valeu_Max_Locations;


var Int_VL = document.getElementById('ID_Select_VL').value;
var Int_HL = document.getElementById('ID_Select_HL').value;
var Value_HS = document.getElementById("ID_Input_HS").value;
var Value_VS = document.getElementById("ID_Input_VS").value;

var Value_Delta_H = Value_HS / Int_HL;
var Value_Delta_V = Value_VS / Int_VL;

function Function_Generate_Location(Int_HL, Value_HS, Int_VL, Value_VS) {
    Value_Delta_H = Value_HS / Int_HL;
    Value_Delta_V = Value_VS / Int_VL;
    Array2D_Data = [];
    for (i_Index_H = 0; i_Index_H < Int_HL; i_Index_H++){
        for (i_Index_V = 1; i_Index_V <= Int_VL; i_Index_V++) {
            Array2D_Data.push([i_Index_H * Value_Delta_H, 
                                i_Index_V * Value_Delta_V])
        }
    }

    return Array2D_Data
};

var Array2D_Data = [];
Array2D_Data 
    = Function_Generate_Location(Int_HL, Value_HS, Int_VL, Value_VS);

var Array1D_Data_X_Axis
var Array1D_Data_Y_Axis
var Value_Max_x
var Value_Min_x
var Value_Diff_x
var Value_max_xaxis
var Value_min_xaxis
var Value_Max_y
var Value_Min_y
var Value_Diff_y
var Value_max_yaxis
var Value_min_yaxis

function Function_Update_axis_info() {
    Int_HL = document.getElementById("ID_Select_HL").value;
    Value_HS = document.getElementById("ID_Input_HS").value;
    Int_VL = document.getElementById("ID_Select_VL").value;
    Value_VS = document.getElementById("ID_Input_VS").value;
    Array2D_Data 
        = Function_Generate_Location(Int_HL, Value_HS, Int_VL, Value_VS);
    console.log('update dataew')
    Array1D_Data_X_Axis = Array2D_Data.map(function(Temp_Array) {return Temp_Array[0];});
    Array1D_Data_Y_Axis = Array2D_Data.map(function(Temp_Array) {return Temp_Array[1];});
    console.log(Array1D_Data_X_Axis)
    Value_Max_x = Math.max(...Array1D_Data_X_Axis);
    Value_Min_x = Math.min(...Array1D_Data_X_Axis);
    Value_Diff_x = Math.max(Value_Max_x - Value_Min_x, 10);
    Value_max_xaxis = Math.ceil(Value_Max_x + Value_Diff_x * 0.1);
    Value_min_xaxis = Math.floor(Value_Min_x - Value_Diff_x * 0.1);
    Value_Max_y = Math.max(...Array1D_Data_Y_Axis);
    Value_Min_y = Math.min(...Array1D_Data_Y_Axis);
    Value_Diff_y = Math.max(Value_Max_y - Value_Min_y, 10);
    Value_max_yaxis = Math.ceil(Value_Max_y + Value_Diff_y * 0.1);
    Value_min_yaxis = Math.floor(Value_Min_y - Value_Diff_y * 0.1);
}

Function_Update_axis_info()

var myChart = echarts.init(document.getElementById('JQ_Test_P_1')); 

var option = {
    title: {
        show:false,
    },
    legend: {
        show:false,
    },
    xAxis: {
        name: 'Horizontal (m)',
        nameLocation: 'middle',
        nameTextStyle: {verticalAlign: 'bottom'}, 
        nameGap : '20',
        gridIndex: 0,
        max: Value_max_xaxis,
        min: Value_min_xaxis,
        axisTick: false,
    },
    yAxis: {
        name: 'Height (m)',
        nameRotate: 90,
        nameLocation: 'middle',
        nameGap : '40',
        position: 'left',
        gridIndex: 0,
        max: Value_max_yaxis,
        min: Value_min_yaxis,
        axisTick: false,
    },
    grid: {
        left: 70
    },
    series: [{
        name: '',
        symbolSize: 15,
        type: 'scatter',
        data: Array2D_Data,
    }],
};

myChart.setOption(option); 

function update_fig() {
    Function_Update_axis_info()
    option.xAxis.max = Value_max_xaxis;
    option.xAxis.min = Value_min_xaxis;
    option.yAxis.max = Value_max_yaxis;
    option.yAxis.min = Value_min_yaxis;
    option.series[0].data = Array2D_Data;
    // console.log(option.series.data);
    myChart.clear();
    myChart.setOption(option);
}


function update_for_hor_ver_parameter_change() {
    Temp_SHL = document.getElementById('ID_Select_HL');
    Int_HL = Temp_SHL.options[Temp_SHL.selectedIndex].value;
    Temp_SVL = document.getElementById('ID_Select_VL');
    Int_VL = Temp_SVL.options[Temp_SVL.selectedIndex].value;

    Value_Max_VL = Valeu_Max_Locations / Int_HL;
    Value_Max_HL = Valeu_Max_Locations / Int_VL;

    var i_S = 0;
    for (i_S = 0; i_S < Temp_SVL.options.length; i_S++) {
        if (Temp_SVL.options[i_S].value > Value_Max_VL) {
            Temp_SVL.options[i_S].disabled = true;
            Temp_SVL.options[i_S].hidden = true;
        }
        else {
            Temp_SVL.options[i_S].disabled = false;
            Temp_SVL.options[i_S].hidden = false;
        }
    }
    for (i_S = 0; i_S < Temp_SHL.options.length; i_S++) {
        if (Temp_SHL.options[i_S].value > Value_Max_HL) {
            Temp_SHL.options[i_S].disabled = true;
            Temp_SHL.options[i_S].hidden = true;
        }
        else {
            Temp_SHL.options[i_S].disabled = false;
            Temp_SHL.options[i_S].hidden = false;
        }
    }
    update_fig();

}


// Disable "enter" for simulation to prevent unexpected simulation
$(document).keypress(
    function(event){
      if (event.which == '13') {
        event.preventDefault();
      }
});



