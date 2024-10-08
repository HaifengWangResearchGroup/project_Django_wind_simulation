"""
Creation data:
    Unknown

Update record:
    20190322 Update the function list in description
    20190325 Add Coherence from Hansen-1999-DAWR-RN3375
    20190326 Add 2D location generation function
    2019_03_28:
        Return of Function_HVD_HC_Coherence_Hanse_1999 (For further convenience)
            From: Array3D_Coherence
            To: Array3D_Coherence, [Parameter_Cy, Parameter_Cz]

"""

import numpy

class Class_Wind_Field_Information():
    """
    Description:
        A class defined for obtaining the wind field information:
            Vertical prifile
            Wind speed
            Spectrum
            etc.
    Funciton list:
        - Functino_2D_Location_Generation
        - Function_Vertical_Profile_Hurricane_Vickery 
        - Function_Vertical_Profile_Conventional_Boundary_Layer_with_Category
        - Function_Vertical_Profile_Downburst
        - Function_Profile_Downburst_VRH
        - Function_Vertical_Profile_Hurricane_Reda
        - Function_Correlation_Generation_Jiang_2017
    """
    def __init__(self):
        self.Class_Name = 'Class of wind field information'
        self.Bool_Flag_TV_Corr_MPL = False

    def Functino_2D_Location_Generation(self, \
            Int_Number_Locations_X, Value_X_Min, Value_X_Max, \
            Int_Number_Locations_Y, Value_Y_Min, Value_Y_Max, \
            Int_Number_Locations_Z, Value_Z_Min, Value_Z_Max):
        """
        ------------------------------------------------------------------------
        Output:
            Array2D_Locations:
                [i_Loc, i_Direction]
                [i_Loc, :] = [X_Coordinate, Y_Coordinate, Z_Coordinate] 
        """
        Int_Number_Locations \
            = Int_Number_Locations_X \
                * Int_Number_Locations_Y \
                * Int_Number_Locations_Z
        Array_X_Coordinate \
            = numpy.linspace(Value_X_Min, Value_X_Max, Int_Number_Locations_X)
        Array_Y_Coordinate \
            = numpy.linspace(Value_Y_Min, Value_Y_Max, Int_Number_Locations_Y)
        Array_Z_Coordinate \
            = numpy.linspace(Value_Z_Min, Value_Z_Max, Int_Number_Locations_Z)
        Array2D_Locations = numpy.zeros([Int_Number_Locations, 3])
        for i_Loc_X in range(Int_Number_Locations_X):
            for i_Loc_Y in range(Int_Number_Locations_Y):
                for i_Loc_Z in range(Int_Number_Locations_Z):
                    i_Loc_Part_1 \
                        = i_Loc_X \
                            * Int_Number_Locations_Y \
                            * Int_Number_Locations_Z
                    i_Loc_Part_2\
                        = i_Loc_Y * Int_Number_Locations_Z
                    i_Loc_Part_3 \
                        = i_Loc_Z
                    i_Loc = i_Loc_Part_1 + i_Loc_Part_2 + i_Loc_Part_3
                    Array2D_Locations[i_Loc,0] = Array_X_Coordinate[i_Loc_X]
                    Array2D_Locations[i_Loc,1] = Array_Y_Coordinate[i_Loc_Y]
                    Array2D_Locations[i_Loc,2] = Array_Z_Coordinate[i_Loc_Z]
        return Array2D_Locations

    def Function_Vertical_Profile_Hurricane_Vickery(self, \
            Array_Height, Location):
        """
        Description:
            Reference:
                Vickery-2009-AHBL
        Input:
            Array_Height
            Location: ()
        """
        ## Parameter Setting
        Parameter_k = 0.4 # von Karman coefficient
        if Location == 1:
            Parameter_u_star = 1.440137 # Friction Velocity
            Parameter_Z_Zero = 0.26655711 # 0.03
            Parameter_a = 0.167588219
            # Boundayr layer height parameter, Vickery-2009-AHBL Page 388
            Parameter_H_star = 300
            Parameter_n = 2.034519
        elif Location == 2:
            Parameter_u_star = 1.27882012 # Friction Velocity
            Parameter_Z_Zero = 0.43730792 # 0.03
            Parameter_a = 0.157714363
            # Boundayr layer height parameter, Vickery-2009-AHBL Page 388
            Parameter_H_star = 300
            Parameter_n = 1.82154745
        else:
            Parameter_u_star = 0.986553159 # Friction Velocity
            Parameter_Z_Zero = 0.258025841 # 0.03
            Parameter_a = -0.0269064102
            # Boundayr layer height parameter, Vickery-2009-AHBL Page 388
            Parameter_H_star = 300
            Parameter_n = 2.01358223
        ## Wind speed sub_function
        def Sub_Function_Uz(Value_Height):
            Value_Mewan_Wind_Speed \
                = Parameter_u_star / Parameter_k \
                    * (numpy.log(Value_Height / Parameter_Z_Zero) \
                        - Parameter_a \
                            * (Value_Height / Parameter_H_star)**Parameter_n)
            return Value_Mewan_Wind_Speed
        ## Array Calculation
        Array_Wind_Profile_Azimuthal = numpy.zeros(Array_Height.shape)
        for i_Height in range(Array_Height.size):
            Array_Wind_Profile_Azimuthal[i_Height] \
                = Sub_Function_Uz(Array_Height[i_Height])
        return Array_Wind_Profile_Azimuthal

    def Function_Vertical_Profile_Conventional_Boundary_Layer_with_Category(\
            self, Array_Height, Str_Category):
        """
        Input:
            - Array_Height
            - Str_Category:
                - 'B'
                - 'C'
                - 'D'
        """
        if Str_Category.lower() == 'b':
            Parameter_Max_Velocity = 24
            Parameter_Boundary_Height = 365.76
        elif Str_Category.lower() == 'c':
            Parameter_Max_Velocity = 24
            Parameter_Boundary_Height = 274.32
        elif Str_Category.lower() == 'd':
            Parameter_Max_Velocity = 24
            Parameter_Boundary_Height = 213.36
        Array_Wind_Profile_Boundary_Layer = numpy.zeros(Array_Height.shape)
        for i_Height in range(Array_Height.size):
            if Array_Height[i_Height] <= Parameter_Boundary_Height:
                Array_Wind_Profile_Boundary_Layer[i_Height] \
                    = Parameter_Max_Velocity \
                        / Parameter_Boundary_Height**(1/9.5) \
                        * Array_Height[i_Height]**(1/9.5)
            else:
                Array_Wind_Profile_Boundary_Layer[i_Height] \
                    = Parameter_Max_Velocity
        return Array_Wind_Profile_Boundary_Layer

    def Function_Vertical_Profile_Downburst(self, \
            Array_Height):
        """
        Description:
            Generate vertical profile based on the model proposed by:
                Vicroy-1991-ASSA
            Used by:
                Kwon-2009-GFFN
        ------------------------------------------------------------------------
        Input:
            - Array_Height
        ------------------------------------------------------------------------
        Output:
            - Array_Wind_Profile_Downburst
        """
        # Initial Variables
        Array_Wind_Profile_Downburst = numpy.zeros(Array_Height.shape)
        # Define all Parameters
        Parameter_A = 1.354
        Parameter_c1 = -0.22
        Parameter_c2 = -2.75
        Parameter_Z_Max = 80
        # Parameter Generation
        Parameter_V_Max = 25.624007
        # Profile Generation
        for i_Height in range(Array_Height.size):
            Temp_Calculation_Downburst_Part_1 \
                = numpy.exp(Parameter_c1 * Array_Height[i_Height] \
                                / Parameter_Z_Max) \
                    - numpy.exp(Parameter_c2 * Array_Height[i_Height] \
                                / Parameter_Z_Max)
            Array_Wind_Profile_Downburst[i_Height] \
                = Parameter_A * Parameter_V_Max \
                    * Temp_Calculation_Downburst_Part_1
        return Array_Wind_Profile_Downburst

    def Function_Profile_Downburst_VRH(self, \
            Array_Height, Array_Radius, \
            Parameter_Str_Model_Name, Dict_Parameters_Input):
        """
        Description:
            Generate vertical and radial profiles of Horizontal winds
            based on the specified models
        ------------------------------------------------------------------------
        Input:
            - Array_Height
            - Parameter_Str_Model_Name
                1. 'Vicroy-1991-ASSA'
            - Dict_Parameters_Input
        ------------------------------------------------------------------------
        Output:
            - Array2D_Horizontal_Wind_Speed
        """
        if Parameter_Str_Model_Name == 'Vicroy-1991-ASSA':
            """
            Model proposed by:
                Vicroy-1991-ASSA
            Used by:
                Kwon-2009-GFFN
            """
            # Initial Variables
            Array2D_Horizontal_Wind_Speed \
                    = numpy.zeros([Array_Height.size, Array_Radius.size])
            # Define all Parameters
            Dict_Parameters = {}
            Dict_Parameters['alpha'] = 2 # Radius shaping function variable
                # The value of 2 is suggested in Vicry-1991-ASAA
            Dict_Parameters['c1'] = -0.22
            Dict_Parameters['c2'] = -2.75
            Dict_Parameters['z_m'] = 80 # Height of maximum wind (m)
            Dict_Parameters['r_p'] = 1000 # Rasius of peak horizontal wind (m)
            Dict_Parameters['u_m'] = 25.62 # m/s
            # Update from input
            Dict_Parameters.update(Dict_Parameters_Input)
            # Calculate corresponding parameters
            
            Dict_Parameters['lambda'] \
                = Dict_Parameters['u_m'] * 2 / Dict_Parameters['r_p']\
                    / (numpy.exp(Dict_Parameters['c1']) \
                        - numpy.exp(Dict_Parameters['c2'])) \
                    / numpy.exp(1 / 2 / Dict_Parameters['alpha']) 
                # Scaling factor (1/s)
            # Sub_Function_Define
            def Sub_Function_Radius_Shape(Value_r):
                Temp_Value_Part_1 \
                    = (Value_r / Dict_Parameters['r_p'])\
                        **(2 * Dict_Parameters['alpha'])
                Value_Radius_Shape \
                    = Value_r / 2 \
                        * numpy.exp((2 - Temp_Value_Part_1)\
                                        / 2 / Dict_Parameters['alpha'])
                return Value_Radius_Shape

            def Sub_Function_Vertical_Shape(Value_z):
                Temp_Value_Part_1 \
                    = numpy.exp(Dict_Parameters['c1'] * Value_z \
                        / Dict_Parameters['z_m'])
                Temp_Value_Part_2 \
                    = numpy.exp(Dict_Parameters['c2'] * Value_z \
                        / Dict_Parameters['z_m'])
                Value_Vertical_Shape \
                    = Temp_Value_Part_1 - Temp_Value_Part_2
                return Value_Vertical_Shape

            # Profile Generation
            for i_Radius in range(Array_Radius.size):
                Value_r = Array_Radius[i_Radius]
                Value_Radius_Shape = Sub_Function_Radius_Shape(Value_r)
                for i_Height in range(Array_Height.size):
                    Value_z = Array_Height[i_Height]
                    Value_Vertical_Shape = Sub_Function_Vertical_Shape(Value_z)
                    Array2D_Horizontal_Wind_Speed[i_Height, i_Radius] \
                        = Dict_Parameters['lambda'] \
                            * Value_Radius_Shape \
                            * Value_Vertical_Shape
        return Array2D_Horizontal_Wind_Speed

    def Function_Vertical_Profile_Hurricane_Reda(self, \
            Array_Height, Location):
        if Location == 1:
            # [[  2.39975494e+01   1.96770447e-01   2.23008107e+02
            #   1.00000000e+02]]
            Parameter_v_g = 23.9975494
            Parameter_H_star = 223.008107
            Parameter_a_1 = 0.196770447
        elif Location == 2:
            # [[  1.97882608e+01   1.43105858e-01   3.29795375e+02
            #   1.00000000e+02]]
            Parameter_v_g =  19.7882608
            Parameter_H_star = 329.795375
            Parameter_a_1 = 0.143105858
        else:
            # [[  2.04571315e+01   2.99357329e-02   9.54804670e+02
            #   1.00000000e+02]]
            # [[  2.04571315e+01   2.99357329e-02   9.54804670e+02
            #   1.00000000e+02]]
            Parameter_v_g = 20.4571315
            Parameter_H_star = 954.80467
            Parameter_a_1 = 0.0299357329
        Parameter_v_rg = -4.6972
        Parameter_B = -1.6223
        Parameter_k_m = 100

        Temp_Calculation_Azimuthal_Part_2 \
            = -(1 + Parameter_a_1) * numpy.cos(200 / Parameter_H_star) \
                + numpy.sin(200 / Parameter_H_star)
        Temp_Calculation_Azimuthal_Part_1 \
            = 1 + Parameter_a_1 * Parameter_H_star / Parameter_k_m \
                    * numpy.exp(-200 / Parameter_H_star) \
                    * Temp_Calculation_Azimuthal_Part_2
        Value_V_200 = Parameter_v_g * Temp_Calculation_Azimuthal_Part_1
        Temp_Calculation_Azimuthal_Part_3 \
            = -1 / Parameter_H_star * (-(1 + Parameter_a_1) \
                    * numpy.cos(200 / Parameter_H_star) \
                + numpy.sin(200 / Parameter_H_star))
        Temp_Calculation_Azimuthal_Part_2 \
            = 1 / Parameter_H_star *((1 + Parameter_a_1) \
                    * numpy.sin(200 / Parameter_H_star) \
                + numpy.cos(200 / Parameter_H_star))
        Temp_Calculation_Azimuthal_Part_1 \
            = Parameter_v_g * Parameter_a_1 \
                    * Parameter_H_star / Parameter_k_m \
                    * numpy.exp(-200 / Parameter_H_star)
        Value_V_differentiation_200 \
            = Temp_Calculation_Azimuthal_Part_1 \
                    * (Temp_Calculation_Azimuthal_Part_2 \
                        + Temp_Calculation_Azimuthal_Part_3)

        Parameter_u_star = 80 * Value_V_differentiation_200
        Parameter_z_zero \
            = numpy.exp(numpy.log(200) - Value_V_200 * 0.4 / Parameter_u_star)

        Array_Wind_Profile_Azimuthal = numpy.zeros(Array_Height.shape)
        Array_Wind_Profile_Radial = numpy.zeros(Array_Height.shape)
        for i_Height in range(Array_Height.size):
            if Array_Height[i_Height] >= 200:
                Temp_Calculation_Azimuthal_Part_2 \
                    = -(1 + Parameter_a_1) * numpy.cos(Array_Height[i_Height] \
                            / Parameter_H_star) \
                        + numpy.sin(Array_Height[i_Height] / Parameter_H_star)
                Temp_Calculation_Azimuthal_Part_1 \
                    = 1 + Parameter_a_1 * Parameter_H_star \
                            / Parameter_k_m \
                            * numpy.exp(-Array_Height[i_Height] \
                                            / Parameter_H_star) \
                            * Temp_Calculation_Azimuthal_Part_2
                Array_Wind_Profile_Azimuthal[i_Height] \
                    = Parameter_v_g * Temp_Calculation_Azimuthal_Part_1
            else:
                Array_Wind_Profile_Azimuthal[i_Height] \
                    = Parameter_u_star \
                        / 0.4 * numpy.log(Array_Height[i_Height] \
                        / Parameter_z_zero)

            Temp_Calculation_Radial_Part_2 \
                = numpy.cos(Array_Height[i_Height] / Parameter_H_star) \
                    + (1 + Parameter_a_1) \
                        * numpy.sin(Array_Height[i_Height] / Parameter_H_star)
            Temp_Calculation_Radial_Part_1 \
                = Parameter_v_g * Parameter_B * Parameter_a_1 \
                    * Parameter_H_star \
                    / Parameter_k_m \
                    * numpy.exp(-Array_Height[i_Height] / Parameter_H_star) \
                    * Temp_Calculation_Radial_Part_2
            Array_Wind_Profile_Radial[i_Height] \
                = Parameter_v_rg + Temp_Calculation_Radial_Part_1
        return Array_Wind_Profile_Azimuthal, Array_Wind_Profile_Radial

    def Function_HVD_HC_Coherence_Hanse_1999(self, \
            Array2D_Locations, Array_Center_Frequency, \
            Value_U, Value_L):
        """
        Description:
            Consideration of horizontal distance (HD) and vertical distance (VD)
            Horizontal component (HC) generation:
        Reference:
            Proposed in: 
                - Hansen-1999-DAWR-RN3375
            Used in:
                - Peng-2018-FMAI-RN1601
        Input:
            Array2D_Locations: [i_Horizontal, i_Vertical]
            Array_Center_Frequency
            Value_U
            Value_L
        """ 
        Parameter_Cy = 5 # Horizontal decay compoentn RN3375-P10
        Parameter_Cz = 5 # Vertical decay compoentn RN3375-P10
        Int_Number_Locations = Array2D_Locations.shape[0]

        def Sub_Function_Modified_Frequency(Value_n):
            Value_Modified_n \
                = numpy.sqrt(Value_n**2 \
                            + (Value_U / 2 / numpy.pi / Value_L)**2)
            return Value_Modified_n

        def Sbu_Function_Coherence_Value(Value_Modified_n, Value_ry, Value_rz):
            Temp_Value_Part_nUSqrt \
                = Value_Modified_n \
                    / Value_U \
                    * numpy.sqrt((Parameter_Cy * Value_ry)**2 \
                            + (Parameter_Cz * Value_rz)**2)
            Temp_Value_Part_1 \
                = 1 - 1 / 2 *  Temp_Value_Part_nUSqrt
            Temp_Value_Part_2 \
                = numpy.exp(- Temp_Value_Part_nUSqrt)
            Value_Coherence = Temp_Value_Part_1 * Temp_Value_Part_2
            return Value_Coherence

        Array3D_Coherence \
            = numpy.zeros([Int_Number_Locations, \
                            Int_Number_Locations, \
                            Array_Center_Frequency.size])
                            
        for i_n in range(Array_Center_Frequency.size):
            Temp_Value_n \
                = Array_Center_Frequency[i_n]
            Temp_Value_Modified_n \
                = Sub_Function_Modified_Frequency(Temp_Value_n)
            for i_Row in range(Int_Number_Locations):
                for i_Col in range(Int_Number_Locations):
                    Temp_Value_ry \
                        = numpy.abs(Array2D_Locations[i_Row, 1] \
                                    - Array2D_Locations[i_Col, 1])
                    Temp_Value_rz \
                        = numpy.abs(Array2D_Locations[i_Row, 2] \
                                    - Array2D_Locations[i_Col, 2]) 
                    Temp_Value_Coherence \
                        = Sbu_Function_Coherence_Value(\
                            Temp_Value_Modified_n, \
                            Temp_Value_ry, \
                            Temp_Value_rz)
                    Array3D_Coherence[i_Row, i_Col, i_n] = Temp_Value_Coherence
        return Array3D_Coherence, [Parameter_Cy, Parameter_Cz]

    def Function_Correlation_Generation_Jiang_2017(self, \
            Array2D_Locations, Array_Center_Frequency, \
            Array_Time, Array_Signal_Time_Varying_Standard_Deviation, \
            Value_Mean_Standard_Deviation):
        """
        Multi_Scale_Correlation_Coefficient_Generation
            Based on Jiang 2017; Peng 2018
            Applicable for vertically distributed locations
        ------------------------------------------------------------------------
        Input:
            Array2D_Locations:
                [i_Loc, :] = [X_Coordinate, Y_Coordinate, Z_Coordinate] 
                Unit of height is m
            Array_Center_Frequency:
                Unit of Frequency is Hz
            Array_Time
            Array_Signal_Time_Varying_Standard_Deviation
            Value_Mean_Standard_Deviation
        ------------------------------------------------------------------------
        Output:
            Array_4D_TV_Correlation
            Array_4D_CT_Correlation
            Array_3D_Correlation_Base
        """
        # Check the input Array2D_Locations
        Value_X_Span \
            = Array2D_Locations[:,0].max() - Array2D_Locations[:,0].min()
        Value_Y_Span \
            = Array2D_Locations[:,1].max() - Array2D_Locations[:,1].min()
        if Value_X_Span != 0 or Value_Y_Span != 0:
            print('Error: Function_Correlation_Generation_Jiang_2017')
            print('\t Input Array2D_Locations is not vertically distributed')
            return None
        Array_Z_Coordinate = Array2D_Locations[:,2]
        # Parameter setting for the genearation of davenport-like
        # evolutionary coherence
        Parameter_k1 = 2.212
        Parameter_k2 = 0.125
        Parameter_k3 = 0.263
        # Variable_Rename
        Array_TVSD = Array_Signal_Time_Varying_Standard_Deviation
        # Sub_Function define
        def Sub_Function_Base_Correlation_Value(\
                Value_Delta_Coordinate,\
                Value_Frequency):
            Value_K = Parameter_k1 * \
                numpy.exp(- Parameter_k2 *
                        Value_Delta_Coordinate) + Parameter_k3
            Value_Temp_Correlation_Base = numpy.exp(
                - Value_K * Value_Frequency\
                    * Value_Delta_Coordinate)
            return Value_Temp_Correlation_Base
        # Calculation
        Array_3D_Correlation_Base \
            = numpy.zeros((Array_Z_Coordinate.size, \
                            Array_Z_Coordinate.size, \
                            Array_Center_Frequency.size))
        Array_4D_TV_Correlation \
            = numpy.zeros((Array_Z_Coordinate.size, \
                            Array_Z_Coordinate.size, \
                            Array_Center_Frequency.size, \
                            Array_Time.size))
        Array_4D_CT_Correlation \
            = numpy.zeros((Array_Z_Coordinate.size, \
                            Array_Z_Coordinate.size, \
                            Array_Center_Frequency.size, \
                            Array_Time.size))
        for i_Center_Frequency in range(Array_Center_Frequency.size):
            for i_Correlation_Row in range(Array_Z_Coordinate.size):
                for i_Correlation_Col in range(Array_Z_Coordinate.size):
                    if i_Correlation_Row > i_Correlation_Col:
                        Array_3D_Correlation_Base[\
                            i_Correlation_Row, i_Correlation_Col, \
                            i_Center_Frequency] \
                                = Array_3D_Correlation_Base[\
                                    i_Correlation_Col, \
                                    i_Correlation_Row, \
                                    i_Center_Frequency]
                    else:
                        Value_Delta_Coordinate \
                            = numpy.abs(\
                                Array_Z_Coordinate[i_Correlation_Row] \
                                - Array_Z_Coordinate[i_Correlation_Col])
                        Value_Frequency \
                            = Array_Center_Frequency[i_Center_Frequency]
                        Value_Temp_Correlation_Base \
                            = Sub_Function_Base_Correlation_Value(\
                                Value_Delta_Coordinate, Value_Frequency)
                        Array_3D_Correlation_Base[\
                            i_Correlation_Row, i_Correlation_Col,
                            i_Center_Frequency] \
                                = Value_Temp_Correlation_Base
        for i_Time in range(Array_Time.size):
            Array_4D_TV_Correlation[:, :, :, i_Time] \
                = Array_3D_Correlation_Base**(1 / Array_TVSD[i_Time])
            Array_4D_CT_Correlation[:, :, :, i_Time] \
                = Array_3D_Correlation_Base**(1 / Value_Mean_Standard_Deviation)
        return Array_4D_TV_Correlation, \
                Array_4D_CT_Correlation, \
                Array_3D_Correlation_Base
