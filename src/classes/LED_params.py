import numpy as np


class LEDparams:
    """
    Class that connects the Program to a LED consumption
    """
    def __init__(self) -> None:
        self.I_LED = 0.017 # A
        
        # Identified at 0.017 A at 25°C
        radiant_power_dict = {
            'IR':       5.94,
            'Red':      1.70,
            'Orange':   1.18,
            'Green':    4.60,
            'Blue':     6.56,
            'Purple':   8.21,
            'UV':       3.92
        } # mW/cm^2

        # Convert to numpy array
        specific_radiant_power = np.array([radiant_power_dict[color] for color in radiant_power_dict.keys()]) # mW/cm^2

        # Radiant power in W for single hole
        diameter = 6.7 # mm
        cross_section = np.pi * (diameter/10/2)**2 # cm^2
        self.radiant_power = cross_section * specific_radiant_power / 1000 # W

        # Dimensions for checking
        num_colors = len(specific_radiant_power)
        plate_row = 8
        plate_col = 12
        self.dimensions = (num_colors, plate_row, plate_col)

        # Radiant power with right dimensions
        self.radiant_power_3D = self.radiant_power[:, np.newaxis, np.newaxis]

        # Min and max values
        self.P_rad_min = 0 # W
        self.P_rad_max = np.sum(np.ones(self.dimensions) * self.radiant_power_3D) # W

    def program_reader(self, x_matrix_scaled:np.ndarray) -> np.ndarray:
        """
        Returns the total duty cycle, single LED constant current and total radiant power
        """
        self.P_rad = np.sum(x_matrix_scaled * self.radiant_power_3D) # W

        self.x_LED_tot = np.sum(x_matrix_scaled) # total duty cycle assuming equal current for all LEDs

        return self.x_LED_tot, self.I_LED, self.P_rad
    
    def get_x_from_P_rad(self, P_rad:float) -> np.ndarray:
        """
        Returns the duty cycle for the given radiant power
        """

        assert P_rad >= self.P_rad_min, f"Radiant power must be greater than {self.P_rad_min}"
        assert P_rad <= self.P_rad_max, f"Radiant power must be less than {self.P_rad_max}"

        self.P_rad = P_rad
        self.x_LED_tot = self.dimensions[0] * P_rad / sum(self.radiant_power) # total duty cycle assuming equal current and x for all LEDs

        return self.x_LED_tot, self.I_LED, self.P_rad

##########################################################################
if __name__ == '__main__':
    U_BT = 4.0 # V
    LEDpar = LEDparams()

    x_LED, I_LED, P_rad = LEDpar.program_reader()

    I_LED_tot = I_LED * x_LED # A
    P_LED = U_BT * I_LED_tot # W
    Q_LED_tot = P_LED - P_rad # W
    
    print(f"Total duty cycle: {x_LED:.2f}")
    print(f"Total LED power: {P_LED:.2f} W")
    print(f"Total radiant power: {P_rad:.2f} W")
    print(f"Total LED power loss: {Q_LED_tot:.2f} W")