import numpy as np


class LEDparams:
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

        # Single hole
        diameter = 6.7 # mm
        cross_section = np.pi * (diameter/10/2)**2 # cm^2
        self.radiant_power = cross_section * specific_radiant_power / 1000 # W

        # Dimensions for checking
        num_colors = len(specific_radiant_power)
        plate_row = 8
        plate_col = 12
        self.dimensions = (num_colors, plate_row, plate_col)

    def program_reader(self) -> np.ndarray:
        """ Returns the total duty cycle, single LED constant current and total radiant power"""
        x_matrix = np.random.randint(0, 2**16-1, self.dimensions) # TODO implement program reader

        # Min
        # x_matrix = np.zeros(self.dimensions)

        # Max
        # x_matrix = np.ones(self.dimensions) * (2**16-1)

        x_matrix = x_matrix / 5 # TODO decrease losses

        x_matrix_scaled = x_matrix / (2**16-1) # duty cycle

        self.P_r = np.sum(x_matrix_scaled * self.radiant_power[:, np.newaxis, np.newaxis]) # W

        self.x_LED_tot = np.sum(x_matrix_scaled) # total duty cycle assuming equal current for all LEDs

        return self.x_LED_tot, self.I_LED, self.P_r

##########################################################################
if __name__ == '__main__':
    U_BT = 3.7 # V
    LEDpar = LEDparams()

    x_LED, I_LED, P_r = LEDpar.program_reader()

    I_LED_tot = I_LED * x_LED # A
    P_LED = U_BT * I_LED_tot # W
    Q_LED_tot = P_LED - P_r # W
    
    print(f"Total duty cycle: {x_LED:.2f}")
    print(f"Total LED power: {P_LED:.2f} W")
    print(f"Total radiant power: {P_r:.2f} W")
    print(f"Total LED power loss: {Q_LED_tot:.2f} W")