import numpy as np


class LEDparams:
    def __init__(self) -> None:
        self.I_LED = 0.017 # A
        
        # TODO identify values
        radiant_power_dict = {
            'IR':       3.0,
            'Red':      4.0,
            'Orange':   4.0,
            'Green':    4.5,
            'Blue':     5.0,
            'Purple':   5.5,
            'UV':       6.0
        } # mW/cm^2

        # Convert to numpy array
        specific_radiant_power = np.array([radiant_power_dict[color] for color in radiant_power_dict.keys()]) # mW/cm^2

        # Single hole
        diameter = 7 # mm
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

        x_matrix_scaled = x_matrix / (2**16-1) # duty cycle

        self.P_r = np.sum(x_matrix_scaled * self.radiant_power[:, np.newaxis, np.newaxis]) # W

        self.x_LED_tot = np.sum(x_matrix_scaled) # total duty cycle assuming equal current for all LEDs

        return self.x_LED_tot, self.I_LED, self.P_r



##########################################################################
if __name__ == '__main__':
    print("ProgramReader class definition")