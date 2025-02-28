import paho.mqtt.client as mqtt
import time
import datetime
import pandas as pd
import threading
import tkinter as tk
from tkinter import filedialog
import os


class Communicator:
    """
    Class to communicate with the real device. Used to send control inputs
    """
    def __init__(self, diya_name:str, save_data:bool=False, save_frequency:int=60, choose_specific_directory:bool=False, verbose:bool=False) -> None:
        # Define the MQTT broker details
        self.BROKER = "mqtt.119.ovh"
        self.PORT = 1883
        self.TOPIC = "diya" + diya_name
        self.READ_TOPIC = self.TOPIC + "/rx/"
        self.TRANSMIT_TOPIC = self.TOPIC + "/tx/"
        self.KEEP_ALIVE = 10

        self.MEASUREMENT_NAME = ["T1_C", "T2_C", "T3_C", "x_HP", "x_FAN", "U_HP_mV", "I_HP_mA"]
        initial_values = [25, 25, 25, 0, 0, 0, 0]
        self.last_measurement: pd.DataFrame = pd.DataFrame([initial_values], columns=self.MEASUREMENT_NAME)

        self.SAVE_DATA:bool = save_data
        self.SAVE_FREQUENCY:int = save_frequency # in seconds
        self.measurements:pd.DataFrame = pd.DataFrame(columns=self.MEASUREMENT_NAME)
        
        # Variables for saving data at intervals
        self._last_append_time = datetime.datetime.now()
        self._stop_event = threading.Event()
        self.append_thread = threading.Thread(target=self._append_to_history_at_intervals)

        # Set the save directory
        self.save_directory = 'C:\\Users\\giaco\\Git_Repositories\\Semester_Thesis_1\\measurements'

        if choose_specific_directory:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.lift()  # Bring the window to the front
            root.attributes('-topmost', True)  # Keep the window on top of all others
            
            self.save_directory = filedialog.askdirectory(title='Select a Folder to Save the Measurements', initialdir=self.save_directory)
            root.destroy()

        self.verbose:bool = verbose

        # Create a new MQTT client instance
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.BROKER, self.PORT, self.KEEP_ALIVE)

    def on_connect(self, client, userdata, flags, rc, *args) -> None:
        # Reason code 0 means successful connection
        if rc == 0:
            client.subscribe(self.READ_TOPIC)
            if self.verbose:
                print(f"Subscribed to {self.READ_TOPIC}")
        else:
            if self.verbose:
                print("Connection failed")

    def _on_message_parser(self, message) -> None:
        """ message structure: TAS timestamp T_1 T_2 T_3 x_HP x_FAN U_HP I_HP"""

        parts = message.split()

        if parts == [] or "TAS" not in parts[0] or len(parts) != 9:
            if self.verbose:
                print("Received non-standard message")
            return
        
        index = index = datetime.datetime.now()
        values = [float(parts[2]), # T_1 [°C]
                  float(parts[3]), # T_2 [°C]
                  float(parts[4]), # T_3 [°C]
                  int(parts[5]),   # x_HP
                  int(parts[6]),   # x_FAN
                  float(parts[7]), # U_HP [mV]
                  float(parts[8])] # I_HP [mA]
        
        self.last_measurement = pd.DataFrame([values], columns=self.MEASUREMENT_NAME, index=[index])

        if self.verbose:
            print(self.last_measurement)

    def on_message(self, client, userdata, message) -> None:
        last_message = message.payload.decode()

        self._on_message_parser(last_message)

    def _append_to_history_at_intervals(self) -> None:
        while not self._stop_event.is_set():
            if self.last_measurement.empty:
                continue

            if (datetime.datetime.now() - self._last_append_time).seconds >= self.SAVE_FREQUENCY:
                if self.measurements.empty:
                    self.measurements = self.last_measurement
                else:
                    self.measurements = pd.concat([self.measurements, self.last_measurement])
                    self._last_append_time = datetime.datetime.now()
                if self.verbose:
                    print("Data saved to history")
            
            # Sleep briefly to avoid tight looping
            time.sleep(1)

    def send_control_input(self, I_HP:float, x_FAN:float) -> None:
        """ I_HP in A
            x_FAN in % """
        # sign_HP = "+" if I_HP >= 0 else "-"

        # Bound control values
        I_HP = max(min(I_HP, 6.6), -6.6) # TODO better bounds? and pass as ...
        x_FAN = max(min(x_FAN, 100), 0)

        # Format control values
        # I_HP_formatted = f"{sign_HP}{abs(int(I_HP)):03d}"
        # x_FAN_formatted = f"{int(x_FAN):03d}"
        # message_str = f"S{I_HP_formatted}{x_FAN_formatted}"

        I_HP_formatted = f"{abs(int(I_HP*1000)):07d}"
        message_str = f"X{I_HP_formatted}"

        self.client.publish(self.TRANSMIT_TOPIC, message_str)
        if self.verbose:
            print(f"Message sent: I_HP = {I_HP}, x_FAN = {x_FAN}")

    def save_measurements(self) -> None:
        if self.save_directory:
            # File name as current date and diya name
            current_time = time.strftime('%Y%m%d_%H%M%S')
            file_name = f"{current_time}_{self.TOPIC}.csv"
            file_path = os.path.join(self.save_directory, file_name)

            self.measurements.to_csv(file_path, index=True)

            print(f"Data saved to {file_path}.")
        else:
            print("No directory chosen. Data not saved.")

    def start(self) -> None:
        self.client.loop_start()

        if self.SAVE_DATA:
            self._last_append_time = datetime.datetime.now()
            self.append_thread.start()

        print("Communicator started")

    def stop(self) -> None:
        if self.SAVE_DATA:
            self._stop_event.set()
            self.append_thread.join()
            self.save_measurements()

        self.client.loop_stop()
        self.client.disconnect()
        print("Communicator stopped")

    def __del__(self) -> None:
        self.stop()
        print("Communicator deleted")

##########################################################################
if __name__ == '__main__':
    diya_name = "06"
    comm = Communicator(diya_name, save_data=False, save_frequency=2, choose_specific_directory=False, verbose=True)

    I_HP = 0
    x_FAN = 0

    continue_loop = True
    initial_time = datetime.datetime.now()
    duration = 11 # seconds

    try:
        comm.start()
        # comm.send_control_input(x_HP, x_FAN)

        while continue_loop:
            current_time = datetime.datetime.now()
            elapsed_time = current_time - initial_time
            if elapsed_time.total_seconds() > duration:
                continue_loop = False

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        comm.stop()