import paho.mqtt.client as mqtt
import time
import datetime
import pandas as pd
from IPython.display import display
import threading
import tkinter as tk
from tkinter import filedialog
import os


class Communicator:
    def __init__(self, diya_name:str, save_data:bool=False, save_frequency:int=60, choose_specific_directory:bool=False, verbose:bool=False) -> None:
        # Define the MQTT broker details
        self.BROKER = "mqtt.119.ovh"
        self.PORT = 1883
        self.TOPIC = "diya" + diya_name
        self.READ_TOPIC = self.TOPIC + "/rx/"
        self.TRANSMIT_TOPIC = self.TOPIC + "/tx/"
        self.KEEP_ALIVE = 60

        self.MEASUREMENT_NAME = ["T_1", "T_2", "T_3", "x_HP", "x_FAN"]
        self.last_measurement:pd.DataFrame = pd.DataFrame(columns=self.MEASUREMENT_NAME)

        self.SAVE_DATA:bool = save_data
        self.SAVE_FREQUENCY:int = save_frequency # in seconds
        self.measurements:pd.DataFrame = pd.DataFrame(columns=self.MEASUREMENT_NAME)
        
        # Variables for saving data at intervals
        self._last_append_time = datetime.now()
        self._stop_event = threading.Event()
        self.append_thread = threading.Thread(target=self._append_to_history_at_intervals)

        # Set the save directory
        if choose_specific_directory:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.lift()  # Bring the window to the front
            root.attributes('-topmost', True)  # Keep the window on top of all others
            
            self.save_directory = filedialog.askdirectory(title='Select a Folder to Save the Measurements')
            root.destroy()
            
        else:
            self.save_directory = 'C:\\Users\\giaco\\Git_Repositories\\Semester_Thesis_1\\Measurements'

        self.verbose:bool = verbose

        # Create a new MQTT client instance
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.BROKER, self.PORT, self.KEEP_ALIVE)

    def on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            client.subscribe(self.READ_TOPIC)
            if self.verbose:
                print(f"Subscribed to {self.READ_TOPIC}")
        else:
            if self.verbose:
                print("Connection failed")

    def _on_message_parser(self, message) -> None:
        """ Example message:
        message = 20000101T193040 TMP&U 25.44 27.21 33.60 Peltier: 80 Fan: 0"""
        parts = message.split()

        index = index = datetime.datetime.now()
        values = [float(parts[2]), float(parts[3]), float(parts[4]), int(parts[6]), int(parts[8])]
        self.last_measurement = pd.DataFrame([values], columns=self.MEASUREMENT_NAME, index=[index])

        display(self.last_measurement)

    def on_message(self, client, userdata, message) -> None:
        last_message = message.payload.decode()
    
        # Check if the message matches the expected format
        if "Peltier:" in last_message and "Fan:" in last_message:
            # This is a full measurement message
            self._on_message_parser(last_message)
        else:
            print("Received non-standard message:", last_message)

    def _append_to_history_at_intervals(self) -> None:
        while not self._stop_event.is_set():
            if (datetime.now() - self._last_append_time).seconds >= self.SAVE_FREQUENCY:
                self.measurements = pd.concat([self.measurements, self.last_measurement])
                self._last_append_time = datetime.now()
                if self.verbose:
                    print("Data saved to history")
            
            # Sleep briefly to avoid tight looping
            time.sleep(1)

    def send_control_input(self, x_HP:int, x_FAN:int) -> None:
        sign_HP = "+" if x_HP >= 0 else "-"

        # Bound control values
        x_HP = max(min(x_HP, 100), -100)
        x_FAN = max(min(x_FAN, 100), 0)

        # Format control values
        x_HP_formatted = f"{sign_HP}{abs(int(x_HP)):03d}"
        x_FAN_formatted = f"{int(x_FAN):03d}"
        message_str = f"S{x_HP_formatted}{x_FAN_formatted}"

        self.client.publish(self.TRANSMIT_TOPIC, message_str)
        print(f"Message sent: x_HP = {x_HP}, x_FAN = {x_FAN}")

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
            self._last_append_time = datetime.now()
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
    comm = Communicator(diya_name, verbose=True)
    comm.start()
    comm.send_control_input(100, 0)
    time.sleep(5)
    comm.send_control_input(10, 0)
    comm.stop()