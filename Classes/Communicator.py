import paho.mqtt.client as mqtt

class Communicator:
    def __init__(self, diya_name:str, verbose:bool=False) -> None:
        # Define the MQTT broker details
        self.broker = "mqtt.119.ovh"
        self.port = 1883
        self.topic = "diya" + diya_name
        self.read_topic = self.topic + "/rx/"
        self.transmit_topic = self.topic + "/tx/"
        self.keep_alive = 60

        self.last_message = None
        self.verbose = verbose

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port, self.keep_alive)

    def on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            client.subscribe(self.read_topic)
            print(f"Subscribed to {self.read_topic}")
        else:
            print("Connection failed")

    def _on_message_parser(self, message) -> None:
        """ Example message:
        message = 20000101T193040 TMP&U 25.44 27.21 33.60 Peltier: 80 Fan: 0"""
        parts = message.split()

        timestamp = parts[0]
        sensor_label = parts[1]
        T_1 = float(parts[2])
        T_2 = float(parts[3])
        T_3 = float(parts[4])

        peltier_value = int(parts[6])
        fan_value = int(parts[8])

        print("Measurements:", T_1, T_2, T_3)
        print("Peltier Value:", peltier_value)
        print("Fan Value:", fan_value)

    def on_message(self, client, userdata, message) -> None:
        self.last_message = message.payload.decode()
    
        # Check if the message matches the expected format
        if "Peltier:" in self.last_message and "Fan:" in self.last_message:
            # This is a full measurement message
            self._on_message_parser(self.last_message)
        else:
            # Handle other types of messages
            print("Received non-standard message:", self.last_message)

    def send_control_input(self, x_HP:int, x_FAN:int) -> None:
        sign_HP = "+" if x_HP >= 0 else "-"

        # Bound control values
        x_HP = max(min(x_HP, 100), -100)
        x_FAN = max(min(x_FAN, 100), 0)

        # Format control values
        x_HP_formatted = f"{sign_HP}{abs(int(x_HP)):03d}"
        x_FAN_formatted = f"{int(x_FAN):03d}"
        message_str = f"S{x_HP_formatted}{x_FAN_formatted}"

        self.client.publish(self.transmit_topic, message_str)
        print(f"Message sent: x_HP = {x_HP}, x_FAN = {x_FAN}")

    def start(self) -> None:
        self.client.loop_start()
        print("Communicator started")

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        print("Communicator stopped")

    def __del__(self) -> None:
        self.stop()
        print("Communicator deleted")

##########################################################################
import time

if __name__ == '__main__':
    diya_name = "06"
    comm = Communicator(diya_name, verbose=True)
    comm.start()
    comm.send_control_input(100, 0)
    time.sleep(5)
    comm.send_control_input(10, 0)