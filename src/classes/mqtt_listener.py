import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox


class MQTTListener:
    def __init__(self, broker="mqtt.119.ovh", topic_rx="diya07/rx/#", topic_tx="diya07/tx"):
        self.broker = broker
        self.topic_rx = topic_rx
        self.topic_tx = topic_tx
        self.client = mqtt.Client()
        
        # Set up callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Data storage for plotting
        self.data = {
            'time': [],
            'Temperature_1': [],
            'Temperature_2': [],
            'Temperature_3': [],
            'Temperature_setpoint': [],
            'Peltier_percent': [],
            'Fan': [],
            'Peltier_voltage': [],
            'Peltier_current': []
        }
        
        self.plotting_paused = False
        
        # Add LED state storage (96 wells, initially all off)
        self.led_states = [0] * 96  # 0 for off, 1 for on
        
        # Initialize GUI
        self.init_gui()
        
    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker"""
        if rc == 0:
            print(f"Connected to MQTT Broker: {self.broker}")
            # Subscribe to the topic
            self.client.subscribe(self.topic_rx)
        else:
            print(f"Failed to connect, return code: {rc}")
            
    def on_message(self, client, userdata, msg):
        """Callback for when a message is received"""
        try:
            # Split the message into fields
            fields = msg.payload.decode().split()
            
            if len(fields) == 9:  # Ensure we have all expected fields
                message_data = {
                    'Message_type': fields[0],
                    'Timestamp': fields[1],
                    'Temperature_1': float(fields[2]),
                    'Temperature_2': float(fields[3]),
                    'Temperature_3': float(fields[4]),
                    'Peltier_percent': float(fields[5]),
                    'Fan': int(fields[6]),
                    'Peltier_voltage': float(fields[7]),
                    'Peltier_current': float(fields[8])
                }
                
                # Update data for plotting
                self.data['time'].append(time.time())
                self.data['Temperature_setpoint'].append(self.temp_slider.val)
                for key in message_data:
                    if key in self.data:
                        self.data[key].append(message_data[key])
                
                print(f"\nReceived message on topic: {msg.topic}")
                for key, value in message_data.items():
                    print(f"{key}: {value}")
            elif fields[0] == "LED":  # LED state message
                # Expecting format: "LED index1 state1 index2 state2 ..."
                for i in range(1, len(fields), 2):
                    try:
                        index = int(fields[i])
                        state = int(fields[i + 1])
                        if 0 <= index < 96:
                            self.led_states[index] = state
                    except (ValueError, IndexError):
                        print(f"Invalid LED data format")
                self.update_led_display()
            else:
                print(f"Invalid message format: {msg.payload.decode()}")
                
        except Exception as e:
            print(f"Error parsing message: {e}")
            print(f"Raw message: {msg.payload.decode()}")
        
    def init_gui(self):
        """Initialize the matplotlib-based GUI components"""
        # Create main figure with extra space for controls
        self.fig = plt.figure(figsize=(15, 8))
        
        # Create a grid layout with more space between plots and controls
        gs = self.fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.5])
        
        # Create subplot area (3x2) for plots
        self.axs = []
        for i in range(3):
            for j in range(2):
                self.axs.append(self.fig.add_subplot(gs[i, j]))
        
        # Create control panel area on the right
        control_ax = self.fig.add_subplot(gs[:, 2])
        control_ax.axis('off')
        
        # Add LED plate display at the top right
        plate_height = 0.25  # Adjust this value to change size
        plate_width = 0.4
        self.plate_ax = plt.axes([0.75, 0.85, plate_width, plate_height])
        self.plate_ax.set_title('LED States')
        self.update_led_display()
        
        # Add controls in the right column with better spacing
        controls_bbox = control_ax.get_position()
        x0, y0, width, height = controls_bbox.x0, controls_bbox.y0, controls_bbox.width, controls_bbox.height
        
        # Add padding to prevent overlap
        x_pad = width * 0.1
        x0 += x_pad
        width -= 2 * x_pad
        
        # Add play/pause button at the top
        play_button_ax = plt.axes([x0, y0 + 0.95*height, width*0.9, 0.04])
        self.play_button = Button(play_button_ax, 'Pause')
        self.play_button.on_clicked(self.toggle_plotting)
        
        # Vertical spacing for controls
        slider_height = 0.03
        textbox_height = 0.03
        group_spacing = 0.12
        
        # Temperature setpoint controls
        current_y = y0 + 0.85*height
        plt.figtext(x0, current_y, 'Temperature Setpoint (°C)', fontsize=9, ha='left')
        
        current_y -= 0.05
        temp_slider_ax = plt.axes([x0, current_y, width*0.9, slider_height])
        self.temp_slider = Slider(temp_slider_ax, '', 0, 50, valinit=25)
        
        current_y -= 0.05
        self.temp_textbox = plt.axes([x0 + width*0.3, current_y, width*0.3, textbox_height])
        self.temp_text = TextBox(self.temp_textbox, '', initial='25.0')
        
        # Peltier control
        current_y -= group_spacing
        plt.figtext(x0, current_y, 'Peltier Power (%)', fontsize=9, ha='left')
        
        current_y -= 0.05
        peltier_slider_ax = plt.axes([x0, current_y, width*0.9, slider_height])
        self.peltier_slider = Slider(peltier_slider_ax, '', -100, 100, valinit=0)
        
        current_y -= 0.05
        self.peltier_textbox = plt.axes([x0 + width*0.3, current_y, width*0.3, textbox_height])
        self.peltier_text = TextBox(self.peltier_textbox, '', initial='0')
        
        # Fan control
        current_y -= group_spacing
        plt.figtext(x0, current_y, 'Fan Speed (%)', fontsize=9, ha='left')
        
        current_y -= 0.05
        fan_slider_ax = plt.axes([x0, current_y, width*0.9, slider_height])
        self.fan_slider = Slider(fan_slider_ax, '', 0, 100, valinit=0)
        
        current_y -= 0.05
        self.fan_textbox = plt.axes([x0 + width*0.3, current_y, width*0.3, textbox_height])
        self.fan_text = TextBox(self.fan_textbox, '', initial='0')
        
        # Remove send button and connect callbacks that send immediately
        self.temp_slider.on_changed(self.on_temp_slider_change)
        self.temp_text.on_submit(self.on_temp_text_change)
        self.peltier_slider.on_changed(self.on_peltier_slider_change)
        self.peltier_text.on_submit(self.on_peltier_text_change)
        self.fan_slider.on_changed(self.on_fan_slider_change)
        self.fan_text.on_submit(self.on_fan_text_change)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.3,
            hspace=0.4
        )
        
        # Animation function
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=1000)

    def send_control_values(self):
        """Send control values for Peltier and Fan"""
        peltier_value = int(self.peltier_slider.val)
        fan_value = int(self.fan_slider.val)
        temp_setpoint = float(self.temp_slider.val)
        message = f"Peltier {peltier_value} Fan {fan_value} Temp {temp_setpoint:.1f}"
        self.client.publish(self.topic_tx, message)
        print(f"Sent control values: {message}")

    def on_temp_slider_change(self, val):
        self.temp_text.set_val(f"{val:.1f}")
        # Store setpoint in data
        if self.data['time']:  # If we have any data points
            self.data['Temperature_setpoint'].append(val)
        self.send_control_values()

    def on_temp_text_change(self, text):
        try:
            val = float(text)
            if 0 <= val <= 50:
                self.temp_slider.set_val(val)
                self.send_control_values()
            else:
                self.temp_text.set_val(f"{self.temp_slider.val:.1f}")
        except ValueError:
            self.temp_text.set_val(f"{self.temp_slider.val:.1f}")

    def on_peltier_slider_change(self, val):
        self.peltier_text.set_val(f"{int(val)}")
        self.send_control_values()

    def on_peltier_text_change(self, text):
        try:
            val = int(float(text))
            if -100 <= val <= 100:
                self.peltier_slider.set_val(val)
                self.send_control_values()
            else:
                self.peltier_text.set_val(f"{int(self.peltier_slider.val)}")
        except ValueError:
            self.peltier_text.set_val(f"{int(self.peltier_slider.val)}")

    def on_fan_slider_change(self, val):
        self.fan_text.set_val(f"{int(val)}")
        self.send_control_values()

    def on_fan_text_change(self, text):
        try:
            val = int(float(text))
            if 0 <= val <= 100:
                self.fan_slider.set_val(val)
                self.send_control_values()
            else:
                self.fan_text.set_val(f"{int(self.fan_slider.val)}")
        except ValueError:
            self.fan_text.set_val(f"{int(self.fan_slider.val)}")

    def toggle_plotting(self, event):
        """Toggle the plotting state between play and pause"""
        self.plotting_paused = not self.plotting_paused
        self.play_button.label.set_text('Play' if self.plotting_paused else 'Pause')
        plt.draw()

    def update_plot(self, frame):
        """Update the plot with new data"""
        if self.plotting_paused:
            return
        
        for ax in self.axs:
            ax.clear()
        
        # Plot temperatures and setpoint in first plot
        self.axs[0].plot(self.data['time'], self.data['Temperature_1'], label='Temp 1')
        self.axs[0].plot(self.data['time'], self.data['Temperature_2'], label='Temp 2')
        self.axs[0].plot(self.data['time'], self.data['Temperature_3'], label='Temp 3')
        self.axs[0].plot(self.data['time'], self.data['Temperature_setpoint'], 
                        'k--', label='Setpoint', linewidth=1.5)
        
        # Plot current in second plot (swapped position)
        self.axs[1].plot(self.data['time'], self.data['Peltier_current'], label='Current')
        
        # Plot Peltier control in third plot (swapped position)
        self.axs[2].plot(self.data['time'], self.data['Peltier_percent'], label='Peltier %')
        
        # Rest of the plots
        self.axs[3].plot(self.data['time'], self.data['Fan'], label='Fan %')
        self.axs[4].plot(self.data['time'], self.data['Peltier_voltage'], label='Voltage')
        
        # Power calculation in last plot
        if self.data['time']:  # Only if we have data
            power = [v * i for v, i in zip(self.data['Peltier_voltage'], self.data['Peltier_current'])]
            self.axs[5].plot(self.data['time'], power, label='Power (W)')
        
        # Set titles and labels
        titles = ['Temperatures', 'Current', 'Peltier Control',
                 'Fan Control', 'Voltage', 'Power']
        for ax, title in zip(self.axs, titles):
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
        
        # Special formatting for temperature plot
        self.axs[0].set_ylabel('Temperature (°C)')
        
        # Update LED display
        self.update_led_display()
        
        self.fig.canvas.draw()
        
    def update_led_display(self):
        """Update the LED plate display"""
        self.plate_ax.clear()
        
        # Create 8x12 grid
        for i in range(8):  # rows A-H
            for j in range(12):  # columns 1-12
                index = i * 12 + j
                color = 'green' if self.led_states[index] else 'lightgray'
                
                # Draw circle for each well
                circle = plt.Circle((j + 0.5, 7.5 - i), 0.3, 
                                  color=color, fill=True)
                self.plate_ax.add_artist(circle)
                
                # Add row labels (A-H) on the left side
                if j == 0:
                    self.plate_ax.text(-0.2, 7.5 - i, chr(65 + i), 
                                     ha='right', va='center')
                
                # Add column labels (1-12) on the top
                if i == 0:
                    self.plate_ax.text(j + 0.5, 8.7, str(j + 1), 
                                     ha='center', va='bottom')
        
        # Set axis limits and aspect ratio
        self.plate_ax.set_xlim(-0.5, 12.5)
        self.plate_ax.set_ylim(-0.5, 9)
        self.plate_ax.set_aspect('equal')
        self.plate_ax.axis('off')
        
    def start(self):
        """Start the MQTT client and GUI"""
        try:
            # Connect to the broker
            self.client.connect(self.broker, 1883, 60)
            self.client.loop_start()
            
            # Show the plot (replaces Tk mainloop)
            plt.show()
            
        except KeyboardInterrupt:
            print("\nDisconnecting from broker")
            self.client.disconnect()
            self.client.loop_stop()
        except Exception as e:
            print(f"Error: {e}")
            self.client.disconnect()
            self.client.loop_stop()

if __name__ == "__main__":
    # Create and start the listener
    listener = MQTTListener()
    listener.start()