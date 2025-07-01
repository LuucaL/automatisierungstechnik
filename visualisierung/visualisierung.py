"""
Visualization module for sensor data analysis.
Provides simple time series plotting functionality.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import DataStorage, generate_realistic_sensor_data

# Set style for better looking plots
plt.style.use('seaborn-v0_8')

class SensorDataVisualizer:
    """Class for visualizing sensor data."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_time_series(self, data: pd.DataFrame, sensor_type: str, 
                        title: str = None, save_path: str = None):
        """Plot time series data for a specific sensor type."""
        
        # Filter data for specific sensor type
        sensor_data = data[data['sensor_type'] == sensor_type]
        
        if len(sensor_data) == 0:
            print(f"No data found for sensor type: {sensor_type}")
            return
        
        plt.figure(figsize=self.figsize)
        
        # Group by device_id if multiple devices
        devices = sensor_data['device_id'].unique()
        
        for device in devices:
            device_data = sensor_data[sensor_data['device_id'] == device]
            plt.plot(device_data['timestamp'], device_data['value'], 
                    label=f"{device}", linewidth=2, alpha=0.8)
        
        plt.title(title or f"Zeitreihe: {sensor_type}")
        plt.xlabel("Zeit")
        plt.ylabel(f"Wert ({sensor_data['unit'].iloc[0] if len(sensor_data) > 0 else ''})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_multi_series(self, data: pd.DataFrame, sensor_types: list, 
                         title: str = None, save_path: str = None):
        """Plot multiple sensor types on the same graph."""
        
        plt.figure(figsize=self.figsize)
        
        for sensor_type in sensor_types:
            # Filter data for specific sensor type
            sensor_data = data[data['sensor_type'] == sensor_type]
            
            if len(sensor_data) == 0:
                print(f"No data found for sensor type: {sensor_type}")
                continue
            
            # Use the first device if multiple exist
            if len(sensor_data['device_id'].unique()) > 0:
                device = sensor_data['device_id'].unique()[0]
                device_data = sensor_data[sensor_data['device_id'] == device]
                plt.plot(device_data['timestamp'], device_data['value'], 
                        label=f"{sensor_type} ({device})", linewidth=2, alpha=0.8)
        
        plt.title(title or "Multi-Sensor Zeitreihe")
        plt.xlabel("Zeit")
        plt.ylabel("Wert")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

def main():
    """Demonstration of visualization functionality."""
    print("Initializing database...")
    storage = DataStorage()
    
    # Generate sample data if none exists
    if len(storage.get_sensor_data()) == 0:
        print("Generating sample data...")
        sensor_data = generate_realistic_sensor_data(30)  # 30 minutes of data
        for reading in sensor_data:
            storage.store_sensor_data(
                reading['sensor_type'],
                reading['value'],
                reading['unit'],
                reading['device_id']
            )
    
    print("Retrieving data for visualization...")
    data = storage.get_sensor_data()
    
    visualizer = SensorDataVisualizer()
    
    # Plot individual sensor types
    visualizer.plot_time_series(data, 'temperature_red', 
                              "Temperatur Zeitreihe (Rot)")
    
    visualizer.plot_time_series(data, 'vibration_index_blue', 
                              "Vibrations-Index Zeitreihe (Blau)")
    
    # Plot multiple sensors
    visualizer.plot_multi_series(
        data, 
        ['temperature_red', 'temperature_blue', 'temperature_green'],
        "Temperaturvergleich aller Sensoren"
    )
    
    print("Visualization demonstration complete!")

if __name__ == "__main__":
    main()
