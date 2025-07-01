"""
Database module for storing and retrieving sensor data.
Provides CSV-based data storage for automation systems.
"""

import csv
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Dict, List, Optional

class DataStorage:
    """Main class for data storage operations."""
    
    def __init__(self, db_path: str = "database/data.csv"):
        self.db_path = db_path
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'sensor_type', 'value', 'unit', 'device_id'])
    
    def store_sensor_data(self, sensor_type: str, value: float, unit: str = "", device_id: str = "default"):
        """Store sensor data with timestamp."""
        timestamp = datetime.now().isoformat()
        
        with open(self.db_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, sensor_type, value, unit, device_id])
    
    def get_sensor_data(self, sensor_type: Optional[str] = None, 
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> pd.DataFrame:
        """Retrieve sensor data with optional filtering."""
        
        df = pd.read_csv(self.db_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply filters
        if sensor_type:
            df = df[df['sensor_type'] == sensor_type]
        
        if start_time:
            df = df[df['timestamp'] >= pd.to_datetime(start_time)]
        
        if end_time:
            df = df[df['timestamp'] <= pd.to_datetime(end_time)]
        
        return df
    
    def clear_data(self):
        """Clear all stored data."""
        self._ensure_csv_exists()

def generate_realistic_sensor_data(duration_minutes: int = 20) -> List[Dict]:
    """Generate realistic sensor data for testing purposes."""
    data = []
    start_time = datetime.now() - timedelta(minutes=duration_minutes)
    
    # Generate data every 30 seconds
    time_points = int(duration_minutes * 2)
    
    for i in range(time_points):
        timestamp = start_time + timedelta(seconds=i * 30)
        
        # Temperature with daily variation
        base_temp = 23.0 + 2 * np.sin(i * 0.1) + np.random.normal(0, 0.5)
        
        # Vibration with some spikes
        base_vibration = 100 + 10 * np.sin(i * 0.05) + np.random.normal(0, 5)
        if np.random.random() < 0.1:  # 10% chance of spike
            base_vibration += np.random.normal(20, 10)
        
        # Fill level decreasing over time
        fill_level = 650 - (i * 2) + np.random.normal(0, 10)
        
        # Store data for different dispensers
        for dispenser in ['red', 'blue', 'green']:
            dispenser_offset = {'red': 0, 'blue': 5, 'green': -3}[dispenser]
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_type': f'temperature_{dispenser}',
                'value': base_temp + dispenser_offset + np.random.normal(0, 0.2),
                'unit': 'C',
                'device_id': f'dispenser_{dispenser}'
            })
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_type': f'vibration_index_{dispenser}',
                'value': base_vibration + dispenser_offset * 2 + np.random.normal(0, 2),
                'unit': 'index',
                'device_id': f'dispenser_{dispenser}'
            })
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'sensor_type': f'fill_level_{dispenser}',
                'value': max(0, fill_level + dispenser_offset * 10 + np.random.normal(0, 5)),
                'unit': 'grams',
                'device_id': f'dispenser_{dispenser}'
            })
    
    return data

def main():
    """Demonstration of the database functionality."""
    print("Initializing database...")
    
    # Test CSV storage
    csv_storage = DataStorage("database/data.csv")
    
    print("Generating realistic sensor data...")
    sensor_data = generate_realistic_sensor_data(20)  # 20 minutes of data
    
    print(f"Storing {len(sensor_data)} sensor readings...")
    for reading in sensor_data:
        csv_storage.store_sensor_data(
            reading['sensor_type'],
            reading['value'],
            reading['unit'],
            reading['device_id']
        )
    
    print("Data storage complete!")
    
    # Test data retrieval
    print("\nRetrieving temperature data...")
    temp_data = csv_storage.get_sensor_data(sensor_type='temperature_red')
    print(f"Found {len(temp_data)} temperature readings")
    
    print("\nRetrieving vibration data...")
    vib_data = csv_storage.get_sensor_data(sensor_type='vibration_index_blue')
    print(f"Found {len(vib_data)} vibration readings")
    
    print("\nDatabase demonstration complete!")

if __name__ == "__main__":
    main()
