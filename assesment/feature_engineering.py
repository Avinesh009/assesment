import pandas as pd
import os

def engineer_features():
    # Load NASA data (uses 'id' column)
    if os.path.exists('clean_nasa.csv'):
        nasa = pd.read_csv('clean_nasa.csv')
        for window in [5, 10, 30]:
            for sensor in [f"sensor_{i}" for i in range(1,22)]:
                nasa[f'{sensor}_mean_{window}'] = nasa.groupby('id')[sensor].rolling(window).mean().values
        nasa.to_csv('features_nasa.csv', index=False)
    
    # Load Pump data (uses different grouping)
    if os.path.exists('clean_pump.csv'):
        pump = pd.read_csv('clean_pump.csv')
        
        # Use 'machine_id' or 'asset_id' if available, or use timestamp
        group_col = None
        for possible_col in ['machine_id', 'asset_id', 'timestamp']:
            if possible_col in pump.columns:
                group_col = possible_col
                break
        
        if group_col:
            for window in [5, 10, 30]:
                for sensor in [col for col in pump.columns if 'sensor_' in col]:
                    pump[f'{sensor}_delta_{window}'] = pump.groupby(group_col)[sensor].diff(window)
            pump.to_csv('features_pump.csv', index=False)
        else:
            print("Warning: No suitable grouping column found in pump data")

if __name__ == "__main__":
    engineer_features()