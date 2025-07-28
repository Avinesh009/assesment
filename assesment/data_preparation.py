import pandas as pd

# NASA C-MAPSS
def clean_nasa_data():
    columns = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"sensor_{i}" for i in range(1,22)]
    train = pd.read_csv('train_FD001.txt', sep="\s+", header=None, names=columns).dropna(axis=1)
    test = pd.read_csv('test_FD001.txt', sep="\s+", header=None, names=columns).dropna(axis=1)
    
    # Add RUL
    def add_rul(df):
        max_cycle = df.groupby('id')['cycle'].max().reset_index()
        max_cycle.columns = ['id', 'max']
        df = df.merge(max_cycle, on='id', how='left')
        df['RUL'] = df['max'] - df['cycle']
        return df.drop('max', axis=1)
    
    train = add_rul(train)
    train.to_csv('clean_nasa.csv', index=False)

# Pump Sensor Data
def clean_pump_data():
    pump = pd.read_csv('pump_sensor.csv')
    pump['timestamp'] = pd.to_datetime(pump['timestamp'])
    pump['failure_flag'] = (pump['machine_status'] == 'BROKEN').astype(int)
    pump.to_csv('clean_pump.csv', index=False)

if __name__ == "__main__":
    clean_nasa_data()
    clean_pump_data()