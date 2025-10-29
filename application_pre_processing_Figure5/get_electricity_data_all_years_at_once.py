import requests
import pandas as pd
import time

api_key = None
if api_key is None:
    print('please register for an api key on https://www.eia.gov/opendata/')
base_url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"

regions = {
    'California_ISO': {'respondent': 'CISO', 'timezone': 'Pacific'},
    'Texas_ERCOT': {'respondent': 'ERCO', 'timezone': 'Central'},
    'New_York': {'respondent': 'NYIS', 'timezone': 'Eastern'},
    'PJM': {'respondent': 'PJM', 'timezone': 'Eastern'},
    'Midwest': {'respondent': 'MISO', 'timezone': 'Central'},
    'Florida': {'respondent': 'FPL', 'timezone': 'Eastern'},
    'Duke_Energy': {'respondent': 'DUK', 'timezone': 'Eastern'},
    'Arizona': {'respondent': 'AZPS', 'timezone': 'Mountain'},
    'Bonneville': {'respondent': 'BPAT', 'timezone': 'Pacific'},
}

all_data = pd.DataFrame()

# Request ALL years at once for each region
for name, config in regions.items():
    print(f"{name}: ", end="")
    
    params = {
        'api_key': api_key,
        'frequency': 'daily',
        'data[0]': 'value',
        'start': '2019-01-01',  # Start date
        'end': '2024-12-31',     # End date  
        'facets[respondent][]': config['respondent'],
        'facets[type][]': 'D',
        'facets[timezone][]': config['timezone'],
        'length': 5000  # Max records - about 6 years worth
    }
    
    for attempt in range(3):  # Try 3 times if connection fails
        try:
            response = requests.get(base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and 'data' in data['response']:
                    df = pd.DataFrame(data['response']['data'])
                    if not df.empty:
                        df['region'] = name
                        all_data = pd.concat([all_data, df], ignore_index=True)
                        print(f"{len(df)} records")
                        break
                    else:
                        print("no data")
                        break
            else:
                print(f"error {response.status_code}")
                break
                
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            else:
                print(f"failed after 3 attempts")
    
    time.sleep(0.3)

# Save data
if not all_data.empty:
    all_data['period'] = pd.to_datetime(all_data['period'])
    all_data = all_data.sort_values(['region', 'period'])
    #change from megawatt to gigawatt hours
    all_data['value-units'] = 'gigawatthours'
    all_data['value'] = all_data['value'].astype(float) / 1000    
    all_data.to_csv('all_years_at_once_electricity_data.csv', index=False)
    
    print(f"\nTotal: {len(all_data)} records")
    print("\nBy region:")
    print(all_data.groupby('region').size())
else:
    print("No data downloaded")