# Import libraries
import pandas as pd
import numpy as np

# Load the data
cgm_data = pd.read_csv('CGMData.csv')
insulin_data = pd.read_csv('InsulinData.csv')

# Extract the required columns CGM data_data [B,C,AE] from insulin_data [B,C,Q]
cgm_data = cgm_data.iloc[:,[1,2,30]]
insulin_data = insulin_data.iloc[:, [1, 2, 16]]

# Combine date and time columns into single datetime column and place it at the beginning
cgm_data.insert(0, 'datetime', pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time']))
insulin_data.insert(0, 'datetime', pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time']))

# Drop original Date and Time columns
cgm_data = cgm_data.drop(['Date', 'Time'], axis=1)
insulin_data = insulin_data.drop(['Date', 'Time'], axis=1)

# Segment cgm data
cgm_data['day'] = pd.to_datetime(cgm_data['datetime']).dt.date

daily_cgm_segments = cgm_data.groupby('day').filter(lambda x: len(x) == 288 and not x['Sensor Glucose (mg/dL)'].isnull().any())
daily_segments = daily_cgm_segments.sort_values(by='datetime', ascending=False)

# Determine the timestamp when Auto mode starts.
insulin_data.sort_values(by='datetime',inplace=True, ascending=False)
auto_mode_timestamp = insulin_data[insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0,0]

# Get the data from CGM starting from the timestamp when Auto mode starts.
auto_data = daily_segments[daily_cgm_segments['datetime'] >= auto_mode_timestamp]

# Get the manual mode data
manual_data = daily_segments[daily_cgm_segments['datetime'] < auto_mode_timestamp]

# set datetime as index 
auto_data.set_index('datetime', inplace=True)
manual_data.set_index('datetime', inplace=True)

def calculate_metrics(data):

    # Percentage time in hyperglycemia (CGM > 180 mg/dL)
    hyperglycemia = data.loc[data["Sensor Glucose (mg/dL)"] > 180].groupby('day')['Sensor Glucose (mg/dL)'].count() / 288 * 100

    # percentage of time in hyperglycemia critical (CGM > 250 mg/dL)
    hypoglycemia_critical = data.loc[data["Sensor Glucose (mg/dL)"] > 250].groupby('day')['Sensor Glucose (mg/dL)'].count() / 288 * 100
    
    # percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)
    time_in_range = data.loc[(data['Sensor Glucose (mg/dL)'] >= 70) & (data['Sensor Glucose (mg/dL)'] <= 180)].groupby('day')['Sensor Glucose (mg/dL)'].count() / 288 * 100

    # percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)
    time_in_range_secondary = data.loc[(data['Sensor Glucose (mg/dL)'] >= 70) & (data['Sensor Glucose (mg/dL)'] <= 150)].groupby('day')['Sensor Glucose (mg/dL)'].count() / 288 * 100

    # percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)
    hypoglycemia_level_1 = data.loc[data['Sensor Glucose (mg/dL)'] < 70].groupby('day')['Sensor Glucose (mg/dL)'].count() / 288 * 100

    # percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)
    hypoglycemia_level_2 = data.loc[data['Sensor Glucose (mg/dL)'] < 54].groupby('day')['Sensor Glucose (mg/dL)'].count() / 288 * 100

    # Combine the metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'hyperglycemia': hyperglycemia,
        'hypoglycemia_critical': hypoglycemia_critical,
        'time_in_range': time_in_range,
        'time_in_range_secondary': time_in_range_secondary,
        'hypoglycemia_level_1': hypoglycemia_level_1,
        'hypoglycemia_level_2': hypoglycemia_level_2
    })

    # Calculate the mean for each metric
    metrics_mean = metrics_df.mean()

    return metrics_mean

auto_overnight_data = calculate_metrics(auto_data.between_time("0:00:00", "05:59:59"))
auto_daytime_data = calculate_metrics(auto_data.between_time("6:00:00", "23:59:59"))
auto_wholeday_data = calculate_metrics(auto_data)

manual_overnight_data = calculate_metrics(manual_data.between_time("0:00:00", "05:59:59"))
manual_daytime_data = calculate_metrics(manual_data.between_time("6:00:00", "23:59:59"))
manual_wholeday_data = calculate_metrics(manual_data)

# Combine results into a DataFrame
results = pd.DataFrame({
    'Manual Mode': [
        manual_overnight_data[metric] for metric in manual_overnight_data.keys()
    ] + [
        manual_daytime_data[metric] for metric in manual_daytime_data.keys()
    ] + [
        manual_wholeday_data[metric] for metric in manual_wholeday_data.keys()
    ],
    'Auto Mode': [
        auto_overnight_data[metric] for metric in auto_overnight_data.keys()
    ] + [
        auto_daytime_data[metric] for metric in auto_daytime_data.keys()
    ] + [
        auto_wholeday_data[metric] for metric in auto_wholeday_data.keys()
    ]
})

# Transpose the DataFrame to match the desired format
results_transposed = results.T
results_transposed.fillna(0, inplace=True)

# Save the transposed DataFrame to a CSV file
results_transposed.to_csv('Result.csv', header=False, index=False)
