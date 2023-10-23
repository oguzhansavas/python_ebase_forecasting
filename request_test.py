import requests
import json
import pandas as pd

def data_preprocessing(json):
	'''
	Prepares dataframe from json.
	json: json (type: dict) to be preprocessed
	'''
	# Get timeseries info (datetime and values)
	json_list = pd.Series(json['data']['timeSeries'][0])
	dataframe = pd.DataFrame(json_list['points'])

	# Create datetime column, convert to 'datetime' object
	dataframe['time'] = dataframe['time'].replace(':00.000Z', '')
	dataframe['DateTime'] = pd.to_datetime(dataframe['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
	dataframe = dataframe.drop('time', axis=1).set_index('DateTime')

	# Get all numeric features
	dataframe_num = dataframe.select_dtypes(include=['float', 'int'])
	numeric_features = [col for col in dataframe_num.columns]

	# Make sure all numeric features are float
	for feature in numeric_features:
		dataframe = dataframe.astype({feature:'float'})

	return dataframe


# defined the URL variable
url = "http://192.168.168.108:8080/graphql/"
 
body = """
q{ timeSeries(
    codes: ["616807080106809804TM"]
    startPeriod: "2023-03-25T23:00:00Z"
    endPeriod: "2023-03-26T22:00:00Z"
  ) {
    code
    startPeriod
    endPeriod
    version
    interval
    nanHandling
    unit
    points {
      time
      value
    }
  }
}
"""
 
response = requests.post(url=url, json={"query": body})
if response.status_code == 200:
    print("response : ", response.content)
json_data = json.loads(response.content)
meter_df = data_preprocessing(json_data)
