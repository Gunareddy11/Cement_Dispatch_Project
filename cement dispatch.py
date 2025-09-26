import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Cement Dispatch\cement_dispatch_combined_dataset.csv")
df

df.shape

df.info()

df.head()

df.isnull().sum()

#Replacing Missing values with mode imputation for categorial values
# Fill missing values in truck_type with mode (most common type)
df['truck_type'] = df['truck_type'].fillna(df['truck_type'].mode()[0])
df['truck_type']

# Fill missing values in silo_id with mode
df['silo_id'] = df['silo_id'].fillna(df['silo_id'].mode()[0])
df['silo_id']

# Fill missing values in truck_number with mode
df['truck_number'] = df['truck_number'].fillna(df['truck_number'].mode()[0])
df['truck_number']

#Replacing Missing Values with median imputation for Numerical values
# Fill missing values in weight_before_kg with median
df['weight_before_kg'] = df['weight_before_kg'].fillna(df['weight_before_kg'].median())

# Fill missing values in weight_after_kg with median
df['weight_after_kg'] = df['weight_after_kg'].fillna(df['weight_after_kg'].median())

# Fill missing values in cement_loaded_kg with median
df['cement_loaded_kg'] = df['cement_loaded_kg'].fillna(df['cement_loaded_kg'].median())


#Repalcing Missing Values with forward fill for Time-series data
# Fill missing values in dispatch_duration with forward fill (previous value)
print(df.columns)

df['loading_start_time'] = pd.to_datetime(df['loading_start_time'])
df['loading_end_time'] = pd.to_datetime(df['loading_end_time'])

df['dispatch_duration'] = (df['loading_end_time'] - df['loading_start_time']).dt.total_seconds() / 60  # in minutes

df['dispatch_duration'] = df['dispatch_duration'].fillna(method='ffill')



df.isnull().sum()

# Fill truck_id using truck_number if available
df['truck_id'] = df['truck_id'].fillna(df['truck_number'])

# For any remaining nulls
df['truck_id'] = df['truck_id'].fillna("Unknown_Truck")

df['rfid_tag'] = df.groupby('truck_number')['rfid_tag'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['rfid_tag'] = df['rfid_tag'].fillna("Missing_RFID")

# Convert to datetime if not already
df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')

# Fill with mode per truck_id
df['registration_date'] = df.groupby('truck_id')['registration_date'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else pd.NaT))

# If still missing, use median date
df['registration_date'] = df['registration_date'].fillna(df['registration_date'].median())

# Generate surrogate IDs for missing event_id
df['event_id'] = df['event_id'].fillna(pd.Series("EVT_" + (df.index+1).astype(str), index=df.index))

# Convert to datetime
df['assignment_timestamp'] = pd.to_datetime(df['assignment_timestamp'], errors='coerce')
df['detection_timestamp'] = pd.to_datetime(df['detection_timestamp'], errors='coerce')

# Fill from detection_timestamp if available
df['assignment_timestamp'] = df['assignment_timestamp'].fillna(df['detection_timestamp'])

# Fill remaining with median assignment time
df['assignment_timestamp'] = df['assignment_timestamp'].fillna(df['assignment_timestamp'].median())


df['detection_timestamp'] = df['detection_timestamp'].fillna(df['assignment_timestamp'])
df['detection_timestamp'] = df['detection_timestamp'].fillna(df['detection_timestamp'].median())


df.isnull().sum()


# Summary statistics
print(df.describe(include='all'))

# Data types check again
print(df.dtypes)

# Unique values per column
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")



# Boxplot for numerical columns
sns.boxplot(x=df['cement_loaded_kg'])
plt.show()

sns.boxplot(x=df['dispatch_duration'])
plt.show()

# IQR method for detecting outliers
Q1 = df['cement_loaded_kg'].quantile(0.25)
Q3 = df['cement_loaded_kg'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['cement_loaded_kg'] < (Q1 - 1.5*IQR)) | 
              (df['cement_loaded_kg'] > (Q3 + 1.5*IQR))]
print("Outliers found:", outliers.shape[0])


# Turnaround time
df['turnaround_time'] = (df['loading_end_time'] - df['loading_start_time']).dt.total_seconds()/60
df['turnaround_time']

# SLA compliance (example: SLA = 60 minutes)
df['sla_met'] = np.where(df['turnaround_time'] <= 60, 1, 0)
df['sla_met']

# Misrouting flag
df['misrouted'] = np.where(df['assignment_timestamp'] != df['detection_timestamp'], 1, 0)
df['misrouted']

# Load accuracy
df['load_accuracy'] = df['weight_after_kg'] - df['weight_before_kg']
df['load_accuracy']


#Univarite Analysis
# Histogram for cement load
sns.histplot(df['cement_loaded_kg'], kde=True)
plt.show()

# Countplot for truck types
sns.countplot(x='truck_type', data=df)
plt.show()



#Bivariate Analysis
# Average turnaround by truck type
print(df.groupby('truck_type')['turnaround_time'].mean())

# Cement load by silo
df.groupby('silo_id')['cement_loaded_kg'].sum().plot(kind='bar')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


#Time Series Analysis
# Daily cement dispatch
df.groupby(df['assignment_timestamp'].dt.date)['cement_loaded_kg'].sum().plot(figsize=(10,4), title="Daily Cement Dispatch")
plt.show()

# SLA compliance over time
df.groupby(df['assignment_timestamp'].dt.date)['sla_met'].mean().plot(figsize=(10,4), title="SLA Compliance Over Time")
plt.show()








