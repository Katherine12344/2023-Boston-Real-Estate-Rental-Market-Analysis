# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:12:58 2024

@author: Kkath
"""

import pandas as pd
import requests
import warnings
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings('ignore')

url = "https://api.rentcast.io/v1/markets?zipCode=02115&historyRange=12"
headers = {
    "accept": "application/json",
    "X-Api-Key": "Your_api_key"
}
response = requests.get(url, headers=headers)
df1 = pd.json_normalize(data=response.json())

#View Historical Rental Data
zip_code ='02115'
# create empty list
df_list = []
#get all "detailed" columns
rent_detail_hist_cols = [x for x in df1.columns if 'dataByBedrooms' and 'history' in x]
# iterate through "detailed" columns
for x in rent_detail_hist_cols:
    # get column date
    date_str = x.split('.')[2]
    # get column name
    detail_col = 'rentalData.history.' + date_str + '.dataByBedrooms'
    # convert historical data to a dataframe
    _df = pd.DataFrame(df1[detail_col].iloc[0])
    # create columns
    _df['date_str'] = date_str
    _df['zip_code'] = zip_code
    # append to list
    df_list.append(_df)
# comine sub date dataframes
df_detail = pd.concat(df_list)

# move date column to front
# Define move_col_to_front
def move_col_to_front (df, col_name):
    front_col = df[col_name]
    df.drop(labels=[col_name], axis=1, inplace=True)
    df.insert(0, col_name, front_col)
    return df
df_detail = move_col_to_front(df_detail, col_name='date_str')
df_detail = move_col_to_front(df_detail, col_name='zip_code')

#View Historical All Data
# create empty list
df_list1 = []
#get all "detailed" columns
rent_hist_cols = [c for c in df1.columns if 'averageRent' and 'history' in c]
# iterate through "detailed" columns
for c in rent_hist_cols:
    # get column date
    date_str = c.split('.')[2]
    # get column name
    detail_col1 = 'rentalData.history.' + date_str + '.averageRent'
    # get the average rent value
    avg_rent = df1[detail_col1].iloc[0]
    # create a dictionary
    data_dict = {'average_rent': avg_rent, 'date_str': date_str}
    # convert the dictionary to a DataFrame
    _df1 = pd.DataFrame([data_dict])
    # append to list
    df_list1.append(_df1)
# combine sub date dataframes
df_all = pd.concat(df_list1)
df_all = df_all.drop_duplicates()

# Create a sqlite engine
engine = create_engine('sqlite://', echo=False)
# Explore the dataframe as a table to the sqlite engine
df_detail.to_sql("boston", con=engine, index=False)

# Create a Dataframe of all the rental prices of the whole market and detailed properties
with engine.begin() as conn:
    detail = text("""
    SELECT date_str, bedrooms, averageRent
    FROM boston
    GROUP BY date_str, bedrooms
    """)
    details = pd.read_sql_query(detail, conn)
# Pivot the DataFrame
pivoted_details = details.pivot(index='date_str', columns='bedrooms', values='averageRent')
# Rename the columns if necessary
pivoted_details.columns = ['0_bedrooms', '1_bedroom', '2_bedrooms', '3_bedrooms', '4_bedrooms', '5_bedrooms', '6_bedrooms']
# Reset the index so that 'date_str' becomes a column again
pivoted_details.reset_index(inplace=True)
merged_df = pd.merge(df_all, pivoted_details, on='date_str', how='inner')
merged_df = move_col_to_front(merged_df, col_name='date_str')
fourth_to_last_non_null = merged_df['6_bedrooms'][::-1].dropna().iloc[0]
merged_df['6_bedrooms'] = merged_df['6_bedrooms'][::-1].fillna(value=fourth_to_last_non_null, limit=3)[::-1]

#Average Rental Price
plt.figure(figsize=(10,5))
plt.plot(merged_df['date_str'],merged_df['average_rent'],marker='o', linestyle='-')
plt.title('Average Rent Price Trend by month')
plt.xlabel('Month')
plt.ylabel('Rent.avg($)')
plt.show()

# Explore Average Rent Price and Listings of Different Properties in the Last 12 Months
with engine.begin() as conn:
    bosquery = text("""
    SELECT bedrooms, ROUND(AVG(averageRent),2) AS avgrent, ROUND(AVG(totalListings),2) AS avglistings
    FROM boston
    GROUP BY bedrooms
    ORDER BY bedrooms ASC    
    """)
    roomasc = pd.read_sql_query(bosquery, conn)
# Plotting
fig, ax1 = plt.subplots(figsize=(15, 8))
# First bar plot for avgrent
roomasc.plot(kind='bar', x='bedrooms', y='avgrent', ax=ax1, width=0.4, position=1, color='b', legend=False)
# Creating a twin axis for avglistings
ax2 = ax1.twinx()
roomasc.plot(kind='bar', x='bedrooms', y='avglistings', ax=ax2, width=0.4, position=0, color='r', legend=False)
# Set the title and labels
plt.title('Average Rent and Total Listings by Number of Bedrooms in Boston')
ax1.set_xlabel('Number of Bedrooms')
ax1.set_ylabel('Average Rent ($)', color='b')
ax2.set_ylabel('Average Listings', color='r')
# Show the value of each bar on the top
for ax in [ax1, ax2]:
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# Adjust layout to make room for the legend (if any)
plt.tight_layout()
plt.show()

#plotting a line plot to show average price trend of different room type
# Retrieve the data from the database
with engine.begin() as conn:
    avg = text("""
    SELECT date_str, bedrooms, ROUND(AVG(averageRent),2) AS avgmonrent
    FROM boston
    GROUP BY date_str, bedrooms
    ORDER BY date_str, bedrooms
    """)
    avgrent = pd.read_sql_query(avg, conn)
# Ensure date_str is sorted in ascending order
avgrent['date_str'] = pd.to_datetime(avgrent['date_str'])
avgrent = avgrent.sort_values('date_str')
bedroom_counts = sorted(avgrent['bedrooms'].unique())
max_bedrooms = max(bedroom_counts)
# Normalize the bedroom count to the range of the colormap
norm = mcolors.Normalize(vmin=min(bedroom_counts), vmax=max_bedrooms)
# Now plot the data
plt.figure(figsize=(10, 5))
cmap = plt.cm.Reds
# First, get a list of unique bedroom counts in the DataFrame.
bedroom_counts = avgrent['bedrooms'].unique()
# Plot each bedroom count on a separate line.
for bedroom in bedroom_counts:
    subset = avgrent[avgrent['bedrooms'] == bedroom]
    color = cmap(norm(bedroom))
    plt.plot(subset['date_str'], subset['avgmonrent'], marker='o', linestyle='-', label=f'Bedrooms: {bedroom}',color=color)
plt.title('Average Rent Price Trend by Month and Number of Bedrooms')
plt.xlabel('Month')
plt.ylabel('Average Rent ($)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # This adds a legend to distinguish the different lines.
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlap
plt.show()

# plotting a bar plot to show Listings Number of Properties of Different Room Numbers
with engine.begin() as conn:
    listing = text("""
    SELECT date_str, bedrooms, totalListings
    FROM boston
    GROUP BY date_str, bedrooms
    """)
    listingall = pd.read_sql_query(listing, conn)
# Increase figure size
# Convert 'date_str' to datetime to sort and ensure proper plotting
listingall['date_str'] = pd.to_datetime(listingall['date_str'])
listingall.sort_values(by=['date_str', 'bedrooms'], inplace=True)
# Pivot the DataFrame to get unique dates as index and bedroom numbers as columns
pivot_df = listingall.pivot(index='date_str', columns='bedrooms', values='totalListings')
# Plotting
ax = pivot_df.plot(kind='bar', figsize=(15, 8), width=0.8)
# Rotate the x-axis labels and set the frequency of the ticks
plt.xticks(rotation=45)
# Set grid
ax.yaxis.grid(True)
# Set the title and labels
plt.title('Total Property Listings by Number of Bedrooms')
plt.xlabel('Month')
plt.ylabel('Listing Numbers')
# Set legend
plt.legend(title='Bedrooms', bbox_to_anchor=(1.05, 1), loc='upper left')
# Adjust layout to make room for the legend
plt.tight_layout()
# Show fewer x-axis labels to avoid clutter
n = max(len(pivot_df) // 12, 1)  # aiming for one label per month, adjust denominator as needed
ax.set_xticks(ax.get_xticks()[::n])
# Assuming the index is in datetime format, format the labels as 'YYYY-MM'
ax.set_xticklabels([pd.to_datetime(date).strftime('%Y-%m') if not pd.isnull(date) else '' for date in pivot_df.index][::n])
plt.show()

# Max and Min Rent comparison
with engine.begin() as conn:
    maxmin = text("""
    SELECT date_str, bedrooms, minRent, maxRent, averageRent
    FROM boston
    GROUP BY date_str, bedrooms
    """)
    maxmins = pd.read_sql_query(maxmin, conn)
# Create a figure with subplots for each unique bedroom number in the dataframe
unique_bedrooms = maxmins['bedrooms'].unique()
n_bedrooms = len(unique_bedrooms)
# Create a figure with appropriate number of subplots based on the number of unique bedroom counts
fig, axes = plt.subplots(n_bedrooms, 1, figsize=(10, n_bedrooms * 6), squeeze=False)
# Flatten the axes array for easy iteration
axes = axes.flatten()
# Loop through each subplot and plot the data for each bedroom count
for idx, bedroom in enumerate(unique_bedrooms):
    # Filter the dataframe for the current bedroom count
    df_bedrooms = maxmins[maxmins['bedrooms'] == bedroom]
    # Stacked bar plot for the current bedroom count
    df_bedrooms.set_index('date_str')[['minRent', 'maxRent']].plot(kind='bar', stacked=True, ax=axes[idx], legend=idx==0)
    # Setting the title for each subplot
    axes[idx].set_title(f'{bedroom} Bedroom(s)')
    # Setting x-axis label
    axes[idx].set_xlabel('Date')
    # Setting y-axis label for the first subplot only for clarity
    if idx == 0:
        axes[idx].set_ylabel('Rent')
# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Plotting correlation
correlation_matrix = merged_df.iloc[:, -8:].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.show()