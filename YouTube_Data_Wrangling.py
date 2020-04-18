#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The csv files (USvideos.csv, MXvideos.csv) are available @https://www.kaggle.com/datasnaek/youtube-new
#The dfs are created below from two files. 
#Steps to clean data are:
#1.Removing unnecessary columns
#2.Adding “category_name” column
#3.Dropping rows with NaN values
#4.Outliers


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
usa_df=pd.read_csv("USvideos.csv", parse_dates=[5])
mx_df=pd.read_csv("MXvideos.csv",engine='python')


# In[3]:


usa_df.info()


# In[4]:


mx_df.info()


# In[5]:


#usa_df['category_id'].unique()


# In[6]:


import requests
temp_category_usa={}
temp_category_mx={}

url="https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode=US&key=AIzaSyDj7XbNnUjmUHGKxrPEy_KBmuh6J5gDoes"
res=requests.get(url)
cat_json_usa=res.json()


url1="https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode=MX&key=AIzaSyDj7XbNnUjmUHGKxrPEy_KBmuh6J5gDoes"
res1=requests.get(url1)
cat_json_mx=res1.json()


for i in range(len(cat_json_usa['items'])):
    temp_category_usa[cat_json_usa['items'][i]['id']]=cat_json_usa['items'][i]['snippet']['title']
#category_usa


for i in range(len(cat_json_mx['items'])):
    temp_category_mx[cat_json_mx['items'][i]['id']]=cat_json_mx['items'][i]['snippet']['title']
#category_mx

#The temp_category_usa and temp_category_mx produce a dictionary with keys as strings and I need to redo that to have an integer value i.e.
#without ' ' around the number. If it is left with '' then map command cannot match the integer value in usa_df_dropcol to string val in temp_category_mx
category_usa= {int(old_key): val for old_key, val in temp_category_usa.items()}
category_mx= {int(old_key): val for old_key, val in temp_category_mx.items()}

#category_mx


# In[7]:


print(category_usa == category_mx)


# In[8]:


usa_df_dropcol=usa_df.drop(['video_id','thumbnail_link','description','tags','channel_title'], axis=1)


# In[9]:


mx_df_dropcol=mx_df.drop(['video_id','thumbnail_link','description','tags','channel_title'], axis=1)


# In[10]:


usa_df_dropcol.head(10)


# In[11]:


usa_df_dropcol['category_name']=usa_df_dropcol['category_id'].map(category_usa)


# In[12]:


mx_df_dropcol['category_name']=mx_df_dropcol['category_id'].map(category_mx)


# In[13]:


####For use later
###usa_df_dropcol[(usa_df_dropcol['likes']== 0) & (usa_df_dropcol['ratings_disabled'] == False)]


# In[14]:


usa_df_dropcol.info()
#From the info one can see that there are no null values in any of the cols for usa_df_dropcol


# In[15]:


mx_df_dropcol.info()
#From the info one can see that there are no null values in any of the cols for mx_df_dropcol except the new added col, 'category_name'
#Investingating


# In[16]:


temp_rows_drop=mx_df_dropcol
temp_rows_drop.info()


# In[17]:


rowstodel=mx_df_dropcol[mx_df_dropcol['category_name'].isnull()].index
mx_df_dropcol.drop(rowstodel, inplace=True)
mx_df_dropcol.info()


# In[18]:


usa_df_clean=usa_df_dropcol
mx_df_clean=mx_df_dropcol


# In[19]:


pd.options.display.float_format = '{:,.1f}'.format
usa_df_clean[['views','likes','dislikes','comment_count']].describe()


# In[20]:


mx_df_clean[['views','likes','dislikes','comment_count']].describe()


# In[21]:


#Searching Outliers


# In[22]:


#sns.boxplot(x=usa_df_clean['views'],showfliers = False)


# In[23]:


plt.subplot(1,2,1)
plt.title('USA-Outliers')
sns.boxplot(x=usa_df_clean['views'])

plt.subplot(1,2,2)
plt.title('Mexico-Outliers')
sns.boxplot(x=mx_df_clean['views'])


# In[24]:


#import numpy as np
#size=40949
#y=usa_df_clean['views']
#removed_outliers=y.between(y.quantile(.05),y.quantile(.95))
#print(str(y[removed_outliers].size) + "/" + str(len(usa_df_clean)) + " data points remain.")
#y[removed_outliers].plot.get_figure()

#sns.boxplot(x=y[removed_outliers])


# In[25]:


temp_mx=mx_df_clean #saving the cleaned seperate US df in another temp_mx


# In[26]:


temp_usa=usa_df_clean #saving the cleaned seperate MX df in another temp_mx


# In[27]:


#Combining the two country's data together. Added another column 'country' to both cleaned dataframes and them appended MX under USA df. 


# In[28]:


mx_df_clean['country']='Mexico'


# In[29]:


usa_df_clean['country']='USA'


# In[30]:


combined_usa_mx_df=usa_df_clean.append(mx_df_clean)


# In[31]:


combined_usa_mx_df.info()


# In[32]:


#RESETTING INDEX AS APPENDING MX DF TO USA DF made the index out of order
combined_usa_mx_df.reset_index(inplace = True, drop = True) 

#DROPPING DUPLICATES
combined_usa_mx_df.drop_duplicates(inplace=True)

#RESETTING INDEX one more time as dropping dups made the index out of order again
combined_usa_mx_df.reset_index(inplace = True, drop = True)

#combined_usa_mx_df.info()
#remaining records=81051


# In[33]:


# Making trend_date and publish time as datetime objects

#PUBLISH TIME:
combined_usa_mx_df['publish_time']=pd.to_datetime(combined_usa_mx_df['publish_time'],utc=True)
combined_usa_mx_df['publish_time'] = combined_usa_mx_df['publish_time'].apply(lambda x: x.date())
combined_usa_mx_df['publish_time']=pd.to_datetime(combined_usa_mx_df['publish_time'])


#TRENDING_DATE
#Since the trending_date is in non triditional format i.e stored as an object with format YY.DD.MM a function to reorder these 
#values is necessary. Tempdate func reorders the values to show YYMMDD

def tempdate(a):
    b=a.split(".")
    c=b[1]+b[2]+b[0]
    return c

combined_usa_mx_df['trending_date_new']=combined_usa_mx_df['trending_date'].apply(tempdate)
combined_usa_mx_df['trending_date_new']=pd.to_datetime(combined_usa_mx_df['trending_date_new'])
combined_usa_mx_df['trending_date']=combined_usa_mx_df['trending_date_new']
combined_usa_mx_df.drop(['trending_date_new'],axis=1,inplace=True)
combined_usa_mx_df['trending_date']=pd.to_datetime(combined_usa_mx_df['trending_date'])
#combined_usa_mx_df.info()
#Both dateTime columns have been converted to dateTime type successfully
###


# In[34]:


combined_usa_mx_df.info()


# In[36]:


#dropping all the rows where trending date is earlier than video's publishing date. Started with 81051 rows. Almost 10K rows will be dropped based on this criteria
#leaving 70795 rows

t=combined_usa_mx_df[(combined_usa_mx_df.trending_date) < (combined_usa_mx_df.publish_time)].index

combined_usa_mx_df.drop(t,inplace=True)
combined_usa_mx_df.info()


# In[ ]:




