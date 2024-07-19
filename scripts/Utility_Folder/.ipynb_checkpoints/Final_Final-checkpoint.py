#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, isnan, when
from pyspark.sql.functions import to_timestamp,to_date,count
from pyspark.sql.functions import desc
from pyspark.sql.functions import desc, sum as sum_agg
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, datediff, sum as spark_sum, count, max as spark_max
from datetime import date
import scipy.stats as stat
import pylab  #,clust_plot
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import sys
sys.path.append('/Users/bhoomikan/Documents/Big_Data/')
from Final_Project.utility import kelbow,clust_plot,plot_data,outlier
#%%
spark = SparkSession.builder.appName("Final_Project").getOrCreate()

data = spark.read.csv('/Users/bhoomikan/Documents/Big_Data/bank_transactions.csv', header=True, inferSchema=True)

#%%
#Data Preprocessing

df = data.dropna()

flag = 0
for column in data.columns:
    condition = (col(column).isNull() | isnan(col(column)))
    if flag == 0:
        conditions = condition
        flag = 1
    else:
        conditions |= condition
        print(conditions)

filtered_df = data.where(~conditions)

filteredn_df = filtered_df.where((filtered_df['TransactionTime'] > 0) & (filtered_df['TransactionAmount (INR)'] > 0))
##some 1000 rows removed
date_format = "dd/MM/yy"
dfn = filteredn_df.withColumn("TransactionDate_N", to_date(filteredn_df["TransactionDate"], "d/M/yy"))

location_counts = dfn.groupBy('CustLocation').agg(count('TransactionID').alias('transaction_count'))
total_transactions = dfn.count()
location_percentages = location_counts.withColumn('percentage', (col('transaction_count') / total_transactions) * 100)

ordered_location_percentages = location_percentages.orderBy(desc('percentage'))
distinct_locations = ordered_location_percentages.select('CustLocation').distinct()
distinct_locations.count()

top_forty_location_percentages = ordered_location_percentages.limit(40)
top_forty_location_percentages.show()

sums_top_twenty = top_forty_location_percentages.agg(
    sum_agg(col('transaction_count')).alias('sum_transaction_count'),
    sum_agg(col('percentage')).alias('sum_percentage')
)

cust_locations_list = [row['CustLocation'] for row in top_forty_location_percentages.select('CustLocation').distinct().collect()]
df_f = dfn.where(dfn['CustLocation'].isin(cust_locations_list))

#%%
#Calculate recency,frequency and monetary
latest_date = date(2016, 10, 22)
# Define a UDF to calculate the recency in days
def calculate_recency(transaction_date):
    return (latest_date - transaction_date).days

# Register the UDF
calculate_recency_udf = udf(calculate_recency, IntegerType())

# Group by CustomerID and aggregate
rfm_df = df_f.groupBy("CustomerID").agg(
    calculate_recency_udf(spark_max("TransactionDate_N")).alias("Recency"),
    count("TransactionID").alias("Frequency"),
    spark_sum("TransactionAmount (INR)").alias("Monetary")
)

#%%
rfm_df_pd = rfm_df.toPandas()
#%%
plot_data(rfm_df_pd,'Recency')
plot_data(rfm_df_pd,'Frequency')
plot_data(rfm_df_pd,'Monetary')
#%%
rfm_df_pd['Recency_Boxcox'],parameters=stat.boxcox(rfm_df_pd['Recency']+1)
rfm_df_pd['Monetary_log'] = np.log1p(rfm_df_pd['Monetary'])
rfm_df_pd['Frequency_log'] = np.log1p(rfm_df_pd['Frequency'])
#%%
plot_data(rfm_df_pd,'Recency_Boxcox')
plot_data(rfm_df_pd,'Frequency_log')
plot_data(rfm_df_pd,'Monetary_log')
#%%
rfm_df_pd.describe()#644016
#%%
rfm_df_pd = outlier(rfm_df_pd,'Recency')
rfm_df_pd = outlier(rfm_df_pd,'Frequency')
rfm_df_pd = outlier(rfm_df_pd,'Monetary')
#%%
rfm_df_pd.describe()#507150
#%%
def binf(df,feature):
    binned_data = pd.cut(df[feature],bins=4)

    frequency = binned_data.value_counts()
    sorted_frequency = frequency.sort_index(ascending=False)
    sorted_frequency.plot(kind='bar')

    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Data Points in Each Bin:{feature}')
    plt.show()
binf(rfm_df_pd,'Recency')
binf(rfm_df_pd,'Recency_Boxcox')
binf(rfm_df_pd,'Frequency')
binf(rfm_df_pd,'Frequency_log')
binf(rfm_df_pd,'Monetary')
binf(rfm_df_pd,'Monetary_log')
#%%
df = spark.createDataFrame(rfm_df_pd)

assembler = VectorAssembler(inputCols=['Recency_Boxcox', 'Frequency_log', 'Monetary_log'], outputCol='features')
df_assembled = assembler.transform(df)

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=False)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

#%%
vectors = df_scaled.select("scaledFeatures").rdd.map(lambda row: row['scaledFeatures'].toArray())
local_vectors = vectors.collect()
numpy_array = np.array(local_vectors)
#print(numpy_array)
#%%
kelbow(numpy_array)  
#%%
k=4
kmeans = KMeans(featuresCol='scaledFeatures', k=k, seed=1, initMode='k-means||', maxIter=1000)
model = kmeans.fit(df_scaled)
predictions = model.transform(df_scaled)


evaluator = ClusteringEvaluator(featuresCol='scaledFeatures', metricName='silhouette', distanceMeasure='squaredEuclidean')
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared euclidean distance = {silhouette:.2f}")
#%%
print(predictions)
#%%
df_predict = predictions.withColumnRenamed('prediction', 'Cluster')
print(df_predict.show())
#%%
dfp_p = df_predict.toPandas()
f1= numpy_array[:,0]
f2= numpy_array[:,1]
f3=numpy_array[:,2]
f4 = dfp_p['Recency_Boxcox']
f5 = dfp_p['Frequency_log']
f6 = dfp_p['Monetary_log']
label_c = dfp_p['Cluster']
df = pd.DataFrame({
    'Recency': f1,
    'Frequency': f2,
    'Monetary': f3,
    'Recency_Boxcox': f4,
    'Frequency_log': f5,
    'Monetary_log': f6,
    'Clusters': label_c,
})
print(df.head(10))
#%%
df.describe()
#%%
#clust_plot(df)
#%%
plt.figure(figsize=(10, 6))
plt.title('Customer Segmentation based on Recency and Monetary')

# Plot each cluster with a different color and label
clusters = df['Clusters'].unique()
for cluster in clusters:
    subset = df[df['Clusters'] == cluster]
    plt.scatter(subset['Recency'], subset['Monetary'], label=cluster, s=50, cmap='Set1')

plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend(title='Cluster')
plt.show()
#%%
cluster_summary = df_predict.groupBy('Cluster').agg({'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'Cluster': 'count'})
cluster_summary.show()

# %%
clusters = dfp_p['Clusters'].unique()
cols=['CustomerID','Recency','Monetary','Cluster']
dfp_p = dfp_p[cols]
for cluster in clusters:
    subset = dfp_p[dfp_p['Clusters'] == cluster]
    filename = f"fname={cluster}.csv"
    subset.to_csv(filename, index=False)

#%%
cluster_summary = df_predict.groupBy('Cluster').agg({'Recency_Boxcox': 'mean','Monetary_log': 'mean', 'Cluster': 'count'})
cluster_summary.show()
# %%
binf(rfm_df_pd,'Recency_Boxcox')
binf(rfm_df_pd,'Monetary_log')
# %%
print(dfp_p.head(10))
# %%
