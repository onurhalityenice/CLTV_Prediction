##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Models
##############################################################

##############################################################
# PART 1: Preparing the Data
##############################################################

# Read flo_data_20K.csv data
import pandas as pd
import datetime as dt

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# Reading OmniChannel.csv data. Making a copy of the dataframe.
df_ = pd.read_csv("Miuul/WEEK_3/RFM/flo_data_20k.csv")
df = df_.copy()
df.head()

# Define outlier_thresholds and replace_with_thresholds functions required to suppress outliers
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# Suppression of outliers of "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"
num_cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in num_cols:
    replace_with_thresholds(df, col)

df.describe([0.5, 0.7, 0.8, 0.9, 0.95, 0.99]).T

# Omnichannel customers shop both online and offline platforms.
# Creation of new variables for the total number of purchases and spending of each customer.
df["New_order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["New_customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Converting the type of variables that express dates
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# PART 2: Creating the CLTV Data Structure
###############################################################

# Defining 2 days after the date of the last purchase in the data set as the analysis date
df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

# Creating a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values
cltv_df = pd.DataFrame()
cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "frequency": df["order_num_total"],
             "monetary_cltv_avg": df["customer_value_total"] / df["order_num_total"]})

cltv_df.head()

###############################################################
# PART 3: Creation of BG/NBD and Gamma-Gamma Models, calculation of 6-month CLTV
###############################################################

# Creation of BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate expected purchases from customers in 3 months
# and add them to cltv dataframe as exp_sales_3_month
cltv_df["exp_sales_3_Month"] = bgf.predict(4*3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# Estimate expected purchases from customers in 6 months
# and add them to cltv dataframe as exp_sales_6_month
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# A review of the top 10 buyers in the 3rd and 6th months. Making a comparison
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

# Creating the Gamma-Gamma model.
# Estimating the average value of customers and adding it to cltv dataframe as exp_average_value
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

# Calculation of 6 months CLTV and adding it to the dataframe with the name cltv
cltv = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency_cltv_weekly'],
                                    cltv_df['T_weekly'],
                                    cltv_df['monetary_cltv_avg'],
                                    time=6,
                                    freq="W",
                                    discount_rate=0.01)

cltv_df["cltv"] = cltv

# Observation of the 20 people with the highest CLTV value
cltv_df.sort_values("cltv",ascending=False)[:20]

###############################################################
# PART 4: Creating Segments by CLTV
###############################################################

# Segment all customers into 4 segments according to 6-month CLTV
# and add the group names to the dataset
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# Interpretation of segmenting customers into 4 segments based on CLTV scores
cltv_df.groupby('cltv_segment').agg({"cltv": ["mean", "count", "min", "max"],
                                     "recency_cltv_weekly":["mean", "count", "min", "max"],
                                     "T_weekly":["mean", "count", "min", "max"],
                                     "frequency":["mean", "count", "min", "max"]})


### THE END ###