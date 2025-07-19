import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import logging  # For logging messages
import matplotlib.pyplot as plt  # For plotting
import statistics as stats  # For statistical computations
import seaborn as sns  # For visualizations

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load dataset
df_main = pd.read_excel("Lab Session Data.xlsx")

# Extract feature matrix and target vector
features = df_main.iloc[:, 1:4].values
target = df_main.iloc[:, 4].values

# Log dimensions and vector count
logging.info(f"Feature space dimensions: {features.shape[1]}")
logging.info(f"Total vectors: {features.shape[0]}")

# Compute pseudo-inverse and log
rank = np.linalg.matrix_rank(features)
logging.info(f"Matrix Rank: {rank}")
pinv_matrix = np.linalg.pinv(features)
logging.info(f"Pseudo-Inverse:\n{pinv_matrix}")

# Estimate cost per product
estimated_cost = np.dot(pinv_matrix, target)
logging.info(f"Estimated cost per product:\n{estimated_cost}")

def compute_model_vector(pinv, target_vec):
    return np.dot(pinv, target_vec)

model_vector = compute_model_vector(pinv_matrix, target)
logging.info(f"Model vector:\n{model_vector}")

# Customer classification based on payment

df_main["Customer_Category"] = df_main["Payment (Rs)"].apply(lambda x: "Rich" if x > 200 else "Poor")
df_customers = df_main[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Customer_Category"]]
logging.info(f"\nCustomer Classification:\n{df_customers}")

# Load stock data
df_stock = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
prices = df_stock["Price"].values

# Compute statistics on stock prices
avg_price = stats.mean(prices)
var_price = stats.variance(prices)
logging.info(f"Average Price: {avg_price}")
logging.info(f"Price Variance: {var_price}")

# Price statistics by day/month
wed_prices = df_stock[df_stock["Day"] == "Wed"]["Price"].astype(float)
apr_prices = df_stock[df_stock["Month"] == "Apr"]["Price"].astype(float)
logging.info(f"Wednesday Mean Price: {stats.mean(wed_prices)}")
logging.info(f"April Mean Price: {stats.mean(apr_prices)}")

# Change% stats
wed_change = pd.to_numeric(df_stock[df_stock["Day"] == "Wed"]["Chg%"], errors="coerce")
profit_prob = (wed_change > 0).mean()
logging.info(f"Profit Probability on Wednesdays: {profit_prob}")

# Scatter plot of change% by day
day_to_num = {"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6,"Sun":7}
df_stock["Day_Num"] = df_stock["Day"].map(day_to_num)
plt.scatter(df_stock["Day_Num"], pd.to_numeric(df_stock["Chg%"], errors='coerce'))
plt.xlabel("Day of Week")
plt.ylabel("Change %")
plt.title("Stock Price Change by Day")
plt.show()

# Load thyroid dataset
thyroid_df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
thyroid_df.replace('?', np.nan, inplace=True)

# Columns to clean and normalize
num_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
thyroid_df[num_cols] = thyroid_df[num_cols].apply(pd.to_numeric, errors='coerce')

# Outlier detection

def detect_outliers(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)

outlier_cols = [col for col in num_cols if detect_outliers(thyroid_df, col).any()]

for col in num_cols:
    fill_val = thyroid_df[col].median() if col in outlier_cols else thyroid_df[col].mean()
    thyroid_df[col] = thyroid_df[col].fillna(fill_val)
    logging.info(f"Filled missing in '{col}' with {'median' if col in outlier_cols else 'mean'}")

# Handle categorical column missing values
cat_cols = thyroid_df.select_dtypes(include='object').columns
for col in cat_cols:
    thyroid_df[col] = thyroid_df[col].fillna(thyroid_df[col].mode()[0])

logging.info("Data cleaning complete. Saving file...")
thyroid_df.to_excel("Imputed_data.xlsx", index=False, engine='openpyxl')

# Normalize numeric columns
def min_max_norm(df, columns):
    for col in columns:
        min_val, max_val = df[col].min(), df[col].max()
        if min_val == max_val:
            logging.warning(f"'{col}' has constant value; skipping normalization.")
            continue
        df[col] = (df[col] - min_val) / (max_val - min_val)
        logging.info(f"Normalized '{col}'")

min_max_norm(thyroid_df, num_cols)

# Similarity Metrics

def smc_jaccard(vec1, vec2):
    m00 = np.sum((vec1 == 0) & (vec2 == 0))
    m11 = np.sum((vec1 == 1) & (vec2 == 1))
    m01 = np.sum((vec1 == 0) & (vec2 == 1))
    m10 = np.sum((vec1 == 1) & (vec2 == 0))
    smc = (m00 + m11) / (m00 + m01 + m10 + m11) if (m00 + m01 + m10 + m11) != 0 else 0
    jc = m11 / (m11 + m01 + m10) if (m11 + m01 + m10) != 0 else 0
    return smc, jc

def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 and norm2 else 0

bin_features = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']
thyroid_df[bin_features] = thyroid_df[bin_features].replace({'t': 1, 'f': 0})

vec_a, vec_b = thyroid_df.loc[0, bin_features].values, thyroid_df.loc[1, bin_features].values
smc_val, jc_val = smc_jaccard(vec_a, vec_b)
cos_sim = cosine_similarity(thyroid_df.loc[0, num_cols].values, thyroid_df.loc[1, num_cols].values)
logging.info(f"SMC: {smc_val}, Jaccard: {jc_val}, Cosine: {cos_sim}")

# Generate similarity matrices and heatmaps
def compute_similarity_matrices(data):
    n = data.shape[0]
    smc_matrix = np.identity(n)
    jc_matrix = np.identity(n)
    cos_matrix = np.identity(n)
    for i in range(n):
        for j in range(i+1, n):
            smc, jc = smc_jaccard(data[i], data[j])
            cos = cosine_similarity(data[i], data[j])
            smc_matrix[i, j] = smc_matrix[j, i] = smc
            jc_matrix[i, j] = jc_matrix[j, i] = jc
            cos_matrix[i, j] = cos_matrix[j, i] = cos
    return smc_matrix, jc_matrix, cos_matrix

def draw_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

subset_bin = thyroid_df.loc[:19, bin_features].values
smc_mtx, jc_mtx, cos_mtx = compute_similarity_matrices(subset_bin)
logging.info("Similarity matrices computed.")
draw_heatmap(smc_mtx, "Simple Matching Coefficient Heatmap")
draw_heatmap(jc_mtx, "Jaccard Coefficient Heatmap")
draw_heatmap(cos_mtx, "Cosine Similarity Heatmap")
