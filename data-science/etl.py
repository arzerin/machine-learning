from pyspark.sql import SparkSession
from functools import reduce  # Import reduce
from pyspark.sql.functions import expr


# initialize SparkSession
spark = SparkSession.builder \
    .appName("My ETL Pipeline") \
    .getOrCreate()
    

# load the CSV file into a Spark DataFrame
file_path = "content/temperature.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# display the schema and preview the data
print("Schema")
df.printSchema() #insights into the schema, comparable to df.info() in Pandas
print("Schema")
df.show(5)    # preview the data, comparable to df.head() in Pandas

# Show the total number of rows in the DataFrame
total_rows = df.count()
print(f"Total number of rows: {total_rows}")

# Count rows with null values in the "ISO2" column before filling
missing_count = df.filter(df["ISO2"].isNull()).count()
print(f"Total rows with missing ISO2: {missing_count}")

# fill missing values for country codes
df = df.fillna({"ISO2": "Unknown"})


# drop rows where all temperature values are null
# Identify temperature columns
temperature_columns = [col for col in df.columns if col.startswith('F')]
df = df.dropna(subset=temperature_columns, how="all")

# reshape temperature data to have 'Year' and 'Temperature' columns
df_pivot = df.selectExpr(
    "ObjectId", "Country", "ISO2", "ISO3",
    "stack(62, " + 
    ",".join([f"'F{1961 + i}', F{1961 + i}" for i in range(62)]) +
    ") as (Year, Temperature)"
)

# convert 'Year' column to integer
df_pivot = df_pivot.withColumn("Year", expr("int(substring(Year, 2, 4))"))
df_pivot.show(5)

output_path = "content/generated/processed_temperature.parquet"
df_pivot.write.mode("overwrite").parquet(output_path)

# load the saved parquet file
processed_df = spark.read.parquet(output_path)
processed_df.show(5)


# Write the reshaped data to a CSV file
output_csv_path = "content/generated/temperature_updated.csv"
df_pivot.write.mode("overwrite").option("header", True).csv(output_csv_path)

print(f"Data saved as CSV to: {output_csv_path}")
