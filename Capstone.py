# 📌 Import Required Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, when, broadcast
from pyspark.sql.types import IntegerType, StringType, FloatType

# ✅ Step 1: Initialize Optimized Spark Session
spark = SparkSession.builder \
    .appName("RetailSupplyChainAnalysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.autoBroadcastJoinThreshold", "50MB") \
    .getOrCreate()

# ✅ Step 2: Load & Preprocess Data

# 2.1 Load Sales Data (Partitioned Parquet)
sales_data = spark.read.parquet("/home/labuser/Documents/CapstoneData/sales_data/")

# 2.2 Load Inventory Data (CSV)
inventory_data = spark.read.csv("/home/labuser/Documents/CapstoneData/inventory.csv", header=True, inferSchema=True)

# 2.3 Load Supplier Data (JSON)
supplier_data = spark.read.json("/home/labuser/Documents/CapstoneData/suppliers.json")

# ✅ Step 3: Handle Missing Data & Type Casting
sales_data = sales_data.na.fill({"SalesAmount": 0.0})
inventory_data = inventory_data.na.fill({"StockLevel": 0})
supplier_data = supplier_data.na.fill({"Performance": "Unknown", "AvgDeliveryTime": 0.0})

# Type Casting for Consistency
sales_data = sales_data.withColumn("SalesAmount", col("SalesAmount").cast(FloatType()))
inventory_data = inventory_data.withColumn("StockLevel", col("StockLevel").cast(IntegerType()))
supplier_data = supplier_data.withColumn("AvgDeliveryTime", col("AvgDeliveryTime").cast(FloatType()))

# ✅ Step 4: Add Calculated Columns
inventory_data = inventory_data.withColumn(
    "StockStatus",
    when(col("StockLevel") < 20, "Low Stock")
    .when(col("StockLevel") > 100, "Surplus Stock")
    .otherwise("Normal Stock")
)

# ✅ Step 5: Ensure `Category` is Added Correctly to Inventory Data
inventory_with_category = inventory_data.join(
    sales_data.select("ProductID", "Category").distinct(), 
    on="ProductID", 
    how="left"
)

# **Fix Missing Categories**
inventory_with_category = inventory_with_category.na.fill({"Category": "Unknown"})

# ✅ Step 6: Rename `Category` from Inventory Data to Avoid Ambiguity
inventory_with_category = inventory_with_category.withColumnRenamed("Category", "InventoryCategory")

# ✅ Step 7: Register DataFrames as SQL Tables
sales_data.createOrReplaceTempView("sales")
inventory_with_category.createOrReplaceTempView("inventory")
supplier_data.createOrReplaceTempView("suppliers")

# ✅ Step 8: SQL-Based Insights

# 8.1 Identify Sales Trends by Year & Category
sales_trends = spark.sql("""
    SELECT Year, Category, SUM(SalesAmount) AS TotalSales
    FROM sales
    GROUP BY Year, Category
    ORDER BY Year DESC, TotalSales DESC
""")
print("Sales Trends by Year & Category:")
sales_trends.show()

# 8.2 Best-Performing Suppliers (On-Time Delivery & Sales Contribution)
supplier_performance = spark.sql("""
    SELECT s.SupplierID, s.SupplierName, s.Performance, 
           AVG(s.AvgDeliveryTime) AS AvgDeliveryTime, 
           SUM(sales.SalesAmount) AS TotalSales
    FROM sales
    JOIN suppliers s ON sales.SupplierID = s.SupplierID
    GROUP BY s.SupplierID, s.SupplierName, s.Performance
    ORDER BY TotalSales DESC
""")
print("Best Performing Suppliers:")
supplier_performance.show()

# 8.3 Identify Low-Stock Products
low_stock_products = spark.sql("""
    SELECT ProductID, StockLevel, StockStatus 
    FROM inventory 
    WHERE StockLevel < 20
""")
print("Low Stock Products:")
low_stock_products.show()

# ✅ Step 9: Join & Aggregate Data

# Using **broadcast join** for small datasets (supplier_data)
supplier_broadcast = broadcast(supplier_data)

# Join Sales, Inventory, and Supplier Data
final_df = sales_data.join(inventory_with_category, "ProductID", "inner").join(supplier_broadcast, "SupplierID", "inner")

# ✅ Step 10: Explicitly Select Required Columns to Avoid Ambiguity
final_df = final_df.select(
    col("Year"),
    col("Category"),  # Category from Sales Data
    col("SalesAmount"),
    col("AvgDeliveryTime")
)

# ✅ Step 11: Aggregate Total Sales & Avg Delivery Time by Category & Year
aggregated_data = final_df.groupBy("Year", "Category").agg(
    sum("SalesAmount").alias("TotalSales"),
    avg("AvgDeliveryTime").alias("AvgDeliveryTime")
)

print("Aggregated Sales & Delivery Time:")
aggregated_data.show()

# ✅ Step 12: Optimized Processing
final_df.cache()  # Cache intermediate results for performance
sales_data = sales_data.repartition("Year").cache()  # Repartition sales by Year
inventory_with_category = inventory_with_category.repartition("InventoryCategory").cache()  # Fix: Use Renamed Category

# ✅ Step 13: Save Partitioned Output to Parquet
aggregated_data.write.mode("overwrite").partitionBy("Year", "Category").parquet("/home/labuser/Documents/CapstoneOutput/aggregated_sales")

supplier_performance.write.mode("overwrite").parquet("/home/labuser/Documents/CapstoneOutput/supplier_performance")

low_stock_products.write.mode("overwrite").parquet("/home/labuser/Documents/CapstoneOutput/low_stock_products")

print("✅ All Final Outputs Saved Successfully!")
