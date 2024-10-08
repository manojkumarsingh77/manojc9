import org.apache.spark.sql.SparkSession
import org.apache.hudi.DataSourceWriteOptions._
import org.apache.spark.sql.SaveMode

val spark = SparkSession.builder()
  .appName("HudiWriteExample")
  .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")  // KryoSerializer
  .config("spark.sql.hive.convertMetastoreParquet", "false")
  .getOrCreate()

val data = Seq(
  ("1", "John", "Doe"),
  ("2", "Jane", "Smith")
)

val df = spark.createDataFrame(data).toDF("id", "first_name", "last_name")

df.write.format("hudi")
  .option("hoodie.table.name", "test_hudi_table")  // Set the Hudi table name
  .option(TABLE_TYPE_OPT_KEY, "COPY_ON_WRITE")
  .option(PRECOMBINE_FIELD_OPT_KEY, "id")
  .option(RECORDKEY_FIELD_OPT_KEY, "id")
  .option(PARTITIONPATH_FIELD_OPT_KEY, "")
  .mode(SaveMode.Overwrite)
  .save("/tmp/hudi_test")
