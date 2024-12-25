import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._

// Create SparkSession
val spark = SparkSession.builder()
  .appName("MovieAnalysis")
  .master("local")
  .getOrCreate()

// Define MySQL connection properties
val jdbcHostname = "localhost"
val jdbcPort = "3306"
val jdbcDatabase = "moviedb"
val jdbcUsername = "root"
val jdbcPassword = "password"

// Set up the JDBC URL for MySQL
val jdbcUrl = s"jdbc:mysql://${jdbcHostname}:${jdbcPort}/${jdbcDatabase}"

// Read data from MySQL into a DataFrame
val moviesDF= spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "movies")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()

val tagsDF= spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "tags")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()
val genomeTagsDF= spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "genometags")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()
val ratingsDF= spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "ratings")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()

val moviesDF = spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "movies")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()


val genomeScoresDF = spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "genomeScores")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()

val linkDF = spark.read
  .format("jdbc")
  .option("url", jdbcUrl)
  .option("dbtable", "link")
  .option("user", jdbcUsername)
  .option("password", jdbcPassword)
  .load()

//Updated
val joinedDF = ratingsDF.join(moviesDF, Seq("movieId"), "inner")
val selectedData = joinedDF.select("userId", "movieId", "genres", "timestamp", "rating")

// Feature Encoding for Categorical Features (like genres)
val indexer = new StringIndexer().setInputCol("genres").setOutputCol("genreIndex")
val indexed = indexer.fit(selectedData).transform(selectedData)
val encoder = new OneHotEncoder().setInputCol("genreIndex").setOutputCol("genreVec").fit(indexed)
val encoded = encoder.transform(indexed)

// Assemble Features into a Single Vector (including categorical and numerical features)
val assembler = new VectorAssembler().setInputCols(Array("userId", "movieId", "genreVec", "timestamp")).setOutputCol("features")

val assembledData = assembler.transform(encoded).select("rating", "features")

val Array(training, test) = assembledData.randomSplit(Array(0.8, 0.2))

// Linear Regression
val lr = new LinearRegression().setLabelCol("rating").setFeaturesCol("features")
val lrModel = lr.fit(training)

// Make predictions on the test set
val predictions_lr = lrModel.transform(test)
predictions_lr.show()


//Evaluation

//RMSE
predictions_lr.select(sqrt(avg((col("rating") - col("prediction")) * (col("rating") - col("prediction"))))).as("rmse").show()

//MAE
predictions_lr.select(avg(abs(col("rating") - col("prediction")))).as("mae").show()

//R-squared error
predictions_lr.select(
  corr(col("prediction"), col("rating")) * corr(col("prediction"), col("rating"))
).as("r2").show()

//Accuracy
val threshold = 0.5
val accuracy = predictions_lr.select((sum(when(abs(col("rating") - col("prediction")) <= threshold, 1).otherwise(0)) / count(col("rating"))).as("accuracy")).show()


// Random Forest
val rfr = new RandomForestRegressor().setLabelCol("rating").setFeaturesCol("features")
val rfrModel = rfr.fit(training)

// Make predictions on the test set
val predictions_rfr = rfrModel.transform(test)
predictions_rfr.show()


//Evaluation

//RMSE
predictions_rfr.select(sqrt(avg((col("rating") - col("prediction")) * (col("rating") - col("prediction"))))).as("rmse").show()

//MAE
predictions_rfr.select(avg(abs(col("rating") - col("prediction")))).as("mae").show()

//R-squared error
predictions_rfr.select(
  corr(col("prediction"), col("rating")) * corr(col("prediction"), col("rating"))
).as("r2").show()

//Accuracy
val threshold = 0.5
val accuracy = predictions_rfr.select((sum(when(abs(col("rating") - col("prediction")) <= threshold, 1).otherwise(0)) / count(col("rating"))).as("accuracy")).show()


//Gradient Boosting Regressor

val gbr = new GBTRegressor().setLabelCol("rating").setFeaturesCol("features")
val gbtModel = rfr.fit(training)

// Make predictions on the test set
val predictions_gbt = gbtModel.transform(test)
predictions_gbt.show()


//Evaluation

//RMSE
predictions_gbt.select(sqrt(avg((col("rating") - col("prediction")) * (col("rating") - col("prediction"))))).as("rmse").show()

//MAE
predictions_gbt.select(avg(abs(col("rating") - col("prediction")))).as("mae").show()

//R-squared error
predictions_gbt.select(
  corr(col("prediction"), col("rating")) * corr(col("prediction"), col("rating"))
).as("r2").show()

//Accuracy
val threshold = 0.5
val accuracy = predictions_gbt.select((sum(when(abs(col("rating") - col("prediction")) <= threshold, 1).otherwise(0)) / count(col("rating"))).as("accuracy")).show()