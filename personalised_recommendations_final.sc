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

// Import necessary functions
import org.apache.spark.sql.functions._

// Parse genres column in moviesDF to extract individual genres
val moviesWithGenresDF = moviesDF.withColumn("genresArray", split(col("genres"), "\\|"))
  .withColumn("genre", explode(col("genresArray")))
  .drop("genresArray")

// Show sample data with extracted genres
moviesWithGenresDF.show(5)


// Group tags by movieId to collect tag information for movies
val groupedTagsDF = tagsDF.groupBy("movieId")
  .agg(collect_list("tag").alias("tags"))

// Show sample data with grouped tags
groupedTagsDF.show(5)

// Import ALS and required libraries
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator



// Prepare ratings data for ALS model
val ratings = ratingsDF.select("userId", "movieId", "rating")
  .na.drop() // Remove rows with any NaN or null values in rating column

ratingsDF.describe().show()

// Split the data into training and test sets (80% training, 20% test)
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// Create the ALS model
val als = new ALS()
  .setMaxIter(10)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")

// Fit the ALS model to the training data
val model = als.fit(ratings)

// Evaluate the model by computing RMSE on the test data
val predictions = model.transform(test)
val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")

//val rmse = evaluator.evaluate(predictions)
//println(s"Root-mean-square error = $rmse")

val userId = 1234 // Simulated userID
val userGenre = "Comedy" // Simulated genre preference
val userTag = "thriller" // Simulated tag preference

// Filter movies based on the user's preferred genre
val genreFilteredMovies = moviesWithGenresDF.filter(col("genre") === userGenre)

// Get tags for movies
val userTagRelevance = genomeTagsDF.filter(col("tag") === userTag).select("tagId").first().getInt(0)
//erTagRelevance.show()
val tagRelevanceDF = genomeScoresDF.filter(col("tagId") === userTagRelevance)
//gRelevanceDF.show()

// Combine collaborative filtering predictions and tag relevance scores
val specificUserRecommendations = model.recommendForUserSubset(Seq(userId)).select("userId", "recommendations")
val userPredictions = specificUserRecommendations.select(col("userId"), explode(col("recommendations")))
  .select(col("userId"), col("col.movieId"), col("col.rating"))
  .alias("predictions")

// Assuming top 10 recommendations
//userPredictions.show()
val relevantMoviesDF = genreFilteredMovies.join(tagRelevanceDF, "movieId")
relevantMoviesDF.show()

import org.apache.spark.sql.functions.{col, expr}

val userTopRecommendations = relevantMoviesDF.alias("relevant")
  .join(userPredictions.select(col("userId"), explode(col("recommendations")))
    .select(col("userId"), col("col.movieId"), col("col.rating"))
    .alias("predictions"), expr("relevant.movieId = predictions.movieId") && col("r1.movieId") ===userId)
  .orderBy(col("predictions.rating").desc)
  .select(col("relevant.movieId"), col("relevant.title"), col("predictions.rating"), col("relevant.relevance"))
  .limit(50)

userTopRecommendations.show()
val distinctRecommendations = userTopRecommendations.dropDuplicates("movieId")

// Assuming distinctRecommendations is your existing DataFrame with columns: movieId, title, rating, relevance
// linkDF contains columns: movieId, ImdbID, TmdbID

val finalRecommendations = distinctRecommendations // Your existing DataFrame
  .join(linkDF, Seq("movieId"), "left") // Perform a left join based on movieId column
  .select("movieId", "title", "rating", "relevance", "ImdbID", "TmdbID")

finalRecommendations.show(10)
