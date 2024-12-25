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

//Schema Printing
println("moviesDF")
moviesDF.printSchema()
println("tagsDF")
tagsDF.printSchema()
println("genomeTagsDF")
genomeTagsDF.printSchema()
println("genomeScoresDF")
genomeScoresDF.printSchema()
println("ratingsDF")
ratingsDF.printSchema()
println("linkDF")
linkDF.printSchema()

//Descriptive Analysis
//1. Ratings Analysis
// Group by 'rating' column and count occurrences
val ratingCounts = ratingsDF
  .groupBy("rating")
  .agg(count("rating").alias("count"))
  .sort("rating")

ratingCounts.show()



//2. Movies with highest ratings
// Count the number of ratings for each movie
val moviePopularityByRatings = ratingsDF
  .groupBy("movieId")
  .agg(count("rating").alias("numRatings"))
  .sort(desc("numRatings"))

//moviePopularityByRatings.show()

// Join with moviesDF to get movie details
val popularMoviesWithDetails = moviePopularityByRatings
  .join(moviesDF, Seq("movieId"))
  .select("movieId", "title", "numRatings","genres")

val top20PopularMovies = popularMoviesWithDetails
  .sort(desc("numRatings"))
  .limit(20)

top20PopularMovies.show()

//3. User activity

// Count the number of ratings given by each user
val userActivity = ratingsDF
  .groupBy("userId")
  .agg(count("rating").alias("numRatings"))
  .sort(desc("numRatings"))

// Show the user activity
userActivity.show()

//Filtering Users based on activity threshold
val threshold = 100

val activeUsers = userActivity.filter(s"numRatings > $threshold")
activeUsers.show()


//Distribution of User ratings

// Calculate summary statistics of user ratings
val ratingStats = userActivity
  .select("numRatings")
  .summary("min", "25%", "50%", "75%", "max")

ratingStats.show()


//Genre Based Analysis

// Split genres into separate rows
val moviesWithGenres = moviesDF
  .withColumn("genre", explode(split(col("genres"), "\\|")))

val ratingsWithGenres = ratingsDF
  .join(moviesWithGenres, Seq("movieId"), "inner")
  .select("userId", "rating", "genre")

// Show the resulting DataFrame
ratingsWithGenres.show()

//4. Average Rating by Genre
val avgRatingByGenre = ratingsWithGenres
  .groupBy("genre")
  .agg(avg("rating").alias("avgRating"))
  .sort(desc("avgRating"))

avgRatingByGenre.show()


//5. Number of Ratings per Genre
val ratingsCountByGenre = ratingsWithGenres
  .groupBy("genre")
  .agg(count("rating").alias("numRatings"))
  .sort(desc("numRatings"))

ratingsCountByGenre.show()

//6. Genre Trends over Time
val moviesWithGenres = moviesDF
  .withColumn("genre", explode(split(col("genres"), "\\|")))

moviesWithGenres.show()

val ratingsWithGenres = ratingsDF
  .join(moviesWithGenres, Seq("movieId"), "inner")
  .select("userId", "rating", "genre", "timestamp")

val ratingsWithYear = ratingsWithGenres
  .withColumn("year", year(from_unixtime(col("timestamp"))))

val genreTrends = ratingsWithYear
  .groupBy("genre", "year")
  .agg(count("rating").alias("numRatings"))
  .sort("genre", "year")



val windowSpec = Window.partitionBy("year").orderBy(desc("numRatings"))

val rankedGenres = genreTrends.withColumn("rank", row_number().over(windowSpec))

val topGenreByYear = rankedGenres.filter(col("rank") === 1).select("year", "genre", "numRatings")

topGenreByYear.show()


//7. Temporal trends

val ratingsWithTime = ratingsDF
  .withColumn("year", year(from_unixtime(col("timestamp"))))

val avgRatingsOverTime = ratingsWithTime
  .groupBy("year")
  .agg(avg("rating").alias("avgRating"))
  .sort("year")

avgRatingsOverTime.show()

val ratingCountsOverTime = ratingsWithTime
  .groupBy("year")
  .agg(count("rating").alias("numRatings"))
  .sort("year")

ratingCountsOverTime.show()

//8. Association Mining


val movieBaskets = ratingsDF
  .groupBy("userId")
  .agg(collect_set("movieId").alias("ratedMovies"))

movieBaskets.show()

import org.apache.spark.ml.fpm.{FPGrowth, FPGrowthModel}

val fpGrowth = new FPGrowth()
  .setItemsCol("ratedMovies")
  .setMinSupport(0.1)
  .setMinConfidence(0.5)

val model: FPGrowthModel = fpGrowth.fit(movieBaskets)

val frequentItemsets = model.freqItemsets

frequentItemsets.show()


// Group tags by movieId to collect all tags associated with each movie
val movieTags = tagsDF
  .groupBy("movieId")
  .agg(collect_set("tag").alias("tags"))

// Show movie tags
movieTags.show()

//9. Tag Based Analysis
// Explode the tags column to have one row per tag
val explodedTags = movieTags.withColumn("tag", explode(col("tags")))

// Count occurrences of each tag
val tagCounts = explodedTags
  .groupBy("tag")
  .agg(count("tag").alias("tagCount"))
  .sort(desc("tagCount"))

// Show tag frequencies
tagCounts.show()



// Co-occurrence matrix
// Self-join on userId to find pairs of movies rated by the same user
val moviePairs = ratingsDF.alias("r1")
  .join(ratingsDF.alias("r2"), col("r1.userId") === col("r2.userId") && col("r1.movieId") < col("r2.movieId"))
  .select(col("r1.movieId").alias("movie1"), col("r2.movieId").alias("movie2"))

val movieConnections = moviePairs
  .groupBy("movie1", "movie2")
  .agg(count("*").alias("coOccurrences"))
  .orderBy(desc("coOccurrences"))

movieConnections.show()
