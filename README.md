# Movie-Recommendation-System
Developed a movie recommendation system that predicts user ratings and provides personalized movie suggestions using machine learning and filtering techniques.

## Approach
Connect Spark with MySQL, leveraging a JDBC (Java Database Connectivity) connector. This connector acts as a bridge between Spark and the MySQL database, enabling efficient communication and data transfer.
Implementation Steps - Installing the JDBC driver for MySQL in the Spark environment then Configuring the Spark session to use the JDBC driver and Establishing a connection to the MySQL database using appropriate connection parameters. Used Spark SQL or DataFrame API to execute queries and retrieve data from MySQL into Spark.

* This project showcased the power of data analytics in predicting movie success and audience preferences.
* The personalized recommendation system, driven by machine learning, promises a more engaging movie experience by tailoring suggestions to individual tastes.
* By merging data analysis, predictive modeling, and personalized recommendations, this project highlights a path toward a more tailored and immersive movie-watching journey.
* Embracing these insights and personalization techniques has the potential to reshape how audiences interact with movies, offering a more satisfying and captivating cinematic experience.

### RECOMENDATION MODEL (Collaborative Filtering with ALS)

* Collaborative Filtering, a key element in recommendation systems, was implemented using ALS (Alternating Least Squares). ALS, an iterative optimization algorithm, helps uncover latent factors in user-item interactions. By identifying patterns in user preferences and behaviors, ALS aids in generating accurate recommendations.
* In addition to Collaborative Filtering, *Content-Based Filtering* enhances the recommendation engine with added sophistication. This method involves examining inherent movie features like genres and tags to comprehend their characteristics. By matching user preferences with these intrinsic features, Content-Based Filtering improves the recommendation process.
* The combination of Collaborative and Content-Based Filtering strengthens the recommendation model. By merging patterns of user-item interactions with movie characteristics, the model gains a nuanced understanding of user preferences. This integration is crucial for providing diverse and accurate movie suggestions.

## Sample User input
![image](https://github.com/user-attachments/assets/efee9d6d-36ac-411a-bdda-863dfbfcd265)

## Sample recommendation
![image](https://github.com/user-attachments/assets/903f6be2-fb7a-4eeb-bf02-b83b9d70206d)
