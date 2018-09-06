Predicting Online News Popularity with Spark
================

Using Apache Spark to predict Online News Popularity
----------------------------------------------------

We will try to predict Online News Popularity using Linear regression from **Apache Spark MLlib**

First we download the zip data file, unzip it and then move the csv file to the project directory ...

``` r
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip 

## or with 
curl -s 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip' -o OnlineNewsPopularity.zip

## then 
tar -xvjf OnlineNewsPopularity.zip 
# x OnlineNewsPopularity/
# x OnlineNewsPopularity/OnlineNewsPopularity.names
# x OnlineNewsPopularity/OnlineNewsPopularity.csv

mv OnlineNewsPopularity/OnlineNewsPopularity.csv project/path
cd project/path
```

Then we create a JAR package containing the application’s code, then use the spark-submit script to run our program ...

``` r
sbt package

## wait until it finishes, then ...
spark-submit \
--class onlineNewsPopularity \
--master local[2] \
target/scala-2.10/onlinenewspopularity_2.10-1.0.jar

## ******** Training Metrics ********
## Training Sum of Squared Error is: 24047.329603984064
## Training Root-Mean-Square-Error is: 0.8714524231877528
## ******** Test Metrics ********
## Test Sum of Squared Error is: 6162.951282751717
## Test Root-Mean-Square-Error is: 0.8802962993189899
```
