import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

object onlineNewsPopularity extends App {

	val conf = new SparkConf().setAppName("onlineNewsPopularity").setMaster("local[2]")
  	val sc = new SparkContext(conf)
  	val sqlContext = new SQLContext(sc)

  	// reading the file excluding the header and the first two non-predictive columns
  	val data = sc.textFile("OnlineNewsPopularity.csv").
  	filter(line => !line.contains("url")).
  	map(line => line.split(","))
	val dataPoints = data.map(row => new LabeledPoint(row.last.toDouble, 
  		Vectors.dense(row.slice(2, row.length-1).map(str => str.toDouble)))).cache()

	// split the data into training and test
  	val splits = dataPoints.randomSplit(Array(0.8, 0.2))
  	val training = splits(0)
  	val test = splits(1)

  	// scale the data 
  	val scaler = new StandardScaler(withMean = true, withStd = true).
  	fit(training.map(p => p.features))
  	val scaledTraining = training.map(p => new LabeledPoint(p.label, scaler.transform(p.features))).cache()
  	val scaledTest = test.map(p => new LabeledPoint(p.label, scaler.transform(p.features))).cache()

  	// the model
  	val regModel = new LinearRegressionWithSGD().setIntercept(true)
	regModel.optimizer.setNumIterations(1000).setStepSize(1)
	val model = regModel.run(scaledTraining)

	// evaluating on training set
	val trPredictions:RDD[Double] = model.predict(scaledTraining.map(obs => obs.features))
	val trActual:RDD[Double] = scaledTraining.map(_.label)
	val TRpredictedVSactual: RDD[(Double, Double)] = trPredictions.zip(trActual)
	val trSSE = TRpredictedVSactual.map{case (predicted, actual) =>
 		math.pow(actual - predicted, 2)
	}.sum()
	val trMSE = trSSE / scaledTraining.count
	val trRMSE = math.sqrt(trMSE)
	println(s"******** Training Metrics ********")
	println(s"Training Sum of Squared Error is: $trSSE")
	println(s"Training Root-Mean-Square-Error is: $trRMSE")

	// evaluating on test set
	val predictions:RDD[Double] = model.predict(scaledTest.map(obs => obs.features))
	val actual:RDD[Double] = scaledTest.map(_.label)
	val predictedVSactual: RDD[(Double, Double)] = predictions.zip(actual)
	val SSE = predictedVSactual.map{case (predicted, actual) =>
 		math.pow(actual - predicted, 2)
	}.sum()
	val MSE = SSE / scaledTest.count
	val RMSE = math.sqrt(MSE)
	println(s"******** Test Metrics ********")
	println(s"Test Sum of Squared Error is: $SSE")
	println(s"Test Root-Mean-Square-Error is: $RMSE")
}