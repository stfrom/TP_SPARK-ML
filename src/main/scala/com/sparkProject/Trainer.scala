package com.sparkProject


import breeze.numerics.pow
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IDF, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
//import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.CountVectorizer
//import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   val filepath = "/Users/charlesrohmer/MasterBigData/INF729-Spark_Hadoop/SPARK_TP/TP_ParisTech_2017_2018_starter/data/"
   val filename = filepath + "prepared_trainingset"
   var df=spark.read.format("parquet").load(filename)

    /**df.show(10) **/
  df.show(10)

    /** TF-IDF **/

    val tokenizerPL =  new   RegexTokenizer()
      .setPattern( "\\W+")
      .setGaps(true)
      .setInputCol( "text" )

    val removerPL = new StopWordsRemover()
      .setInputCol(tokenizerPL.getOutputCol)
      .setOutputCol("filtered")


    val countVectorPL = new CountVectorizer()
      .setInputCol(removerPL.getOutputCol)

    val idfPL = new IDF().setInputCol(countVectorPL.getOutputCol).setOutputCol("tfidf")


    val countryindexerPL = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    val currencyindexerPL = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")


    /** VECTOR ASSEMBLER **/

    val assemblerPL = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign","hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    /** MODEL **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol( "raw_predictions" )
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)



    /** PIPELINE **/
    // Configure an ML pipeline,

    val pipeline = new Pipeline()
      .setStages(Array(tokenizerPL, removerPL,countVectorPL,idfPL, countryindexerPL,currencyindexerPL,assemblerPL, lr))


    /** TRAINING AND GRID-SEARCH **/


    // Split the data into training and test sets (30% held out for testing).
    val Array(training, test) = df.randomSplit(Array(0.1, 0.9))

    // definission des valeurs pour le parametre de la LR:
    val regParamArray = Array.iterate[Double](math.pow(10,-8),4)( x => x * math.pow(10,2))
    // .addGrid(lr.regParam, Array(0.00000001, 0.000001, 0.0001, 0.01))

    // definission des valeurs pour le parametre de countVector:
    val minDFArray = (55.0 to 95.0 by 20).toArray
    // .addGrid(countVectorPL.minDF, Array(55.0, 75.0, 95.0))


    val paramGrid = new ParamGridBuilder()
      .addGrid(countVectorPL.minDF, minDFArray)
      .addGrid(lr.regParam, regParamArray)
      .build()

    // il s'agit d'un classificateur binaire, je choisis l'evaluateur BinaryClass,

    val regEval = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setPredictionCol("predictions")
      .setLabelCol("final_status")




    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(regEval)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)  // on split la base en 3 66% training, 33% pour le test



    val cvModel = cv.fit(training)

    val df_WithPredictions = cvModel.transform(test)

    val metrics = regEval.evaluate(df_WithPredictions)

    df_WithPredictions.groupBy("final_status","predictions").count.show()

    print (metrics)

    // enregistreement du modele selectionné
    cvModel.write.overwrite().save("/Users/charlesrohmer/MasterBigData/INF729-Spark_Hadoop/SPARK_TP/TP_ParisTech_2017_2018_starter/spark-logistic-regression-model2")

    /** And load it back in during production
    val sameModel = PipelineModel.load(path "/spark-logistic-regression-model")

    **/


  }
}
