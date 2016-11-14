package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}


object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    val df = spark.read.parquet("/home/jean-michel/Downloads/cleanedDataFrame.parquet")

    // on enlève provisoirement les colonnes koi_disposition et rowid qui ne sont pas des variables d'input
    val df_features = df.drop("koi_disposition", "rowid")

    // on crée un vecteur des variables
    val assembler = new VectorAssembler()
      .setInputCols(df_features.columns)
      .setOutputCol("features")

    // on ajoute une colonne features au df initial avec la fonction assembler définie précédemment
    val df_transformed = assembler.transform(df)

    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")

    // on ajoute une colonne label codant en 0 et 1 la colonne "koi_disposition"
    val df_final = indexer.fit(df_transformed).transform(df_transformed)

    // on sépare en deux le jeu de données pour l'entraînement et le test du modèle
    // sachant que le jeu de training va être à nouveau scindé en deux, on ne conserve qu'une petite proportion de données de test
    val Array(trainingData, testData) = df_final.randomSplit(Array(0.9, 0.1))

    // on crée le modèle de régression logistique
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      //.setRegParam(0.3)  // on commente RegParam qui va faire l'objet des tests des valeurs de la grid
      .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setStandardization(true)  // on normalise les données
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(5.0e-5)  // stop criterion of the algorithm based on its convergence
      // setTol abaissé pour éviter l'absence de convergence (cf.warning ci-dessous), cela réduit la performance de 1% environ
      // 16/11/12 21:03:42 WARN LogisticRegression: LogisticRegression training finished but the result is not converged because: max iterations reached
      .setMaxIter(600)  // a security stop criterion to avoid infinite loops

    // définition de la grille des hyper-paramètres
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.000001,0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0))
      .build()

    // on sépare les données de training en jeu d'entraînement et de validation (70% pour l'entraînement)
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)   // le modèle à évaluer
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)  // la grille d'hyper-paramètres appliqués au modèle
      .setTrainRatio(0.7)

    // Le nombre de données est important, on peut se contenter d'un trainValidationSplit pour sélectionner les hper-paramètres
    val model = trainValidationSplit.fit(trainingData)
    //println("model.explainParams = " + model.explainParams(), "model.bestModel = " + model.bestModel, "model.validationMetrics = " + model.validationMetrics)

    // obtention des prédictions réalisées par le meilleur modèle sélectionné
    val predictions = model.transform(testData)

    predictions.select("probability", "label", "prediction").show(50)

    // création d'un évaluateur pour mesurer l'erreur de prédiction
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")

    val ROC = evaluator.evaluate(predictions)  // evaluate doit être appliqué à  un DataSet
    println("ROC sur les données de test = " + ROC)

    


  }

}