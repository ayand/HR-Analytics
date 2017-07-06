import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//import org.apache.spark.ml.classification.DecisionTreeClassifier

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("HR_comma_sep.csv")

val hrDataAll = data.select(data("left").as("label"), $"satisfaction_level", $"last_evaluation", $"number_project", $"average_montly_hours", $"time_spend_company", $"Work_accident", $"promotion_last_5years", $"sales", $"salary")

val salesIndexer = new StringIndexer().setInputCol("sales").setOutputCol("SalesIndex")
val salesEncoder = new OneHotEncoder().setInputCol("SalesIndex").setOutputCol("SalesVec")

val salaryIndexer = new StringIndexer().setInputCol("salary").setOutputCol("SalaryIndex")
val salaryEncoder = new OneHotEncoder().setInputCol("SalaryIndex").setOutputCol("SalaryVec")

val assembler = new VectorAssembler().setInputCols(Array("SalaryVec", "SalesVec", "satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years")).setOutputCol("features")

val Array(training, test) = hrDataAll.randomSplit(Array(0.7, 0.3), seed = 12345)

val lr = new LogisticRegression().setMaxIter(30).setRegParam(0.1).setElasticNetParam(0.2)

val pipeline = new Pipeline().setStages(Array(salesIndexer, salesEncoder, salaryIndexer, salaryEncoder, assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiate metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
//println(metrics.precision)

model.save("LogisticRegression")
