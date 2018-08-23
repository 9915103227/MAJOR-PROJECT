import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j._


object kMean extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc=new SparkContext("local[*]","")
  val data=sc.textFile("/home/pranav/Desktop/MAJOR PROJECT/K_Means/samp*")
  val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
  val iter=300
  for(i<-1 to 10){
    val clusters = KMeans.train(parsedData, i, iter)
    val WSSSE = clusters.computeCost(parsedData)
    println(i+" "+WSSSE)
  }
}