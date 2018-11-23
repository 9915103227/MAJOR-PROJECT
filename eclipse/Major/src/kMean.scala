import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j._


object kMean extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc=new SparkContext("local[*]","")
  val data=sc.textFile("/home/pranav/Desktop/MAJOR PROJECT/K_Means/cancerNew.csv")
  val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
  val iter=10000
  val noOfClusterIteration=199// check from no of cluster=0 to no of cluster=30;
  var wsse=new Array[Double](noOfClusterIteration) 
  for(i<-1 to noOfClusterIteration){
    val clusters = KMeans.train(parsedData, i, iter)
    val WSSSE = clusters.computeCost(parsedData)
    println(WSSSE)
    wsse(i-1)=WSSSE
  }
  val m=(wsse(noOfClusterIteration-1)-wsse(0))/(noOfClusterIteration-1);
  val c=wsse(0)-m*1;
  var maxDistPoint=0;
  var tmpDist=0.0;
  for(i<-1 to noOfClusterIteration){
    val dist=Math.abs(wsse(i-1)-m*i-c);
    if(dist>tmpDist){
      maxDistPoint=i;
      tmpDist=dist;
    }
  }
  print(maxDistPoint);
  /*var diffWsse=new Array[Double](9)
  for(i<-0 to 8){
    diffWsse(i)=Math.abs(wsse(i+1)-wsse(i))
    println("difference in wssse between "+(i+1)+" and "+(i+2)+" = "+diffWsse(i))
  }*/
}