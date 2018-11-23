import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j._


object tmp extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  def readCSV() : Array[Array[Double]] = {
  scala.io.Source.fromFile("/home/pranav/Desktop/MAJOR PROJECT/K_Means/mall.csv")
    .getLines()
    .map(_.split(",").map(_.trim.toDouble))
    .toArray
}
  val X=readCSV()
  val n=X.length
  val iter=15000
  val dim=X(0).length
  val minV = new Array[Double](dim)
  val maxV=new Array[Double](dim)
  for(i<-0 until dim){
    minV(i)=X(0)(i)
    maxV(i)=X(0)(i)
  }
  for(i<-1 until n){
    for(j<-0 until dim){
      if(minV(j)>X(i)(j)){
        minV(j)=X(i)(j)
      }
      if(maxV(j)<X(i)(j)){
        maxV(j)=X(i)(j)
      }
    }
  }
  val diff=new Array[Double](dim)
  for(i<-0 until dim){
    diff(i)=maxV(i)-minV(i)
  }
  var wsse=new Array[Double](n)
  for(i<-1 to n){
    val centroid=Array.ofDim[Double](n,dim)
    for(j<-0 until i){
      /*val alp=(((j*j*j*i)%n))
      centroid(j)(0)=X(alp)(0)
      centroid(j)(1)=X(alp)(0)
      centroid(j)(2)=X(alp)(0)*/
      for(k<-0 until dim){
        centroid(j)(k)=minV(k)+diff(k)/i*(j+1)
      }
    }
    val WSSSE=kmeans.kmeans(3, X, n, i, centroid, iter, i)
    wsse(i-1)=WSSSE
  }
  

  val m=(wsse(n-1)-wsse(1))/(n-2);
  val c=wsse(1)-m*2;
  var maxDistPoint=0;
  var tmpDist=0.0;
  for(i<-2 to n){
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






