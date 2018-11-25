import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

// javac -cp '.:weka.jar' Main.java
// java -cp '.:weka.jar' Main

// weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a

public class Main{

  public static final int populationSize = 10;

  public static int folds = 5;
  
  public static int minLayers = 1;
  public static int maxLayers = 3;
  
  public static int minNeurons = 10;
  public static int maxNeurons = 16;
  
  public static int minEpochs = 250;
  public static int maxEpochs = 750;
  
  public static double minLearningRate = 0.2;
  public static double maxLearningRate = 0.5;
  
  public static double minMomentum = 0.1;
  public static double maxMomentum = 0.3;

  public static final Random evalRandom = new Random();

  public static Instances data;

  public static int randomInt(int min, int max){
    Random r = new Random();
    return r.nextInt(max - min) + min;
  }

  public static double randomDouble(double min, double max){
    Random r = new Random();
    return ((max - min) * r.nextDouble() + min);
  }

  public static String buildHiddenLayers(int layers){
    StringBuilder sb = new StringBuilder();
    for(int i = 0; i < (layers - 1); i++){
      sb.append(randomInt(minNeurons, maxNeurons));
      sb.append(",");
    }
    sb.append(randomInt(minNeurons, maxNeurons));
    return sb.toString();
  }

  public static Child randomChild(){
    String hiddenLayers = buildHiddenLayers(randomInt(minLayers, maxLayers));
    int epochs = randomInt(minEpochs, maxEpochs);
    double learningRate = randomDouble(minLearningRate, maxLearningRate);
    double momentum = randomDouble(minMomentum, maxMomentum);

    return(new Child(data, folds, evalRandom, hiddenLayers, epochs, learningRate, momentum));
  }

  public static ArrayList<Child> firstGen(){
    ArrayList<Child> children = new ArrayList<Child>();
    for(int i = 0; i < populationSize; i++){
      children.add(randomChild());
    }
    return children; 
  }
  
  public static ArrayList<Child> createChildren(){
    ArrayList<Child> children = new ArrayList<Child>();
    return children; 
  }

  public static void main(String []args){
    long startTime = System.nanoTime();
    ArrayList<Child> population = new ArrayList<Child>();
    try{
      DataSource source = new DataSource("./out.csv");
      data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);
      population = firstGen();
    }catch(Exception e){
      System.out.println(e);
      return;
    }
    for(int i = 0; i < population.size(); i++){
      System.out.println(population.get(i).getSummary());
      System.out.println();
    }

    long endTime   = System.nanoTime();
    long totalTime = endTime - startTime;
    System.out.println("Segundos transcurridos: " + TimeUnit.NANOSECONDS.toSeconds(totalTime));
  }
}