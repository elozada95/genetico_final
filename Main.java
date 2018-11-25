import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;

// javac -cp '.:weka.jar' Main.java
// java -cp '.:weka.jar' Main

// weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a

public class Main{

  public static final int populationSize = 20;

  public static final int generations = 30;

  public static final int mutationRate = 5; //[0% - 100%]

  public static final int folds = 5;
  
  public static final int minLayers = 1;
  public static final int maxLayers = 2;
  
  public static final int minNeurons = 10;
  public static final int maxNeurons = 16;
  
  public static final int minEpochs = 250;
  public static final int maxEpochs = 750;
  
  public static final double minLearningRate = 0.2;
  public static final double maxLearningRate = 0.5;
  
  public static final double minMomentum = 0.1;
  public static final double maxMomentum = 0.3;

  public static final Random evalRandom = new Random();

  public static Instances data;

  public static double [] matingProbability;

  public static int randomInt(int min, int max){
    Random r = new Random();
    return r.nextInt(max - min) + min;
  }

  public static double randomDouble(double min, double max){
    Random r = new Random();
    return ((max - min) * r.nextDouble() + min);
  }

  public static void getMatingProbability(){
    matingProbability = new double [populationSize];
    int chances = (populationSize * (populationSize + 1)) / 2;
    for(int i = 0; i < populationSize; i++){
      matingProbability[i] = (double)(i+1)/(double)chances;
      if(i > 0){
        matingProbability[i] += matingProbability[i-1];
      }
    }
  }

  public static int getMate(){
    Random r = new Random();
    double rand = r.nextDouble();
    for(int i = 0; i < populationSize-1; i++){
      if(rand < matingProbability[i]){
        return (populationSize - i - 1);
      }
    }
    return 0;
  }

  public static boolean probabilityRoll(int chance){
    int roll = randomInt(0, 99);
    if(roll < chance){
      return true;
    }
    return false;
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

  public static Child mate(Child father, Child mother){
    String hiddenLayers = father.getH();
    int epochs = father.getN();
    double learningRate = mother.getL();
    double momentum = mother.getM();

    if(probabilityRoll(mutationRate)){
      int gene = randomInt(0, 3);
      switch(gene){
        case 0:
          hiddenLayers = buildHiddenLayers(randomInt(minLayers, maxLayers));
          break;
        case 1:
          epochs = randomInt(minEpochs, maxEpochs);
          break;
        case 2:
          learningRate = randomDouble(minLearningRate, maxLearningRate);
          break;
        case 3:
          momentum = randomDouble(minMomentum, maxMomentum);
          break;
        default:
          break;
      }
    }

    return(new Child(data, folds, evalRandom, hiddenLayers, epochs, learningRate, momentum));
  }

  public static void insertionSort(ArrayList<Child> list, Child newChild){
    for(int i = 0; i < list.size(); i++){
      if(list.get(i).getErrorRt() > newChild.getErrorRt()){
        list.add(i, newChild);
        return;
      }
    }
    list.add(newChild);
  }

  public static ArrayList<Child> firstGen(){
    ArrayList<Child> children = new ArrayList<Child>();
    for(int i = 0; i < populationSize; i++){
      insertionSort(children, randomChild());
    }
    return children; 
  }

  public static ArrayList<Child> nextGen(ArrayList<Child> curGen){
    ArrayList<Child> children = new ArrayList<Child>();
    HashSet<ArrayList<Integer>> pairings = new HashSet<ArrayList<Integer>>();
    while(pairings.size() < populationSize - 1){
      ArrayList<Integer> pair = new ArrayList<Integer>();
      int a = getMate();
      int a2 = getMate();

      if(a2 < a){
        a = a2;
      }
      int b;
      do{
        b = getMate();
        int b2 = getMate();
        if(b2 < b){
          b = b2;
        }
      }while(a == b);
      pair.add(a);
      pair.add(b);
      if(!pairings.contains(pair)){
        pairings.add(pair);
      }
    }
    insertionSort(children, curGen.get(0));
    for (ArrayList<Integer> p : pairings) {
      insertionSort(children, mate(curGen.get(p.get(0)), curGen.get(p.get(1))));
    }
    return children;
  }
  
  public static ArrayList<Child> createChildren(){
    ArrayList<Child> children = new ArrayList<Child>();
    return children; 
  }

  public static void printGenerationSummary(ArrayList<Child> gen, int genNum){
    double avgError = 0;
    for(Child curChild : gen){
      avgError += curChild.getErrorRt();
    }
    avgError /= gen.size();

    double bestErrorRt = gen.get(0).getErrorRt();

    System.out.println();
    System.out.println("Generation " + genNum);
    System.out.println("Generation's average accuracy: " + ((1.0 - avgError)*100.0) + "%");
    System.out.println("Best candidate's accuracy: " + ((1.0 - bestErrorRt)*100.0) + "%");
    System.out.println();
  }

  public static void printChild(Child curChild){
    System.out.println("Hidden layers: " + curChild.getH());
    System.out.println("Epochs: " + curChild.getN());
    System.out.println("Learning rate: " + curChild.getL());
    System.out.println("Momentum: " + curChild.getM());
    System.out.println();
    System.out.println(curChild.getSummary());
    System.out.println(curChild.getMatrix());
  }

  public static void main(String []args){
    long startTime = System.nanoTime();
    getMatingProbability();
    ArrayList<Child> prevGeneration = new ArrayList<Child>();
    ArrayList<Child> curGeneration = new ArrayList<Child>();
    try{
      DataSource source = new DataSource("./input.csv");
      data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);
      curGeneration = firstGen();
      printGenerationSummary(curGeneration, 1);
      for(int i = 1; i < generations; i++){
        prevGeneration = new ArrayList<Child>(curGeneration);
        curGeneration = nextGen(prevGeneration);
        printGenerationSummary(curGeneration, i+1);
      }
      System.out.println("Best candidate found:");
      printChild(curGeneration.get(0));
    }catch(Exception e){
      System.out.println(e);
      return;
    }

    long endTime   = System.nanoTime();
    long totalTime = endTime - startTime;
    System.out.println("Seconds elapsed: " + TimeUnit.NANOSECONDS.toSeconds(totalTime));
  }
}