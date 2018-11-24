import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

// javac -cp '.:weka.jar' Main.java
// java -cp '.:weka.jar' Main

// weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a

public class Main{
  public static void main(String []args){
      try{
        DataSource source = new DataSource("./out.csv");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        Child myChild = new Child(data, 5, new Random());
        System.out.println(myChild.getErrorRt());
        System.out.println(myChild.getSummary());
      }catch(Exception e){
        System.out.println(e);
        return;
      }
  }
}