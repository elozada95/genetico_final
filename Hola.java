import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

// sudo javac -cp '.:weka.jar' Hola.java
// sudo java -cp '.:weka.jar' Hola

// weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a

public class Hola{
  public static void main(String []args){
    try{
      DataSource source = new DataSource("./out.csv");
      Instances data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);

      MultilayerPerceptron net = new MultilayerPerceptron();
      net.buildClassifier(data);

      Evaluation eval = new Evaluation(data);

      //eval.evaluateModel(net, data);

      eval.crossValidateModel(net, data, 10, new Random());

      double errorz = eval.errorRate();

      System.out.println(net.getValidationSetSize());
      System.out.println(errorz);
      System.out.println(eval.toSummaryString());
      //System.out.println(eval.toMatrixString());
      //System.out.println(net.toString());

    }catch(Exception e){
      System.out.println(e);
      return;
    }
    
  }

  public static void printOptions(MultilayerPerceptron net){
    String[] arrs = net.getOptions();
    for(int i = 0; i < arrs.length; i++){
      System.out.println(i+":"+arrs[i]);

    }
  }
}