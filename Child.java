import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class Child{
    double errorRt;
    String summary;

    public Child(Instances data, int folds, Random evalRandom){
        try{
            MultilayerPerceptron netwk = new MultilayerPerceptron();
            netwk.buildClassifier(data);

            Evaluation eval = new Evaluation(data);

            eval.crossValidateModel(netwk, data, folds, evalRandom);

            errorRt = eval.errorRate();
            summary = eval.toSummaryString();
        }catch(Exception e){
            System.out.println(e);
            return;
        }
        
    }

    public double getErrorRt(){
        return errorRt;
    }

    public String getSummary(){
        return summary;
    }
}