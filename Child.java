import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class Child{
    
    MultilayerPerceptron netwk = new MultilayerPerceptron();
    
    double errorRt;
    String summary;
    String confusionMatrix;


    public Child(Instances data, int folds, Random evalRandom, String hiddenLayers, int epochs, double learningRate, double momentum){
        
        netwk.setHiddenLayers(hiddenLayers);
        netwk.setTrainingTime(epochs);
        netwk.setLearningRate(learningRate);
        netwk.setMomentum(momentum);
            
        try{
            netwk.buildClassifier(data);

            Evaluation eval = new Evaluation(data);

            eval.crossValidateModel(netwk, data, folds, evalRandom);

            errorRt = eval.errorRate();
            summary = eval.toSummaryString();
            confusionMatrix = eval.toMatrixString();
        }catch(Exception e){
            System.out.println(e);
            return;
        }
        
    }

    public String getH(){
        return netwk.getHiddenLayers();
    }

    public int getN(){
        return netwk.getTrainingTime();
    }

    public double getL(){
        return netwk.getLearningRate();
    }

    public double getM(){
        return netwk.getMomentum();
    }

    public double getErrorRt(){
        return errorRt;
    }

    public String getSummary(){
        return summary;
    }

    public String getMatrix(){
        return confusionMatrix;
    }
}