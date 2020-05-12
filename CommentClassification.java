
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class CommentClassification {

    public static void main(String[] args) throws Exception {

        //load data set for training//
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("/Users/Nada/NetBeansProjects/DeclutteringTool/web/WEB-INF/DeclutterData.arff"));
        Instances train = new Instances(breader);
        train.setClassIndex(train.numAttributes() - 1);


        //test data set for testing 
        breader = new BufferedReader(new FileReader("/Users/Nada/NetBeansProjects/DeclutteringTool/web/WEB-INF/JabrefComments0-1000.arff"));
        Instances test = new Instances(breader);
        test.setClassIndex(test.numAttributes() - 1);

        // classifier applied 
        J48 svm = new J48();
        svm.buildClassifier(train);
        Instances labeled = new Instances(test);

        
        //for label instances
        for (int i = 0; i < test.numInstances(); i++) {
            double clsLabel = svm.classifyInstance(test.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }

        // Evaluation using cross validation (10 folds) and external test data 
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(svm, test);
        //eval.crossValidateModel(svm, train, 10, new Random(1));
        //System.out.println("Estimated Accuracy: " + Double.toString(eval.pctCorrect()));

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("correct:" + eval.correct());
        System.out.println("error:" + eval.errorRate());
        System.out.println("incorrect:" + eval.incorrect());

        //save labeled data if it is informative or not informative 
        BufferedWriter writer = new BufferedWriter(
                new FileWriter("/Users/Nada/NetBeansProjects/DeclutteringTool/web/WEB-INF/labelledTestData.arff"));
        writer.write(labeled.toString());
    }
}
