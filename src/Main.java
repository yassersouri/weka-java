import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;

public class Main {
	public static void main(String[] args) {
		//pre-process the data
		Preprocess.main(null);
		
		//read pre-processed files
		Instances data = Utils.readFromFile("data/temp/filtered.arff");
		data.setClassIndex(data.numAttributes()-1);
		
		//build classifier
		Id3 classifier = new Id3();
		
		//evaluate
		evaluateClassifier(data, classifier);
	}
	
	public static void evaluateClassifier(Instances data, Classifier classifier){
		// Z_n for 95% interval
		double Z_n = 1.96;
		int nFolds = 10;
		
		double errorS = 0.0;
		try {
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(classifier, data, nFolds, new Random(1));
			errorS = eval.meanAbsoluteError();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		double interval = Z_n * Math.sqrt(errorS * (1 - errorS) / (data.numInstances()/nFolds));
		double lowerBound = errorS - interval;
		double upperBound = errorS + interval;
		
		System.out.print("10-fold Cross Validation with 95% confidence interval: ");
		System.out.printf("[%.3f - %.3f]\n", lowerBound, upperBound);
	}
}
