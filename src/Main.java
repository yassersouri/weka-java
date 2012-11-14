import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;

public class Main {
	public static void main(String[] args) {
		//pre-process the data
		Preprocess.main(null);
		
		//read pre-processed files
		Instances data = Utils.readFromFile("data/temp/filtered.arff");
		data.setClassIndex(data.numAttributes()-1);
		
		System.out.println("ID3");
		evaluateId3(data);
		
		System.out.println("=====================================");
		
		System.out.println("C4.5 - REP");
		evaluateC45(data, true);

		System.out.println("=====================================");
		
		System.out.println("C4.5 - no REP");
		evaluateC45(data, false);
		
		System.out.println("=====================================");
		
		System.out.println("KNN - K=10");
		evaluateKNN(data, 10);
		
	}
	
	public static void evaluateId3(Instances data){
		Id3 classifier = new Id3();
		
		evaluateClassifier(data, classifier);
	}
	
	public static void evaluateC45(Instances data, boolean reducedErrorPruning){
		J48 classifier = new J48();
		
		classifier.setReducedErrorPruning(reducedErrorPruning);
		evaluateClassifier(data, classifier);
	}
	
	public static void evaluateKNN(Instances data, int k){
		IBk classifier = new IBk(k);
		
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
		
		System.out.println(nFolds + "-fold Cross Validation with 95% confidence interval: ");
		System.out.printf("[%.3f - %.3f]\n", lowerBound, upperBound);
	}
}
