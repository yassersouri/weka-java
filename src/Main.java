import java.io.*;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;

public class Main {
	public static void main(String[] args) {
		System.out.println("Machine Learning - Programming Assignment 1");
		Instances data = null;
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader("data/input/churn.arff"));
			ArffReader arffReader = new ArffReader(reader);
			data = arffReader.getData();
			data.setClassIndex(data.numAttributes()-1);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			// Account Length
			data = Utils.discretizeAttribute(data, "2", 10);
			
			// Area Code
			data = Utils.numericToNominal(data, "3");
			
			//VMail Messages, Day Mins
			data = Utils.discretizeAttribute(data, "7,8,9,10,11,12,13,14,15,16,17", 10);
			
			//Intl Calls, Intl Charge
			data = Utils.discretizeAttribute(data, "18,19,20", 5);
			
			// Remove Phone
			data = Utils.removeAttribute(data, "4");
		} catch (Exception e) {
			System.err.println("Filtering Attributes Failed!");
			e.printStackTrace();
		}
		
		try {
			ArffSaver arffSaver = new ArffSaver();
			arffSaver.setFile(new File("data/temp/filtered.arff"));
			arffSaver.setInstances(data);
			arffSaver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
