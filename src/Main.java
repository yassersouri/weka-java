import java.io.*;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Main {
	public static void main(String[] args) {
		System.out.println("Machine Learning - Programming Assignment 1");
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader("data/input/churn.arff"));
			ArffReader arffReader = new ArffReader(reader);
			Instances instances = arffReader.getData();
			instances.setClassIndex(instances.numAttributes()-1);
			
			Instances data = null;
			try {
				data = Utils.removePhoneAttribute(instances);
			} catch (Exception e) {
				System.out.println("Removing Phone Attribute Failed!");
				e.printStackTrace();
			}
			
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
