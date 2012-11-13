import java.io.*;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Main {
	public static void main(String[] args) {
		System.out.println("Machine Learning - Programming Assignment 1");
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader("data/input/churn.arff"));
			ArffReader arffReader = new ArffReader(reader);
			Instances data = arffReader.getData();
			data.setClassIndex(data.numAttributes()-1);
			
			try {
				// Remove Phone
				data = Utils.removeAttribute(data, "4");
				
				// Account Length
				data = Utils.discretizeAttribute(data, "2", 10);
				
				// Area Code
				data = Utils.numericToNominal(data, "3");
				
				
				
				
			} catch (Exception e) {
				System.err.println("Filtering Attributes Failed!");
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
