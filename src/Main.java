import java.io.*;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Main {
	public static void main(String[] args) {
		System.out.println("Machine Learning - Programming Assignment 1");
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader("data/input/churn.arff"));
			ArffReader arffReader = new ArffReader(reader);
			Instances instances = arffReader.getData();
			instances.setClassIndex(instances.numAttributes()-1);
			
			Remove filter = new Remove();
			filter.setInvertSelection(false);
			filter.setAttributeIndices("4");
			try {
				filter.setInputFormat(instances);
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			
			Instances newData = null;
			try {
				newData = Filter.useFilter(instances, filter);
			} catch (Exception e) {
				// TODO Auto-generated catch block
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
