import weka.core.Instances;

public class Preprocess {
	public static void main(String[] args) {
		System.out.println("Machine Learning - Programming Assignment 1");
		Instances data = Utils.readFromFile("data/input/churn.arff");
		data.setClassIndex(data.numAttributes()-1);
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
		Utils.writeToFile(data, "data/temp/filtered.arff");
	}
}
