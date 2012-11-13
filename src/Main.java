import java.io.*;

public class Main {
	public static void main(String[] args) {
		System.out.println("Machine Learning - Programming Assignment 1");
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader("data/input/churn.arff"));
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
