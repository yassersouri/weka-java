import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Utils {
	
	public static Instances removeAttribute(Instances instances, String attribute) throws Exception{
		Remove filter = new Remove();
		filter.setInvertSelection(false);
		filter.setAttributeIndices(attribute);
		filter.setInputFormat(instances);
		return Filter.useFilter(instances, filter);
	}
	
	public static Instances discretizeAttribute(Instances data, String attribute, int numBins) throws Exception{
		Discretize filter = new Discretize();
		filter.setAttributeIndices(attribute);
		filter.setBins(numBins);
		filter.setMakeBinary(false);
		filter.setInputFormat(data);
		return Filter.useFilter(data, filter);
	}
	
	public static Instances numericToNominal(Instances data, String attribute) throws Exception{
		NumericToNominal filter = new NumericToNominal();
		filter.setAttributeIndices(attribute);
		filter.setInputFormat(data);
		return Filter.useFilter(data, filter);
	}
}
