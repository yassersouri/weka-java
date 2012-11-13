import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class Utils {
	public static String INDEX_OF_PHONE_ATTRIBUTE = "4";
	public static Instances removePhoneAttribute(Instances instances) throws Exception{
		Remove filter = new Remove();
		filter.setInvertSelection(false);
		filter.setAttributeIndices(INDEX_OF_PHONE_ATTRIBUTE);
		filter.setInputFormat(instances);
		return Filter.useFilter(instances, filter);
	}
}
