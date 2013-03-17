package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.BasicDatum;
import fig.basic.LogInfo;


/***
 * Class that does the learning
 * @author Aju
 *
 */

public class SRLPredictionLearner extends Learner {
  
  /***
   * Method that will learn parameters for the model and return it.
   * @return Parameters learnt.
   */
  public Params learn(List<Example> dataset, FeatureExtractor ff) {
	  
		List<BioDatum> data = ff.setFeaturesTrain(dataset);
		
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.role()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		LogInfo.logs(classifier.weightsAsMapOfCounters());
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
	    return parameters;
	    /*
	dataset = ds;
	// add the features
	List<BioDatum> data = ff.setFeaturesTrain(dataset);
	LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(data, true);
	
	double[] initial = new double[obj.domainDimension()];

	QNMinimizer minimizer = new QNMinimizer(15);
	double[][] weights = obj.to2D(minimizer.minimize(obj, 1e-4, initial, -1, null));
	
	double[][] weightsAll = new double[ArgumentRelation.getSemanticRoles().size()][obj.featureIndex.size()];
	
	System.out.println(obj.labelIndex.toString());
	for (String srl : ArgumentRelation.getSemanticRoles()) {
		boolean addedLabel = false;
		if (!Utils.stringObjectContains(obj.labelIndex.indexes.keySet(), srl)) {
			obj.labelIndex.add(srl);
			addedLabel = true;
		}
		int indexOfSrl = obj.labelIndex.indexOf(srl);
		for (int cntr = 0; cntr < obj.featureIndex.size(); cntr++) {
			if (addedLabel) {
				weightsAll[indexOfSrl][cntr] = Double.NEGATIVE_INFINITY;
			} else {
				weightsAll[indexOfSrl][cntr] = weights[indexOfSrl][cntr];
			}
		}
	}
	
	parameters.setWeights(weightsAll);
	//parameters.setFeatureIndex(obj.featureIndex);
	//parameters.setLabelIndex(obj.labelIndex);
    return parameters;
    */
  }
}
