package edu.stanford.nlp.bioprocess;

import java.util.List;


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
  public Params learn(List<Example> ds, FeatureExtractor ff) {
	dataset = ds;
	// add the features
	List<Datum> data = ff.setFeaturesTrain(dataset);
	LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(data);
	double[] initial = new double[obj.domainDimension()];

	QNMinimizer minimizer = new QNMinimizer(15);
	double[][] weights = obj.to2D(minimizer.minimize(obj, 1e-4, initial, -1, null));
	
	double[][] weightsAll = new double[ArgumentRelation.getSemanticRoles().size()][obj.featureIndex.size()];
	
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
	parameters.setFeatureIndex(obj.featureIndex);
	parameters.setLabelIndex(obj.labelIndex);
    return parameters;
  }
}
