package edu.stanford.nlp.bioprocess;

import java.util.List;


/***
 * Class that does the learning
 * @author Aju
 *
 */

public class EntityPredictionLearner extends Learner {
  
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
	parameters.setWeights(weights);
	parameters.setFeatureIndex(obj.featureIndex);
	parameters.setLabelIndex(obj.labelIndex);
    return parameters;
  }
}
