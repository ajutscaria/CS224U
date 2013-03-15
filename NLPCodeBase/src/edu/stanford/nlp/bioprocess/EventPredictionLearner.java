package edu.stanford.nlp.bioprocess;

import java.util.List;

import fig.basic.LogInfo;

/***
 * Class that does the learning
 * @author Aju
 *
 */

public class EventPredictionLearner extends Learner {
	
	@Override
	public Params learn(List<Example> dataset, FeatureExtractor ff) {
		// add the features
		List<Datum> data = ff.setFeaturesTrain(dataset);
		LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(data);
		double[] initial = new double[obj.domainDimension()];

		QNMinimizer minimizer = new QNMinimizer(15);
		double[][] weights = obj.to2D(minimizer.minimize(obj, 1e-4, initial, -1, null));
		parameters.setWeights(weights);
		parameters.setFeatureIndex(obj.featureIndex);
		parameters.setLabelIndex(obj.labelIndex);
		
		//for(int i = 0; i<obj.featureIndex.size(); i++) {
		//	LogInfo.logs(obj.featureIndex.get(i) + ":" + weights[0][i] + "," + weights[1][i]);
		//}
		
	    return parameters;
	}
}
