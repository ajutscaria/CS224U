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

public class EventPredictionLearner extends Learner {
	
	@Override
	public Params learn(List<Example> dataset, FeatureExtractor ff) {
		List<BioDatum> data = ff.setFeaturesTrain(dataset);
		
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.label()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		LogInfo.logs(classifier.weightsAsMapOfCounters());
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
	    return parameters;
	}
	/*
	public Params learn(List<Example> dataset, EventFeatureFactory ff) {
		List<BioDatum> data = ff.setFeaturesTrain(dataset);
		
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.label()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		LogInfo.logs(classifier.weightsAsMapOfCounters());
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
	    return parameters;
	}*/
}
