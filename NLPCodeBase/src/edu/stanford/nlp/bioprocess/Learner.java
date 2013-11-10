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

public class Learner {
  //Parameters used by the model
  Params parameters;
  //List of examples used to learn the parameters
  List<Example> dataset;
  //Maximum number of iterations to be run
  final static int maxIterations= 100;
  
  /***
   * Constructor to initialize the Learner with a list of training examples.
   * @param ds - List of training examples to learn the parameters from.
   */
  public Learner() {
    parameters = new Params();
  }
  
  /***
   * Method that will learn parameters for the model and return it.
   * @return Parameters learnt.
   */
	public Params learn(List<Example> dataset, FeatureExtractor ff) {
		List<BioDatum> data = ff.setFeaturesTrain(dataset);
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.label()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
		//System.out.println("Entity param length in learner 1:"+parameters.weights.length);
	    return parameters;
	}
	
	public Params learn(List<Example> dataset, FeatureExtractor ff, Params param) { //param: event parameters
		//System.out.println("Learner for entity");
		//System.out.println("Event parameters length:"+param.weights.length);
		List<BioDatum> data = ff.setFeaturesTrain(dataset, param);
		
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.label()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
		//System.out.println("Entity param length in learner 2:"+parameters.weights.length);
	    return parameters;
	}
	
	public Params learn(List<Example> dataset, EventRelationFeatureFactory ff) {
		List<BioDatum> data = ff.setFeaturesTrain(dataset);
		
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.label()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
	    return parameters;
	}
	
	public Params learn(List<Example> dataset, EventRelationFeatureFactory ff, Params param) {
		//List<BioDatum> data = ff.setFeaturesTrain(dataset);
		List<BioDatum> data = ff.setFeaturesTrain(dataset, param);
		GeneralDataset<String, String> dd = new Dataset<String, String>();
		for(BioDatum d:data) {
			dd.add(new BasicDatum<String, String>(d.features.getFeatures(), d.label()));
		}
		
		LinearClassifierFactory<String, String> lcFactory = new LinearClassifierFactory<String, String>();
		LinearClassifier<String,String> classifier = lcFactory.trainClassifier(dd);	
		
		parameters.setWeights(classifier.weights());
		parameters.setFeatureIndex(dd.featureIndex);
		parameters.setLabelIndex(dd.labelIndex);
	    return parameters;
	}
}
