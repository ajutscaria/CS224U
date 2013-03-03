package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;

/***
 * Class that does the learning
 * @author Aju
 *
 */

public class LearnerEvent {
  //List of examples used to learn the parameters
  List<Example> dataset;
  //Maximum number of iterations to be run
  final static int maxIterations= 100;
  
  /***
   * Constructor to initialize the Learner with a list of training examples.
   * @param ds - List of training examples to learn the parameters from.
   */
  public LearnerEvent(List<Example> ds) {
    dataset = ds;
  }
  
  
  public double learnAndPredictNew(List<Example> testData) {
	    //for(Example ex:testData)
	    //	for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
	    //		printTree(sentence);
	    //	}
	    FeatureFactoryEvents ff = new FeatureFactoryEvents();
		// add the features
		List<Datum> data = ff.setFeaturesTrain(dataset);
		List<Datum> predicted = new ArrayList<Datum>();
		
		LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(data);
		double[] initial = new double[obj.domainDimension()];

		QNMinimizer minimizer = new QNMinimizer(15);
		double[][] weights = obj.to2D(minimizer.minimize(obj, 1e-4, initial,
				-1, null));

		for(Example ex:testData) {
			System.out.println(String.format("==================EXAMPLE %s======================",ex.id));
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				List<Datum> test = ff.setFeaturesTest(sentence);
				
				//for(EventMention event:sentence.get(EventMentionsAnnotation.class)) {
					//System.out.println("------------------Event " + event.getValue()+"--------------");
					//List<Datum> testDataEvent = new ArrayList<Datum>();
					//for(Datum d:test)
						//if(d.eventNode == event.getTreeNode()) {
							//testDataEvent.add(d);
						//}
					List<Datum> testDataWithLabel = new ArrayList<Datum>();
	
					for (int i = 0; i < test.size(); i += obj.labelIndex.size()) {
						testDataWithLabel.add(test.get(i));
					}
					MaxEntModel maxEnt = new MaxEntModel(obj.labelIndex, obj.featureIndex, weights);
					maxEnt.decodeForEntity(testDataWithLabel, test);
					
					//IdentityHashMap<Tree, Pair<Double, String>> map = new IdentityHashMap<Tree, Pair<Double, String>>();
	//
		//			for(Datum d:testDataWithLabel) {
			//			if (Utils.subsumesEvent(d.entityNode, sentence)) {
				//			map.put(d.entityNode, new Pair<Double, String>(0.0, "O"));
				//		} else {
					//		map.put(d.entityNode, new Pair<Double, String>(d.getProbability(), d.guessLabel));
						//}
					//}
					
				//	DynamicProgramming dynamicProgrammer = new DynamicProgramming(sentence, map, testDataWithLabel);
					//dynamicProgrammer.calculateLabels();
					
					predicted.addAll(testDataWithLabel);
					
					//System.out.println(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
					System.out.println("\n---------GOLD ENTITIES-------------------------");
					for(Datum d:testDataWithLabel) 
						if(d.label.equals("E"))
							System.out.println(d.eventNode + ":" + d.label);
					
					System.out.println("---------PREDICTIONS-------------------------");
					for(Datum d:testDataWithLabel)
						if(d.guessLabel.equals("E") || d.label.equals("E"))
							System.out.println(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.label, d.guessLabel));
					System.out.println("------------------------------------------\n");
				}
			//}
		}
		
				
		double f1 = Scorer.score(predicted);
		
		return f1;//testData;
	}
  
  /*
  public double learnAndPredict(List<Example> testData) {
	    //for(Example ex:testData)
	    //	for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
	    //		printTree(sentence);
	    //	}
	    FeatureFactory ff = new FeatureFactory();
		// add the features
		List<Datum> data = ff.setFeaturesTrain(dataset);
		
		LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(data);
		double[] initial = new double[obj.domainDimension()];

		QNMinimizer minimizer = new QNMinimizer(15);
		double[][] weights = obj.to2D(minimizer.minimize(obj, 1e-4, initial,
				-1, null));

		List<Datum> test = ff.setFeaturesTest(testData);
	    //System.out.println(obj.labelIndex.size());
		List<Datum> testDataInDatum = new ArrayList<Datum>();

		for (int i = 0; i < test.size(); i += obj.labelIndex.size()) {
			testDataInDatum.add(test.get(i));
		}
		Viterbi viterbi = new Viterbi(obj.labelIndex, obj.featureIndex, weights);
		viterbi.decode(testDataInDatum, test);
		
		for(Datum d:testDataInDatum)
			if(d.guessLabel.equals("E") || d.label.equals("E"))
				System.out.println(String.format("%-20s Gold: %s, Predicted: %s", d.word, d.label, d.guessLabel));
		
		double f1 = Scorer.score(testDataInDatum);
		
		return f1;//testData;
	} */
  
  private void printTree(CoreMap sentence)
  {
	  String text = sentence.toString();
	  TreebankLanguagePack tlp = new PennTreebankLanguagePack();
	  GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
	  LexicalizedParser lp = LexicalizedParser.loadModel();
	  Tree tree = lp.apply(text);
	  tree.pennPrint();
  }

}
