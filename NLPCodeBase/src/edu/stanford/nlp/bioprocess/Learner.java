package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;

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
  public Learner(List<Example> ds) {
    dataset = ds;
    parameters = new Params();
  }
  
  /***
   * Method that will learn parameters for the model and return it.
   * @return Parameters learnt.
   */
  public double[][] learn() {
	//checkTree();
	/*  
    for(Example example:dataset) {
      System.out.println("\n\nExample: " + example.id + "\nEntities in the paragraph\n-----------------------------");
      
      for(EntityMention entity:example.gold.get(EntityMentionsAnnotation.class)) {
        //System.out.println(entity.prettyPrint());
        //SemanticGraph graph = entity.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
  	  	//System.out.println(graph);
        //List<IndexedWord> words = Utils.findNodeInDependencyTree(entity);
        //Utils.checkEntityHead(words, entity.getSentence());
      }
      
      System.out.println("\nEvents in the paragraph\n-------------------------------");
      for(EventMention event:example.gold.get(EventMentionsAnnotation.class)){
        //System.out.println(event.prettyPrint());
        //SemanticGraph graph = event.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
	  	//System.out.println(graph);
      }
    }*/
    //return train();
	  return new double[0][0];
  }
  
  public double learnAndPredict(List<Example> testData) {
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
			System.out.println(String.format("%-20s Gold: %s, Predicted: %s", d.word, d.label, d.guessLabel));
		
		double f1 = Scorer.score(testDataInDatum);
		
		return f1;//testData;
	}
  
  private void checkTree()
  {
	  System.out.println("In check tree");
	  String text = "A particular region of each X chromosome contains several genes involved in the inactivation process.";
	  TreebankLanguagePack tlp = new PennTreebankLanguagePack();
	  GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
	  LexicalizedParser lp = LexicalizedParser.loadModel();
	  Tree tree = lp.apply(text);
	  tree.pennPrint();
  }

}
