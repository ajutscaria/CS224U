package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

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
  public Params learn() {
    for(Example example:dataset) {
      System.out.println("\n\nExample: " + example.id + "\nEntities in the paragraph\n-----------------------------");
      for(EntityMention entity:example.gold.get(EntityMentionsAnnotation.class)) {
        System.out.println(entity.prettyPrint());
      }
      
      System.out.println("\nEvents in the paragraph\n-------------------------------");
      for(EventMention event:example.gold.get(EventMentionsAnnotation.class))
        System.out.println(event.prettyPrint());
      break;
    }
    return parameters;
  }
  
  /***
   * Find the list of nodes in the Semantic graph that an entity maps to.
   * @param sentence - The sentence in which we are looking for the nodes
   * @param span - The extend of the entity
   * @return
   */
  private List<IndexedWord> findNodeInDependencyTree(CoreMap sentence, Span span) {
    SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
    //System.out.println(graph);
    //IndexedWord root = graph.getFirstRoot();
    ArrayList<IndexedWord> dependencyNodes = new ArrayList<IndexedWord>();
    for(IndexedWord word : graph.getAllNodesByWordPattern(".*")) {
      //System.out.println(word.value() + "--" + word.index());
      if(word.index() - 1 >= span.start() && word.index()-1 < span.end()) {
        System.out.println(word.value() + ":" + word.beginPosition());
        dependencyNodes.add(word);
      }
      //System.out.println();
    }
    return dependencyNodes;
  }
}
