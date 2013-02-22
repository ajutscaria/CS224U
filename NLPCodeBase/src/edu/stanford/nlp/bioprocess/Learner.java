package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class Learner {
  Params parameters;
  List<Example> dataset;
  final static int maxIterations= 100;
  
  public Learner(List<Example> ds) {
    dataset = ds;
    parameters = new Params();
  }
  
  public Params learn() {
    for(Example example:dataset) {
      System.out.println("\n\nExample: " + example.id + "\nEntities in the paragraphhh\n-----------------------------");
      for(EntityMention entity:example.gold.get(EntityMentionsAnnotation.class)) {
        System.out.println(entity.prettyPrint());// + entity.getExtent());
        //System.out.println(findNodeInDependencyTree(entity.getSentence(), entity.getExtent()));
      }
      
      System.out.println("\nEvents in the paragraph\n-------------------------------");
      for(EventMention event:example.gold.get(EventMentionsAnnotation.class))
        System.out.println(event.prettyPrint());
    }
    return parameters;
  }
  
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
