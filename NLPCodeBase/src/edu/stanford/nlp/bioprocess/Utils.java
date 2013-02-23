package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class Utils {
  public static boolean checkEntityHead(List<IndexedWord> words, CoreMap sentence) {
	  SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
	  //System.out.println(graph);
	  for(IndexedWord word: words) {
		  System.out.println("\nCurrent word : " + word);
		  for(IndexedWord w:graph.getChildList(word)) {
			  System.out.println("\tRelated to - " + w);
		  }
	  }
	  return true;
  }
  
  /***
   * Find the list of nodes in the Semantic graph that an entity maps to.
   * @param sentence - The sentence in which we are looking for the nodes
   * @param span - The extend of the entity
   * @return
   */
  public static List<IndexedWord> findNodeInDependencyTree(ArgumentMention mention) {
	CoreMap sentence = mention.getSentence();
	Span span = mention.getExtent();
    SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
    //System.out.println(graph);
    //IndexedWord root = graph.getFirstRoot();
    ArrayList<IndexedWord> dependencyNodes = new ArrayList<IndexedWord>();
    for(IndexedWord word : graph.getAllNodesByWordPattern(".*")) {
      //System.out.println(word.value() + "--" + word.index());
      if(word.index() - 1 >= span.start() && word.index()-1 < span.end()) {
        //System.out.println(word.value() + ":" + word.beginPosition());
        dependencyNodes.add(word);
      }
      //System.out.println();
    }
    return dependencyNodes;
  }
  
  public static CoreMap getContainingSentence(List<CoreMap> sentences, int begin, int end) {
    for(CoreMap sentence:sentences) {
      if(sentence.get(CharacterOffsetBeginAnnotation.class) <= begin && sentence.get(CharacterOffsetEndAnnotation.class) >= end)
        return sentence;
    }
    return null;
  }
  
  public static Span getSpanFromSentence(CoreMap sentence, int begin, int end) {
    Span span = new Span();
    for(CoreLabel label:sentence.get(TokensAnnotation.class)) {
      if(label.beginPosition() == begin)
        span.setStart(label.index() - 1);
      if(label.endPosition() == end)
        span.setEnd(label.index());
    }
    return span;
  }
  
  public static Span findEntityHeadWord(EntityMention entity) {
	  Span span = new Span();
	  List<IndexedWord> words = findNodeInDependencyTree(entity);
	  SemanticGraph graph = entity.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
	  for(IndexedWord word : words) {
		  for(IndexedWord w : graph.getChildList(word)) {
			  
		  }
	  }
	  return span;
  }
  
}
