package edu.stanford.nlp.bioprocess;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
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
	//System.out.println(begin + ":" + end);
    for(CoreMap sentence:sentences) {
      if(sentence.get(CharacterOffsetBeginAnnotation.class) <= begin && sentence.get(CharacterOffsetEndAnnotation.class) >= end)
        return sentence;
    }
    return null;
  }
  
  public static Span getSpanFromSentence(CoreMap sentence, int begin, int end) {
    Span span = new Span();
    //System.out.println(sentence);
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
	  //System.out.println(entity.getValue() + " " + words.size() );
	  IndexedWord head = words.get(0);
	  //System.out.println("\nFinding head word for  - " + entity.getValue());
	  //System.out.println(graph);
	  for(IndexedWord word : words) {
		  if(!head.get(PartOfSpeechAnnotation.class).startsWith("NN") && word.get(PartOfSpeechAnnotation.class).startsWith("NN")) {
			  head = word;
		  }
		  else if(!head.equals(word) && word.get(PartOfSpeechAnnotation.class).startsWith("NN") && words.contains(graph.getCommonAncestor(head, word)))
			  head = graph.getCommonAncestor(head, word);
	  }
	  //System.out.println(head.index()-1);
	  //System.out.println(entity.getExtent());
	  //System.out.println("Headword  - " + head.originalText());
	  span.setStart(head.index()-1);
	  span.setEnd(head.index());
	  return span;
  }

  public static void addAnnotation(Annotation document, EntityMention entity) {
    if(document.get(EntityMentionsAnnotation.class) == null) {
      List<EntityMention> mentions = new ArrayList<EntityMention>();
      mentions.add(entity);
      document.set(EntityMentionsAnnotation.class, mentions);
    }
    else
      document.get(EntityMentionsAnnotation.class).add(entity);
  }
  
  public static void addAnnotation(Annotation document, EventMention event) {
    if(document.get(EventMentionsAnnotation.class) == null) {
      List<EventMention> mentions = new ArrayList<EventMention>();
      mentions.add(event);
      document.set(EventMentionsAnnotation.class, mentions);
    }
    else
      document.get(EventMentionsAnnotation.class).add(event);
  }
  
  public static void writeFile(List<Example> data, String fileName) {
	  // Write to disk with FileOutputStream
	    try{
	    FileOutputStream f_out = new FileOutputStream(fileName);

	    // Write object with ObjectOutputStream
	    ObjectOutputStream obj_out = new ObjectOutputStream (f_out);

	    // Write object out to disk
	    obj_out.writeObject ( data);
	    }catch (Exception ex) {
	    	
	    }
  }
  
  public static List<Example> readFile(String fileName) {
	// Read from disk using FileInputStream
	  try
	  {
	  FileInputStream f_in = new FileInputStream(fileName);

	  // Read object using ObjectInputStream
	  ObjectInputStream obj_in = new ObjectInputStream (f_in);

	  // Read an object
	  Object obj = obj_in.readObject();
	  return (List<Example>) obj;
	  }catch(Exception ex) {
		  System.out.println(ex.toString());
	  }
	  return null;
  }
  
}
