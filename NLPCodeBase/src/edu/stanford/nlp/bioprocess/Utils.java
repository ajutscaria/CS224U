package edu.stanford.nlp.bioprocess;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.BeginIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.StringUtils;

public class Utils {
  public static List<String> Punctuations = Arrays.asList(".", ",");
  public static int countBad = 0;
	
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
  
  public static List<IndexedWord> findNodeInDependencyTree(CoreMap sentence, Span span) {
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
  
  public static Span findHeadWord(CoreMap sentence, Span nodeSpan) {
	  if(nodeSpan.end()-nodeSpan.start() == 1)
		  return nodeSpan;

	  List<IndexedWord> words = findNodeInDependencyTree(sentence, nodeSpan);
	  if(words.size()==0) {
		  //System.out.println("Span not found in dependency tree.");
		  return new Span(nodeSpan.start(), nodeSpan.start() + 1);
	  }
	  Span span = new Span();
	  SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
	  //System.out.println(nodeSpan + " " + words.size() );
	  
	  IndexedWord head = words.get(0);
	  //System.out.println("\nFinding head word for  - " + entity.getValue());
	  //System.out.println(graph);
	  for(IndexedWord word : words) {
		  if(!head.equals(word) && words.contains(graph.getCommonAncestor(head, word)))
			  head = graph.getCommonAncestor(head, word);
	  }
	  //System.out.println(head.index()-1);
	  //System.out.println(entity.getExtent());
	  //System.out.println("Headword  - " + head.originalText());
	  span.setStart(head.index()-1);
	  span.setEnd(head.index());
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
  
  public static Tree getEntityNodeBest(CoreMap sentence, EntityMention entity) {
	  Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
	  //syntacticParse.setSpans();
	  Span entitySpan = entity.getExtent();
	  for (Tree node : syntacticParse.preOrderNodeList()) {
		  if(node.isLeaf())
			  continue;
		  
		  IntPair span = node.getSpan();
		  if(span.getSource() == entitySpan.start() && span.getTarget() == entitySpan.end()-1) {
			  //System.out.println(node.value());
			  //System.out.println(entity.getValue() + "| Found match - " + node);
			  if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
				  return node;
		  }
		  if(span.getSource() == entitySpan.start() - 1 && span.getTarget() == entitySpan.end() - 1) {
			  //To check for an extra determiner like "a" or "the" in front of the entity
			  String POSTag = sentence.get(TokensAnnotation.class).get(span.getSource()).get(PartOfSpeechAnnotation.class);
			  if(POSTag.equals("DT") || POSTag.equals("PRP$")) {
				  //System.out.println(entity.getValue() + "| Found match - " + node);
				  if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
					  return node;
			  }
		  }
		  if(span.getSource() == entitySpan.start() && span.getTarget() == entitySpan.end()) {
			  //To check for an extra punctuation at the end of the entity.
			  List<Tree> leaves = node.getLeaves();
			  if(Punctuations.contains(leaves.get(leaves.size()-1).toString())) {
				  //System.out.println(entity.getValue() + "| Found match - " + node);
				  if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
				  	return node;
			  }
		  }
	  }
	  return null;  
  }
  
  public static Tree getEntityNode(CoreMap sentence, EntityMention entity) {	  
	  //Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
	  EntityMention entityNew = new EntityMention("id", entity.getSentence(), new Span(entity.getExtentTokenStart(), entity.getExtentTokenEnd()));
	  
	  // Perfect Match
	  Tree bestMatch = getEntityNodeBest(sentence, entityNew);
	  if (bestMatch != null) {
		  return bestMatch;
	  }
	  //System.out.println(entity.getValue());
	  //System.out.println("Missed first section");
	  EntityMention entityNoLastToken = new EntityMention("id", entityNew.getSentence(), new Span(entityNew.getExtentTokenStart(), entityNew.getExtentTokenEnd() -1 ));
	  while(entityNoLastToken.getExtent().end() - entityNoLastToken.getExtent().start() != 0) {
		  //Remove last token
		  bestMatch = getEntityNodeBest(sentence, entityNoLastToken);
		  if (bestMatch != null) {
			  //System.out.println(entity.getValue() + "| Found match - " + bestMatch);
			  return bestMatch;
		  }
		  entityNoLastToken = new EntityMention("id", entityNoLastToken.getSentence(), new Span(entityNoLastToken.getExtentTokenStart(), entityNoLastToken.getExtentTokenEnd() -1 ));
	  }
	  //System.out.println("Missed second section");
	  EntityMention entityNoFirstToken = new EntityMention("id", entityNew.getSentence(), new Span(entityNew.getExtentTokenStart()+1, entityNew.getExtentTokenEnd() ));
	  while(entityNoFirstToken.getExtent().end() - entityNoFirstToken.getExtent().start() != 0) {
		  //Remove first token
		  
		  bestMatch = getEntityNodeBest(sentence, entityNoFirstToken);
		  if (bestMatch != null) {
			  //System.out.println(entity.getValue() + "| Found match - " + bestMatch);
			  return bestMatch;
		  }
		  entityNoFirstToken = new EntityMention("id", entityNoFirstToken.getSentence(), new Span(entityNoFirstToken.getExtentTokenStart()+1, entityNoFirstToken.getExtentTokenEnd() ));
	  }
	  
	  countBad+=1;
	  //System.out.println("No match found for - " + entity.getValue() + ":"+  countBad);

	  //syntacticParse.pennPrint();
	  return null;
  }
  
  public static Tree getSingleEventNode(CoreMap sentence, EventMention event) {
	  Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
	  for(int spanStart = event.getExtentTokenStart(); spanStart < event.getExtentTokenEnd(); spanStart++) {
		  for(Tree node:syntacticParse.postOrderNodeList()) {
			  if(node.isLeaf())
				  continue;
			  
			  IntPair span = node.getSpan();
			  if(span.getSource() == spanStart && span.getTarget() == spanStart && 
					  ( (node.value().startsWith("VB") && !node.firstChild().value().equals("is")) || node.value().startsWith("NN"))) {
				  //System.out.println("Compressing " + event.getValue() + " to " + node);
				  return node;
			  }
		  }
	  }
	  //If everything fails, returns first pre-terminal
	  for(Tree node:syntacticParse.postOrderNodeList()) {
		  if(node.isLeaf())
			  continue;
		  
		  IntPair span = node.getSpan();
		  if(span.getSource() == event.getExtentTokenStart() && span.getTarget() == event.getExtentTokenStart()) {
			  //System.out.println("Compressing " + event.getValue() + " to " + node);
			  return node;
		  }
	  }

	  return null;
  }
  
  public static Tree getEventNode(CoreMap sentence, EventMention event) {
	  Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
	  Span entitySpan = event.getExtent();
	  for(Tree node:syntacticParse.postOrderNodeList()) {
		  if(node.isLeaf())
			  continue;
		  
		  IntPair span = node.getSpan();
		  if(span.getSource() == entitySpan.start() && span.getTarget() == entitySpan.end()-1) {
			  return node;
		  }
		  
		  if(span.getSource() == entitySpan.start() - 1 && span.getTarget() == entitySpan.end() - 1) {
			  //To check for an extra determiner like "a" or "the" in front of the entity
			  String POSTag = sentence.get(TokensAnnotation.class).get(span.getSource()).get(PartOfSpeechAnnotation.class);
			  if(POSTag.equals("DT") || POSTag.equals("PRP$")) {
				  //System.out.println("Matching " + event.getValue() + " with " + node);
				  return node;
			  }
		  }
	  }
	  Tree ret = getSingleEventNode(sentence, event);
	  if(ret!=null)
		  return ret;
	  
	  syntacticParse.pennPrint();
	  System.out.println("No match found for - " + event.getValue());
	  return null;
  }
  
  /*
  public void addTreeNodeAnnotations(CoreMap sentence) {
		 HashMap<Tree, CoreLabel> treeLabelMap = new HashMap<Tree, CoreLabel>();
		 Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		 List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
		 for(Tree leaf : syntacticParse.getLeaves()) {
			 if(leaf.label() instanceof CoreLabel) {
				 CoreLabel label = (CoreLabel) leaf.label();
			     // find matching token in tokens array
			     CoreLabel matching = null;
			     for(CoreLabel l : tokens) {
			    	 if(l.beginPosition() == label.beginPosition() && l.endPosition() == label.endPosition()) {
			    		 matching = l;
			    		 break;
			    	 }
			     }
			     if(matching != null) 
			    	 treeLabelMap.put(leaf, matching);
			     else 
			    	 System.out.println("ERROR: found no matching token for " + label);
			 } else {
				 System.out.println("ERROR: leaf is not CoreLabel instance: " + leaf);
			 }
		 }
	 }*/
  
  public static void addAnnotation(Annotation document, EntityMention entity) {
    if(document.get(EntityMentionsAnnotation.class) == null) {
      List<EntityMention> mentions = new ArrayList<EntityMention>();
      mentions.add(entity);
      document.set(EntityMentionsAnnotation.class, mentions);
    }
    else
      document.get(EntityMentionsAnnotation.class).add(entity);
    
    CoreMap sentence = entity.getSentence();
    if (sentence.get(EntityMentionsAnnotation.class) == null) {
    	List<EntityMention> mentions = new ArrayList<EntityMention>();
        mentions.add(entity);
        sentence.set(EntityMentionsAnnotation.class, mentions);
    }
    else
    	sentence.get(EntityMentionsAnnotation.class).add(entity);
      
  }
  
  public static void addAnnotation(Annotation document, EventMention event) {
    if(document.get(EventMentionsAnnotation.class) == null) {
      List<EventMention> mentions = new ArrayList<EventMention>();
      mentions.add(event);
      document.set(EventMentionsAnnotation.class, mentions);
    }
    else
      document.get(EventMentionsAnnotation.class).add(event);
    
    CoreMap sentence = event.getSentence();
    if (sentence.get(EventMentionsAnnotation.class) == null) {
    	List<EventMention> mentions = new ArrayList<EventMention>();
        mentions.add(event);
        sentence.set(EventMentionsAnnotation.class, mentions);
    }
    else
    	sentence.get(EventMentionsAnnotation.class).add(event);
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
  
  public static IdentityHashSet<Tree> getEntityNodes(Example ex) {
	  IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
	  for(EntityMention entity : ex.gold.get(EntityMentionsAnnotation.class))
  		set.add(entity.getTreeNode());
  	  return set;
  }
  
  public static IdentityHashSet<Tree> getEntityNodesFromSentence(CoreMap sentence) {
	  IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
	  for(EntityMention entity : sentence.get(EntityMentionsAnnotation.class))
  		set.add(entity.getTreeNode());
	  return set;
  }
  
  public static int getMaxHeight(Tree node) {
	  int maxHeight = 0;
	  for(Tree leaf:node.getLeaves())
		  if(leaf.depth() - node.depth() > maxHeight)
			  maxHeight = leaf.depth() - node.depth();
	  return maxHeight;
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
  
  public static boolean isChildOfEntity(Set<Tree> entities, Tree node) {
	for(Tree entity:entities) {
		//System.out.println(entity);
		if(entity != null && (entity.equals(node) || entity.depth(node) != -1))
			return true;
	}
	return false;
  }

  public static RelationType getArgumentMentionRelation(EventMention event, Tree entityNode) {
	for (ArgumentRelation argRel : event.getArguments()) {
		if (argRel.mention.getTreeNode() == entityNode) {
			return argRel.type;
		}
	}
	return RelationType.NONE;
  }
  
  public static String getPathString(List<String> path) {
	  return StringUtils.join(path, " ");
  }

	public static boolean subsumesEvent(Tree entityNode, CoreMap sentence) {
		for (EventMention ev : sentence.get(EventMentionsAnnotation.class)) {
			if (entityNode.dominates(ev.getTreeNode())) {
				return true;
			}
		}
		return false;
	}
}
