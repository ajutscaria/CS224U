package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;

import java.util.Collections;
import java.util.Comparator;

import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentenceIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.IntTuple;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import fig.basic.LogInfo;

public class Utils {
  public static List<String> Punctuations = Arrays.asList(".", ",");
  public static int countBad = 0;
	
  public static boolean checkEntityHead(List<IndexedWord> words, CoreMap sentence) {
	  SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
	  //LogInfo.logs(graph);
	  for(IndexedWord word: words) {
		  LogInfo.logs("\nCurrent word : " + word);
		  for(IndexedWord w:graph.getChildList(word)) {
			  LogInfo.logs("\tRelated to - " + w);
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
    //LogInfo.logs(graph);
    //IndexedWord root = graph.getFirstRoot();
    ArrayList<IndexedWord> dependencyNodes = new ArrayList<IndexedWord>();
    for(IndexedWord word : graph.getAllNodesByWordPattern(".*")) {
      //LogInfo.logs(word.value() + "--" + word.index());
      if(word.index() - 1 >= span.start() && word.index()-1 < span.end()) {
        //LogInfo.logs(word.value() + ":" + word.beginPosition());
        dependencyNodes.add(word);
      }
      //LogInfo.logs();
    }
    return dependencyNodes;
  }
  
  public static List<IndexedWord> findNodeInDependencyTree(CoreMap sentence, Span span) {
	    SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
	    //LogInfo.logs(graph);
	    //IndexedWord root = graph.getFirstRoot();
	    ArrayList<IndexedWord> dependencyNodes = new ArrayList<IndexedWord>();
	    for(IndexedWord word : graph.getAllNodesByWordPattern(".*")) {
	      //LogInfo.logs(word.value() + "--" + word.index());
	      if(word.index() - 1 >= span.start() && word.index()-1 < span.end()) {
	        //LogInfo.logs(word.value() + ":" + word.beginPosition());
	        dependencyNodes.add(word);
	      }
	      //LogInfo.logs();
	    }
	    return dependencyNodes;
	  }
  
  public static CoreMap getContainingSentence(List<CoreMap> sentences, int begin, int end) {
	//LogInfo.logs(begin + ":" + end);
    for(CoreMap sentence:sentences) {
      if(sentence.get(CharacterOffsetBeginAnnotation.class) <= begin && sentence.get(CharacterOffsetEndAnnotation.class) >= end)
        return sentence;
    }
    return null;
  }
  
  public static Span getSpanFromSentence(CoreMap sentence, int begin, int end) {
    Span span = new Span();
    //LogInfo.logs(sentence);
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
		  //LogInfo.logs("Span not found in dependency tree.");
		  return new Span(nodeSpan.start(), nodeSpan.start() + 1);
	  }
	  Span span = new Span();
	  SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
	  //LogInfo.logs(nodeSpan + " " + words.size() );
	  
	  IndexedWord head = words.get(0);
	  //LogInfo.logs("\nFinding head word for  - " + entity.getValue());
	  //LogInfo.logs(graph);
	  for(IndexedWord word : words) {
		  if(!head.equals(word) && words.contains(graph.getCommonAncestor(head, word)))
			  head = graph.getCommonAncestor(head, word);
	  }
	  //LogInfo.logs(head.index()-1);
	  //LogInfo.logs(entity.getExtent());
	  //LogInfo.logs("Headword  - " + head.originalText());
	  span.setStart(head.index()-1);
	  span.setEnd(head.index());
	  return span;
  }
  
  public static Span findEntityHeadWord(EntityMention entity) {
	  Span span = new Span();
	  
	  List<IndexedWord> words = findNodeInDependencyTree(entity);
	  SemanticGraph graph = entity.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
	  //LogInfo.logs(entity.getValue() + " " + words.size() );
	  IndexedWord head = words.get(0);
	  //LogInfo.logs("\nFinding head word for  - " + entity.getValue());
	  //LogInfo.logs(graph);
	  for(IndexedWord word : words) {
		  if(!head.get(PartOfSpeechAnnotation.class).startsWith("NN") && word.get(PartOfSpeechAnnotation.class).startsWith("NN")) {
			  head = word;
		  }
		  else if(!head.equals(word) && word.get(PartOfSpeechAnnotation.class).startsWith("NN") && words.contains(graph.getCommonAncestor(head, word)))
			  head = graph.getCommonAncestor(head, word);
	  }
	  //LogInfo.logs(head.index()-1);
	  //LogInfo.logs(entity.getExtent());
	  //LogInfo.logs("Headword  - " + head.originalText());
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
			  //LogInfo.logs(node.value());
			  //LogInfo.logs(entity.getValue() + "| Found match - " + node);
			  if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
				  return node;
		  }
		  if(span.getSource() == entitySpan.start() - 1 && span.getTarget() == entitySpan.end() - 1) {
			  //To check for an extra determiner like "a" or "the" in front of the entity
			  String POSTag = sentence.get(TokensAnnotation.class).get(span.getSource()).get(PartOfSpeechAnnotation.class);
			  if(POSTag.equals("DT") || POSTag.equals("PRP$")) {
				  //LogInfo.logs(entity.getValue() + "| Found match - " + node);
				  if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
					  return node;
			  }
		  }
		  if(span.getSource() == entitySpan.start() && span.getTarget() == entitySpan.end()) {
			  //To check for an extra punctuation at the end of the entity.
			  List<Tree> leaves = node.getLeaves();
			  if(Punctuations.contains(leaves.get(leaves.size()-1).toString())) {
				  //LogInfo.logs(entity.getValue() + "| Found match - " + node);
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
	  //LogInfo.logs(entity.getValue());
	  //LogInfo.logs("Missed first section");
	  EntityMention entityNoLastToken = new EntityMention("id", entityNew.getSentence(), new Span(entityNew.getExtentTokenStart(), entityNew.getExtentTokenEnd() -1 ));
	  while(entityNoLastToken.getExtent().end() - entityNoLastToken.getExtent().start() != 0) {
		  //Remove last token
		  bestMatch = getEntityNodeBest(sentence, entityNoLastToken);
		  if (bestMatch != null) {
			  //LogInfo.logs(entity.getValue() + "| Found match - " + bestMatch);
			  return bestMatch;
		  }
		  entityNoLastToken = new EntityMention("id", entityNoLastToken.getSentence(), new Span(entityNoLastToken.getExtentTokenStart(), entityNoLastToken.getExtentTokenEnd() -1 ));
	  }
	  //LogInfo.logs("Missed second section");
	  EntityMention entityNoFirstToken = new EntityMention("id", entityNew.getSentence(), new Span(entityNew.getExtentTokenStart()+1, entityNew.getExtentTokenEnd() ));
	  while(entityNoFirstToken.getExtent().end() - entityNoFirstToken.getExtent().start() != 0) {
		  //Remove first token
		  
		  bestMatch = getEntityNodeBest(sentence, entityNoFirstToken);
		  if (bestMatch != null) {
			  //LogInfo.logs(entity.getValue() + "| Found match - " + bestMatch);
			  return bestMatch;
		  }
		  entityNoFirstToken = new EntityMention("id", entityNoFirstToken.getSentence(), new Span(entityNoFirstToken.getExtentTokenStart()+1, entityNoFirstToken.getExtentTokenEnd() ));
	  }
	  
	  countBad+=1;
	  LogInfo.logs("No ENTITY match found for - " + entity.getValue() + ":"+  countBad);

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
					  ( (node.value().startsWith("VB") && !node.firstChild().value().equals("is") && !node.firstChild().value().equals("in")) || node.value().startsWith("NN"))) {
				  //LogInfo.logs("Compressing " + event.getValue() + " to " + node);
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
			  //LogInfo.logs("Compressing " + event.getValue() + " to " + node);
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
			  if(node.headPreTerminal(new CollinsHeadFinder()).value().equals("IN"))
				  return getSingleEventNode(sentence, event);
			  return node.headPreTerminal(new CollinsHeadFinder());
		  }
		  
		  if(span.getSource() == entitySpan.start() - 1 && span.getTarget() == entitySpan.end() - 1) {
			  //To check for an extra determiner like "a" or "the" in front of the entity
			  String POSTag = sentence.get(TokensAnnotation.class).get(span.getSource()).get(PartOfSpeechAnnotation.class);
			  if(POSTag.equals("DT") || POSTag.equals("PRP$")) {
				  return  node.headPreTerminal(new CollinsHeadFinder());
			  }
		  }
	  }
	  Tree ret = getSingleEventNode(sentence, event);
	  if(ret!=null)
		  return ret.headPreTerminal(new CollinsHeadFinder());
	  
	  syntacticParse.pennPrint();
	  LogInfo.logs("No EVENT match found for - " + event.getValue());
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
			    	 LogInfo.logs("ERROR: found no matching token for " + label);
			 } else {
				 LogInfo.logs("ERROR: leaf is not CoreLabel instance: " + leaf);
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
    else {
    	int indexToInsert = 0;
    	for(;indexToInsert < document.get(EventMentionsAnnotation.class).size(); indexToInsert++) {
    		EventMention current = document.get(EventMentionsAnnotation.class).get(indexToInsert);
    		if(current.getSentence().get(CharacterOffsetBeginAnnotation.class) > event.getSentence().get(CharacterOffsetBeginAnnotation.class) || 
    				(current.getSentence().get(CharacterOffsetBeginAnnotation.class) == event.getSentence().get(CharacterOffsetBeginAnnotation.class) && current.getExtent().start() > event.getExtent().start()))
    			break;
    	}
        document.get(EventMentionsAnnotation.class).add(indexToInsert, event);
    }
    
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
  public static IdentityHashMap<Tree, EventType> getEventNodesFromSentence(CoreMap sentence) {
	  IdentityHashMap<Tree, EventType> map = new IdentityHashMap<Tree, EventType>();
	  for(EventMention event : sentence.get(EventMentionsAnnotation.class))
  		map.put(event.getTreeNode(), event.eventType);
	  return map;
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
  
  @SuppressWarnings("unchecked")
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
		  LogInfo.logs(ex.toString());
	  }
	  return null;
  }
  
  public static boolean isChildOfEntity(Set<Tree> entities, Tree node) {
	for(Tree entity:entities) {
		//LogInfo.logs(entity);
		if(entity != null && (entity.equals(node) || entity.depth(node) != -1))
			return true;
	}
	return false;
  }

  
  public static RelationType getArgumentMentionRelation(EventMention event, Tree entityNode) {
	for (ArgumentRelation argRel : event.getArguments()) {
		if (argRel.mention.getTreeNode() == entityNode && ArgumentRelation.getSemanticRoles().contains(argRel.type.toString())) {
			return argRel.type;
		}
	}
	return RelationType.NONE;
  }
 
  
  public static RelationType getArgumentMentionRelation(CoreMap sentence, Tree event, Tree node) {
	  for(EventMention eventMention:sentence.get(EventMentionsAnnotation.class)) {
		if(eventMention.getTreeNode() == event) {
			for (ArgumentRelation argRel : eventMention.getArguments()) {
				if (argRel.mention.getTreeNode() == node && ArgumentRelation.getSemanticRoles().contains(argRel.type.toString())) {
					return argRel.type;
				}
			}
		 }
	  }
	  return RelationType.NONE;
  }
  
  public static RelationType getEventEventRelation(Annotation example, Tree event1, Tree event2) {
	  List<EventMention> list = example.get(EventMentionsAnnotation.class);
	  for(EventMention eventMention:list) {
		if(eventMention.getTreeNode() == event1) {
			for (ArgumentRelation argRel : eventMention.getArguments()) {
				if (argRel.mention.getTreeNode() == event2 && ArgumentRelation.getEventRelations().contains(argRel.type.toString())) {
					//eventMention appears before argRel.mention. So, cause and enables are retained, but sub and next are rotated.
					if(list.indexOf(eventMention) < list.indexOf(argRel.mention)) {
						if(argRel.type == RelationType.Causes || argRel.type == RelationType.Enables || argRel.type == RelationType.CotemporalEvent ||
								argRel.type == RelationType.SameEvent)
							return argRel.type;
						else if(argRel.type == RelationType.NextEvent)
							return RelationType.PreviousEvent;
						else if(argRel.type == RelationType.SuperEvent)
							return RelationType.SubEvent;
					}
					else {
						if(argRel.type == RelationType.NextEvent || argRel.type == RelationType.SuperEvent || argRel.type == RelationType.CotemporalEvent ||
								argRel.type == RelationType.SameEvent)
							return argRel.type;
						else if(argRel.type == RelationType.Causes)
							return RelationType.Caused;
						else if(argRel.type == RelationType.Enables)
							return RelationType.Enabled;
					}
				}
			}
		 }
		else if(eventMention.getTreeNode() == event2) {
			for (ArgumentRelation argRel : eventMention.getArguments()) {
				if (argRel.mention.getTreeNode() == event1 && ArgumentRelation.getEventRelations().contains(argRel.type.toString())) {
					//eventMention appears before argRel.mention. So, cause and enables are retained, but sub and next are rotated.
					if(list.indexOf(eventMention) < list.indexOf(argRel.mention)) {
						if(argRel.type == RelationType.Causes || argRel.type == RelationType.Enables || argRel.type == RelationType.CotemporalEvent ||
								argRel.type == RelationType.SameEvent)
							return argRel.type;
						else if(argRel.type == RelationType.NextEvent)
							return RelationType.PreviousEvent;
						else if(argRel.type == RelationType.SuperEvent)
							return RelationType.SubEvent;
					}
					else {
						if(argRel.type == RelationType.NextEvent || argRel.type == RelationType.SuperEvent || argRel.type == RelationType.CotemporalEvent ||
								argRel.type == RelationType.SameEvent)
							return argRel.type;
						else if(argRel.type == RelationType.Causes)
							return RelationType.Caused;
						else if(argRel.type == RelationType.Enables)
							return RelationType.Enabled;
					}
				}
			}
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
	
	public static CoreLabel findCoreLabelFromTree(CoreMap sentence, Tree node) {
		Tree head = node.headPreTerminal(new CollinsHeadFinder());
		return sentence.get(TokensAnnotation.class).get(head.getSpan().getSource());
	}
	
	public static IndexedWord findDependencyNode(CoreMap sentence, Tree node) {
		//node is null if it cannot be found in the sentence's parse tree.
		if(node == null)
			return null;
		Tree head = node.headPreTerminal(new CollinsHeadFinder());
		//LogInfo.logs("Finding head - " + node + ":" + head + ":" + head.getSpan());
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
	    for(IndexedWord word : graph.getAllNodesByWordPattern(".*")) {
	      //LogInfo.logs(word.value() + "--" + word.index());
	      if(word.index() - 1 == head.getSpan().getSource() && word.index() - 1 == head.getSpan().getTarget()) {
	        //LogInfo.logs("Found - " + word.value());
	        return word;
	      }
	    }
	    //In case head span was not found, return the word in next span. For instance event 'in order' in p27.txt
	    if(node.getSpan().getTarget()-node.getSpan().getSource() > 0) {
	    	for(IndexedWord word : graph.getAllNodesByWordPattern(".*")) {
		      if(word.index() - 1 == head.getSpan().getSource() + 1 && word.index() - 1 == head.getSpan().getTarget() + 1) {
			        //LogInfo.logs("Found - " + word.value());
			        return word;
		      }
		    }
	    }
	    return null;
	}
	
	public static int findDepthInDependencyTree(CoreMap sentence, Tree node) {
		IndexedWord word = findDependencyNode(sentence, node);
		if(word == null)
			return -1;
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		return graph.getShortestDirectedPathEdges(graph.getFirstRoot(), word).size();
	}
	
	public static boolean isNodesRelated(CoreMap sentence, Tree entity, Tree event) {
		//Event ideally wouldn't be parent of full sentence. But, it will tag it as parent because the full sentence has the trigger
		//word as head token.
		if(entity == null || entity.value().equals("S") || entity.value().equals("SBAR"))
			return false;
		IndexedWord entityIndexWord = findDependencyNode(sentence, entity);
		//In case of punctuation marks, there is no head found.
		if(entityIndexWord != null) {
			SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
			//LogInfo.logs("Event == " + event.getTreeNode() + ":" + event.getHeadInDependencyTree());
			IndexedWord word = findDependencyNode(sentence, event);
			if(word == null) return false;
			for(IndexedWord w:graph.getChildList(word)) {
				if(w.equals(entityIndexWord)) {
					//LogInfo.logs(String.format("%s is direct parent of %s in dependency tree", event.getTreeNode(), entity));
					return true;
				}
			}
		}
		//LogInfo.logs(String.format("Don't think %s is parent of %s in dependency tree", event.getTreeNode(), entity));
		
		return false;
	}
	
	public static String getDependencyPath(CoreMap sentence, Tree entity, Tree event) {
		//word as head token.
		if(entity == null || entity.value().equals("S") || entity.value().equals("SBAR"))
			return "";
		IndexedWord entityIndexWord = findDependencyNode(sentence, entity);
		//In case of punctuation marks, there is no head found.
		StringBuilder buf = new StringBuilder();
		if(entityIndexWord != null) {
			SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
			//LogInfo.logs("Event == " + event.getTreeNode() + ":" + event.getHeadInDependencyTree());
			IndexedWord word = findDependencyNode(sentence, event);
			if(word == null) return "";
			if(graph.getShortestDirectedPathEdges(word, entityIndexWord) != null)
				return graph.getShortestDirectedPathEdges(word, entityIndexWord).toString();
		}
		//LogInfo.logs(String.format("Don't think %s is parent of %s in dependency tree", event.getTreeNode(), entity));
		
		return buf.toString();
	}
	
	public static boolean isNodesRelated(CoreMap sentence, Tree entity, EventMention event) {
		//Event ideally wouldn't be parent of full sentence. But, it will tag it as parent because the full sentence has the trigger
		//word as head token.
		if(entity == null || entity.value().equals("S") || entity.value().equals("SBAR"))
			return false;
		IndexedWord entityIndexWord = findDependencyNode(sentence, entity);
		//In case of punctuation marks, there is no head found.
		if(entityIndexWord != null) {
			SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
			//LogInfo.logs("Event == " + event.getTreeNode() + ":" + event.getHeadInDependencyTree());
			for(IndexedWord w:graph.getChildList(event.getHeadInDependencyTree())) {
				if(w.equals(entityIndexWord)) {
					//LogInfo.logs(String.format("%s is direct parent of %s in dependency tree", event.getTreeNode(), entity));
					return true;
				}
			}
		}
		//LogInfo.logs(String.format("Don't think %s is parent of %s in dependency tree", event.getTreeNode(), entity));
		
		return false;
	}
	
    public static String getText(Tree tree) {
    	StringBuilder b = new StringBuilder();
    	for(Tree leaf:tree.getLeaves()) {
    		b.append(leaf.value() + " ");
    	}
    	return b.toString().trim();
    }
    
    public static Set<String> getNominalizedVerbsLong() {
    	Set<String> nominalization = new HashSet<String>();
    	try {
			BufferedReader reader = new BufferedReader(new FileReader("derivation.txt"));
			String line;
			while((line = reader.readLine()) != null){
				String[] splits = line.split("\t");
				if(splits[1].equals("n") && splits[3].equals("v")) {
					nominalization.add(splits[0].trim());
				}
				else if(splits[3].equals("n") && splits[1].equals("v")) {
					nominalization.add(splits[2].trim());
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	//LogInfo.logs(nominalization);
    	return nominalization;
    }
    
    public static Set<String> getNominalizedVerbs() {
    	Set<String> nominalization = new HashSet<String>();
    	try {
			BufferedReader reader = new BufferedReader(new FileReader("nomlex.txt"));
			String line;
			while((line = reader.readLine()) != null){
				if(!line.startsWith("(NOM :ORTH "))
					continue;
				nominalization.add(line.replace("(NOM :ORTH ", "").replace("\"", ""));
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	//LogInfo.logs(nominalization);
    	return nominalization;
    }
    
    public static HashMap<String, String> getVerbForms() {
    	HashMap<String, String> nominalization = new HashMap<String, String>();
    	try {
			BufferedReader reader = new BufferedReader(new FileReader("derivation.txt"));
			String line;
			while((line = reader.readLine()) != null){
				String[] splits = line.split("\t");
				if(splits[1].equals("n") && splits[3].equals("v")) {
					nominalization.put(splits[0].trim(), splits[2].trim());
				}
				else if(splits[3].equals("n") && splits[1].equals("v")) {
					nominalization.put(splits[2].trim(), splits[0].trim());
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	//LogInfo.logs(nominalization);
    	return nominalization;
    }

	public static Set<Tree> getEventNodesForSentenceFromDatum(List<BioDatum> prediction, CoreMap sentence) {
		IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
		for(BioDatum d:prediction)
			if(d.guessLabel.equals("E") && d.sentence.equals(sentence))
				set.add(d.eventNode);
		return set;
	}

	public static Set<Tree> getEntityNodesForSentenceFromDatum(List<BioDatum> prediction, CoreMap sentence) {
		IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
		for(BioDatum d:prediction)
			if(d.guessLabel.equals("E") && d.sentence.equals(sentence))
				set.add(d.entityNode);
		return set;
	}

	public static boolean stringObjectContains(Set<Object> keySet, String srl) {
		for (Object key : keySet) {
			if (((String)key).equals(srl)) {
				return true;
			}
		}
		return false;
	}
	
	public static List<Pair<String, Double>> rankRoleProbs(Counter<String> probs, Index<String> labelIndex) {
		List<Pair<String, Double>> roleProbPairList = new ArrayList<Pair<String, Double>>();
//		for (int i=0; i<probs.length; i++) {
//			roleProbPairList.add(new Pair<String, Double>((String) labelIndex.get(i), probs[i]));
//		}
		for (String role : probs.keySet()) {
			roleProbPairList.add(new Pair<String, Double>(role, probs.getCount(role)));
		}
		Collections.sort(roleProbPairList, new PairComparatorByDouble());
		return roleProbPairList;
	}

	public static void mergeMaps(IdentityHashMap<Tree, String> bigMap, IdentityHashMap<Tree, String> smallMap) {
		for (Tree t : smallMap.keySet()) {
//			if (bigMap.containsKey(t)) {
//				System.out.println("SCREAMING !!!");
//			} else {
				bigMap.put(t, smallMap.get(t));
//			}
		}
	}
	
	public static HashMap<String, Integer> loadClustering() {
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		try {
			BufferedReader reader = new BufferedReader(new FileReader("word_cluster_clark_m_5_i_20_cl_25"));
			String line;
			while((line = reader.readLine()) != null){
				String[] splits = line.trim().split(" ");
				if(splits.length == 3)
					map.put(splits[0].trim(), Integer.parseInt(splits[1].trim()));
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	return map;
		
	}

	public static boolean isEventNextInOrder(List<EventMention> mentions, EventMention event1, EventMention event2) {
		int index1 = mentions.indexOf(event1), index2 = mentions.indexOf(event2);
		if(index1 + 1 == index2)
			return true;
		return false;
	}
	
	public static IntCounter<RelationType> findEventRelationDistribution(List<Example> examples){
		IntCounter<RelationType> counter = new IntCounter<RelationType>();
		for(Example example:examples) {
			int numRelations = 0, numEvents = example.gold.get(EventMentionsAnnotation.class).size();
			for(EventMention evt:example.gold.get(EventMentionsAnnotation.class)) {
				for(ArgumentRelation rel:evt.getArguments()) {
					  if(rel.mention instanceof EventMention) { 
						  counter.incrementCount(rel.type);
						  numRelations ++;
					  }
				}
			}
			int numNONE = numEvents * (numEvents - 1) / 2 - numRelations;
			counter.incrementCount(RelationType.NONE, numNONE);
		}
		return counter;
	}

	public static List<String> findWordsInBetween(Example example,
			EventMention event1, EventMention event2) {
		// TODO Auto-generated method stub
		List<String> words = new ArrayList<String>();
		boolean beginGettingWords = false;
		for(CoreMap sentence:example.gold.get(SentencesAnnotation.class)){
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				if(node.isPreTerminal()) {
					if(node == event1.getTreeNode()) {
						beginGettingWords = true;
					}
					else if(node==event2.getTreeNode()){
						beginGettingWords=false;
					}
					else if(beginGettingWords) {
						//System.out.println(getText(node));
						words.add(getText(node));
					}
				}
			}
		}
		return words;
	}
	public static Pair<Integer, Integer> findNumberOfSentencesAndWordsBetween(Example example,
			EventMention event1, EventMention event2) {
		// TODO Auto-generated method stub
		int sentenceCount = 0, wordCount = 0;
		boolean beginGettingWords = false;
		for(CoreMap sentence:example.gold.get(SentencesAnnotation.class)){
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				if(node.isPreTerminal()) {
					if(node == event1.getTreeNode()) {
						beginGettingWords = true;
					}
					else if(node==event2.getTreeNode()){
						beginGettingWords=false;
					}
					else{
						wordCount++;
					}
				}
			}
			if(beginGettingWords)
				sentenceCount++;
		}
		return new Pair<Integer, Integer>(sentenceCount, wordCount);
	}

	public static String findWordBefore(EventMention event) {
		CoreLabel wordBefore = Utils.findCoreLabelFromTree(event.getSentence(), event.getTreeNode());
		List<CoreLabel> tokens = event.getSentence().get(TokensAnnotation.class);
		
		if(wordBefore.index() >0)
			return tokens.get(wordBefore.index() - 1).originalText();
		return null;
	}
	
	public static String findWordAfter(EventMention event) {
		CoreLabel wordBefore = Utils.findCoreLabelFromTree(event.getSentence(), event.getTreeNode());
		List<CoreLabel> tokens = event.getSentence().get(TokensAnnotation.class);
		
		if(wordBefore.index() < tokens.size() - 1)
			return tokens.get(wordBefore.index() + 1).originalText();
		return null;
	}
	
	public static void printConfusionMatrix(double[][] confusionMatrix, List<String> relations, String fileName) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
			StringBuilder builder = new StringBuilder();
			for(String relation:relations)
				builder.append("," + relation);

			for(int i = 0; i < relations.size(); i++) {
				builder.append("\n" + relations.get(i) );
				for(int j = 0; j < relations.size(); j++) {
					builder.append("," + confusionMatrix[i][j]);
				}
			}
			writer.write(builder.toString());
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}


