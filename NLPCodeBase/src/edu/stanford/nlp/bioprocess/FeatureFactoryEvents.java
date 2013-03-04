package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.StringUtils;

public class FeatureFactoryEvents {
	boolean printDebug = true, printAnnotations = true, printFeatures = false;
    /** Add any necessary initialization steps for your features here.
     *  Using this constructor is optional. Depending on your
     *  features, you may not need to initialize anything.
     */
    public FeatureFactoryEvents() {

    }

    /**
     * Words is a list of the words in the entire corpus, previousLabel is the label
     * for position-1 (or O if it's the start of a new sentence), and position
     * is the word you are adding features for. PreviousLabel must be the
     * only label that is visible to this method. 
     */
 private List<String> computeFeatures(CoreMap sentence, String tokenClass, Tree event) {
    //System.out.println("Current node's text - " + getText(event));
	List<String> features = new ArrayList<String>();
	String currentWord = event.value();
	List<Tree> leaves = event.getLeaves();
	Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
	//System.out.println("value of event is "+event.value());
	//root.pennPrint();
	//System.out.println(event);
	//System.out.println(entity);
	//System.out.println(Trees.getLowestCommonAncestor(entity, event, root));
	//System.out.println(node.depth() +":" +root.depth());
	//IntPair ip = node.getSpan();
	//Span span = new Span(ip.getSource(), ip.getTarget() + 1);
	
	//CoreLabel headToken = sentence.get(TokensAnnotation.class).get(Utils.findHeadWord(sentence, span).start());
	
	//Features of entity
	features.add("POS="+currentWord);
	features.add("firstchild=" + leaves.get(0));
	//features.add("lastchild=" + leaves.get(leaves.size()-1));
	//features.add("numleaves=" + leaves.size());
	//features.add("headword=" + entity.headTerminal(new CollinsHeadFinder()));
	//features.add("headwordPOS=" + entity.headPreTerminal(new CollinsHeadFinder()).value());

	//features.add("parentPOS=" + entity.parent(root).value());
	//features.add("parentheadword=" + entity.parent(root).headTerminal(new CollinsHeadFinder()));
	//features.add("parentheadwordPOS=" + entity.parent(root).headPreTerminal(new CollinsHeadFinder()).value());
	
	//features.add("firstchild=" + leaves.get(0)+",lastchild=" + leaves.get(leaves.size()-1)+","+event.preTerminalYield().get(0).value());
	
//	//features.add("entWordevtPOS=" + entity.value() + "," + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());
	//features.add("entPOSevtPOS=" + entity.value() + "," + event.preTerminalYield().get(0).value());
	//if(entity.value().startsWith("N") && StringUtils.join(Trees.pathNodeToNode(event, root, root), " ").contains("VP"));
	//	features.add("NPEntityAndVPParentForTrigger");
	//features.add("combtrigger=" + entity.headTerminal(new CollinsHeadFinder()) + "," + event.getLeaves().get(0).value());
	
	//Features of event
	//features.add("eventPOS=" + event.value());
	//features.add("eventhead=" + event.preTerminalYield().get(0).value()); 
	//features.add("path=" + Utils.getPathString(Trees.pathNodeToNode(entity, event, Trees.getLowestCommonAncestor(entity, event, root))));
	//features.add("position=" + (event.getSpan().getSource() > entity.getSpan().getSource() ? "before" : "after"));

	//features.add("height=" +  event.depth());
	//features.add("isDT=" + node.value().equals("DT"));
	
	//features.add("headword=" + headToken.originalText());
	
	//features.add("noun=" + (token.get(PartOfSpeechAnnotation.class).startsWith("NN") ? 1 : 0));
	//if(token.index() > 1) {
	//	CoreLabel prev = sentence.get(TokensAnnotation.class).get(token.index() - 2);
		//features.add("prevword=" + prev.originalText());
	//	features.add("prevpos=" + prev.get(PartOfSpeechAnnotation.class));
	//}
	//features.add("trueCase=" + entity.get(TrueCaseAnnotation.class));
	//features.add("ner=" + entity.get(NamedEntityTagAnnotation.class));
	//features.add("role=" + entity.get(RoleAnnotation.class));
	//features.add("stem=" + entity.get(StemAnnotation.class));
	//features.add("prevLabel=" + previousLabel);
	//features.add("word=" + currentWord + ", prevLabel=" + previousLabel);
	String classString = "class=" + tokenClass + ",";
	List<String> updatedFeatures = new ArrayList<String>();
	for(String feature:features)
		updatedFeatures.add(classString + feature);
	
	/** Warning: If you encounter "line search failure" error when
	 *  running the program, considering putting the baseline features
	 *  back. It occurs when the features are too sparse. Once you have
	 *  added enough features, take out the features that you don't need. 
	 */

	return updatedFeatures;
    }

    /** Do not modify this method **/
public List<Datum> setFeaturesTrain(List<Example> data) {
	List<Datum> newData = new ArrayList<Datum>();
	
	for (Example ex : data) {
		if(printDebug || printAnnotations) System.out.println("\n-------------------- " + ex.id + "---------------------");
		for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
			IdentityHashSet<Tree> eventNodes = Utils.getEventNodesFromSentence(sentence);
			if(printDebug) System.out.println(sentence);
			if(printAnnotations) {
				System.out.println("---Events--");
				for(EventMention event: sentence.get(EventMentionsAnnotation.class))
					System.out.println(event.getValue());
				//System.out.println("---Entities--");
				//for(EntityMention entity: sentence.get(EntityMentionsAnnotation.class))
					//System.out.println(entity.getTreeNode() + ":" + entity.getTreeNode().getSpan());
			}
			//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
				//if(printDebug) System.out.println("-------Event - " + event.getTreeNode()+ "--------");
				for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
					if(node.isLeaf()||node.value().equals("ROOT"))
						continue;
					
					String type = "O";
					
					//if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
						//type = "E";
					//}
					if (eventNodes.contains(node)){
						type = "E";
					}
					//if(printDebug) System.out.println(type + " : " + node + ":" + node.getSpan());
					Datum newDatum = new Datum(getText(node), type, node, node);
					newDatum.features = computeFeatures(sentence, type, node);
					if(printFeatures) System.out.println(getText(node) + ":" + newDatum.features);
					newData.add(newDatum);
				//}
			}
		}
		if(printDebug) System.out.println("\n------------------------------------------------");
	}

	return newData;
    }
    
    /** Do not modify this method **/
    /*
	public List<Datum> setFeaturesTest(List<Example> data) {
		// this is so that the feature factory code doesn't accidentally use the
		// true label info
		List<Datum> newData = new ArrayList<Datum>();
	
		// compute features for all possible previous labels in advance for
		// Viterbi algorithm
		for (Example ex : data) {
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				newData.addAll(setFeaturesTest(sentence));
			}
		}
	
		return newData;
    }*/
    
    public List<Datum> setFeaturesTest(CoreMap sentence) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<Datum> newData = new ArrayList<Datum>();
    	List<String> labels = new ArrayList<String>();
    	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

    	labelIndex.put("O", 0);
    	labelIndex.put("E", 1);
    	labels.add("O");
    	labels.add("E");


    	IdentityHashSet<Tree> eventNodes = Utils.getEventNodesFromSentence(sentence);
		//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				for (String possibleLabel : labels) {
					if(node.isLeaf() || node.value().equals("ROOT"))
						continue;
					
					String type = "O";
					
					if (eventNodes.contains(node)) {
						type = "E";
					}
					
					Datum newDatum = new Datum(getText(node), type, node, node);
					newDatum.features = computeFeatures(sentence, possibleLabel, node);
					newData.add(newDatum);
					//prevLabel = newDatum.label;
				}
		    //}
		}
    	return newData;
    }
    
    private String getText(Tree tree) {
    	StringBuilder b = new StringBuilder();
    	for(Tree leaf:tree.getLeaves()) {
    		b.append(leaf.value() + " ");
    	}
    	return b.toString().trim();
    }
    
    private List<CoreLabel> getEntityTokens(Example ex) {
    	List<CoreLabel> lst = new ArrayList<CoreLabel>();
    	for(EntityMention entity : ex.gold.get(EntityMentionsAnnotation.class))
    		lst.add(entity.getHeadToken());
    	return lst;
    }
}