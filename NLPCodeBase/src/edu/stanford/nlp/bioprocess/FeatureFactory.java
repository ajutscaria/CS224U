package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;

public class FeatureFactory {

    /** Add any necessary initialization steps for your features here.
     *  Using this constructor is optional. Depending on your
     *  features, you may not need to intialize anything.
     */
    public FeatureFactory() {

    }

    /**
     * Words is a list of the words in the entire corpus, previousLabel is the label
     * for position-1 (or O if it's the start of a new sentence), and position
     * is the word you are adding features for. PreviousLabel must be the
     * only label that is visible to this method. 
     */
    private List<String> computeFeatures(CoreMap sentence, Tree node, String tokenClass) {
    //node.pennPrint();
    //	System.out.println("Current node's text - " + getText(node));
    	
	List<String> features = new ArrayList<String>();

	String currentWord = node.value();
	List<Tree> leaves = node.getLeaves();
	Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
	//System.out.println(node.depth() +":" +root.depth());
	IntPair ip = node.getSpan();
	Span span = new Span(ip.getSource(), ip.getTarget() + 1);
	
	//CoreLabel headToken = sentence.get(TokensAnnotation.class).get(Utils.findHeadWord(sentence, span).start());
	
	// Baseline Features 
	features.add("POS=" + currentWord);
	//features.add("text=" + getText(node));
	features.add("firstchild=" + leaves.get(0));
	features.add("lastchild=" + leaves.get(leaves.size()-1));
	features.add("numleaves=" + leaves.size());
	features.add("headword=" + node.headTerminal(new CollinsHeadFinder()));
	features.add("headwordPOS=" + node.headPreTerminal(new CollinsHeadFinder()).value());

	features.add("parentPOS=" + node.parent(root).value());
	features.add("parentheadword=" + node.parent(root).headTerminal(new CollinsHeadFinder()));
	features.add("parentheadwordPOS=" + node.parent(root).headPreTerminal(new CollinsHeadFinder()).value());

	//features.add("height=" +  node.depth());
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
		List<Tree> entityNodes = Utils.getEntityNodes(ex);
		for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				if(node.isLeaf()||node.value().equals("ROOT"))
					continue;
				String type = "O";
				//System.out.println(ex.id + ":" +entityNodes.size());
				//if((!node.value().equals("DT")) && (entityNodes.contains(node) || Utils.isChildOfEntity(entityNodes, node))) {
				if((entityNodes.contains(node) || Utils.isChildOfEntity(entityNodes, node))) {
					type = "E";
				}
				
				Datum newDatum = new Datum(getText(node), type, node);
				newDatum.features = computeFeatures(sentence, node, type);
				//System.out.println(getText(node) + ":" + newDatum.features);
				newData.add(newDatum);
			}
		}
	}

	return newData;
    }
    
    /** Do not modify this method **/
    public List<Datum> setFeaturesTest(List<Example> data) {
	// this is so that the feature factory code doesn't accidentally use the
	// true label info
	List<Datum> newData = new ArrayList<Datum>();
	List<String> labels = new ArrayList<String>();
	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

	labelIndex.put("O", 0);
	labelIndex.put("E", 1);
	labels.add("O");
	labels.add("E");

	// compute features for all possible previous labels in advance for
	// Viterbi algorithm
	for (Example ex : data) {
		List<Tree> entityNodes = Utils.getEntityNodes(ex);
		for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				for (String possibleLabel : labels) {
					if(node.isLeaf() || node.value().equals("ROOT"))
						continue;
					String type = "O";
					if(entityNodes.contains(node)) {
						type = "E";
						}
					
					Datum newDatum = new Datum(getText(node), type, node);
					newDatum.features = computeFeatures(sentence, node, possibleLabel);
					newData.add(newDatum);
					//prevLabel = newDatum.label;
				}
		    }
		}

	}

	return newData;
	
    }
    
    public List<Datum> setFeaturesTest(CoreMap sentence, List<Tree> entityNodes) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<Datum> newData = new ArrayList<Datum>();
    	List<String> labels = new ArrayList<String>();
    	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

    	labelIndex.put("O", 0);
    	labelIndex.put("E", 1);
    	labels.add("O");
    	labels.add("E");


		for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
			for (String possibleLabel : labels) {
				if(node.isLeaf()|| node.value().equals("ROOT"))
					continue;
				String type = "O";
				if(entityNodes.contains(node)) {
					type = "E";
					}
				
				Datum newDatum = new Datum(getText(node), type, node);
				newDatum.features = computeFeatures(sentence, node, possibleLabel);
				newData.add(newDatum);
				//prevLabel = newDatum.label;
			}
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
