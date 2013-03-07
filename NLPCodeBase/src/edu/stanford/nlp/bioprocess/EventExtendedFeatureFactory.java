package edu.stanford.nlp.bioprocess;

import java.util.*;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.StringUtils;

public class EventExtendedFeatureFactory extends FeatureExtractor {
	boolean printDebug = false, printAnnotations = false, printFeatures = false;
	Set<String> nominalizations = Utils.getNominalizedVerbs();
   
	public FeatureVector computeFeatures(CoreMap sentence, String tokenClass, Tree event) {
	    //System.out.println("Current node's text - " + getText(event));
		
		List<String> features = new ArrayList<String>();
		String currentWord = event.value();
		List<Tree> leaves = event.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
		List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
		
		features.add("POS="+currentWord);
		features.add("lemma="+token.lemma());
		features.add("word=" + leaves.get(0));
		features.add("POSparentPOS="+currentWord + "," + event.parent(root).value());
		features.add("POSlemma=" + currentWord+","+token.lemma());
		features.add("path=" + Trees.pathFromRoot(event, root));
		
		//Trying to look at entities to improve prediction
		String shortestPath = "";
		int pathLength = Integer.MAX_VALUE, numRelatedEntities = 0;
		for(EntityMention m:sentence.get(EntityMentionsAnnotation.class)) {
			if(Utils.isNodesRelated(sentence, m.getTreeNode(), event)) {
				numRelatedEntities++;
				List<String> nodesInPath = Trees.pathNodeToNode(event, m.getTreeNode(), root);
				if(pathLength > nodesInPath.size()) {
					shortestPath = StringUtils.join(nodesInPath, ",");
				}
				//System.out.println(shortestPath);
			}
		}
		
		if(token.index() > 0) {
			features.add("POStokenbefore=" + currentWord + "," + tokens.get(token.index()-1).originalText());
			//features.add("tokenbefore=" + tokens.get(token.index()-1).originalText());
		}
		//features.add("numrelatedentities" + (numRelatedEntities > 1));
		if(pathLength<Integer.MAX_VALUE) {
			features.add("splpath=" + shortestPath);
			features.add("splpathlength=" + pathLength);
		}
		
		//Nominalization did not give much improvement
		/*if(nominalizations.contains(leaves.get(0).value())) {
			//System.out.println("Adding nominalization - " + leaves.get(0));
			//features.add("nominalization");
		}*/
		//features.add("endsining=" + token.lemma() + "," + leaves.get(0).value().endsWith("ing"));
		//Cannot use this feature when looking at all tree nodes as candidates
			//features.add("POSparentPOSgrandparent="+currentWord + "," + event.parent(root).value() + "," + event.parent(root).parent(root).value());
		//Doesn't seem to work as expected even though the event triggers are mostly close to root in dependency tree.
		//features.add("POSdepdepth=" + currentWord + "," + Utils.findDepthInDependencyTree(sentence, event));
		
		String classString = "class=" + tokenClass + ",";
		List<String> updatedFeatures = new ArrayList<String>();
		for(String feature:features)
			updatedFeatures.add(classString + feature);
	
		FeatureVector fv = new FeatureVector(updatedFeatures);
		return fv;
    }

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
						if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
								!(node.value().equals("NN") || node.value().equals("JJ") || node.value().startsWith("VB")))
							continue;
						
						String type = "O";
						
						//if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
							//type = "E";
						//}
						if (eventNodes.contains(node)){
							type = "E";
						}
						//if(printDebug) System.out.println(type + " : " + node + ":" + node.getSpan());
						Datum newDatum = new Datum(Utils.getText(node), type, node, node);
						newDatum.features = computeFeatures(sentence, type, node);
						if(printFeatures) System.out.println(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
						newData.add(newDatum);
					//}
				}
			}
			if(printDebug) System.out.println("\n------------------------------------------------");
		}
	
		return newData;
    }
    
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
					if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
							!(node.value().startsWith("NN") || node.value().equals("JJ") || node.value().startsWith("VB")))
						continue;
					
					String type = "O";
					
					if (eventNodes.contains(node)) {
						type = "E";
					}
					
					Datum newDatum = new Datum(Utils.getText(node), type, node, node);
					newDatum.features = computeFeatures(sentence, possibleLabel, node);
					newData.add(newDatum);
					//prevLabel = newDatum.label;
				}
		    //}
		}
    	return newData;
    }
}
