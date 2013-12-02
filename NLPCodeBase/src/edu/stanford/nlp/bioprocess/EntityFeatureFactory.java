package edu.stanford.nlp.bioprocess;


import java.util.*;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import fig.basic.LogInfo;

public class EntityFeatureFactory extends FeatureExtractor {
	public EntityFeatureFactory(boolean useLexicalFeatures) {
		super(useLexicalFeatures);
	}

	public static Integer globalcounter = 0;
	public static double goldEntities = 0;
	public static double coveredEntities = 0;
	public static boolean test = false;
	boolean printDebug = false, printAnnotations = false, printFeatures = false;

	public FeatureVector computeFeatures(CoreMap sentence, Tree entity,  Tree event) {
		//Tree event = eventMention.getTreeNode();
		List<String> features = new ArrayList<String>();
		//List<Tree> leaves = entity.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		boolean dependencyExists = Utils.isNodesRelated(sentence, entity, event);
		String depPath = Utils.getDependencyPath(sentence, entity, event);
		Tree parent = entity.parent(root);
		String parentCFGRule = parent.value() + "->";
		for(Tree n:parent.getChildrenAsList()) {
			parentCFGRule += n.value() + "|";
		}
		parentCFGRule = parentCFGRule.trim();
		int containS = 0;
		for(Tree node:entity.getChildrenAsList()){
			if(node.value().equals("S") || node.value().equals("SBAR")){
				containS = 1;
				break;
			}
		}

		//features.add("dep="+dependencyExists);
		//features.add("EntCat="+entity.value());
		//features.add("EntHead=" + entity.headTerminal(new CollinsHeadFinder()));
		features.add("EntContainsS="+containS);
		features.add("EvtLemma="+event.getLeaves().get(0).value());
		features.add("EntCatDepRel=" + entity.value() + ","  + dependencyExists);
		features.add("EntHeadEvtPOS="+Utils.findCoreLabelFromTree(sentence, entity).lemma() + "," + event.preTerminalYield().get(0).value());
		features.add("EvtToEntDepPath=" + ((depPath.equals("")||depPath.equals("[]")) ? 0 :depPath.split(",").length));
		features.add("EntHeadEvtHead=" + entity.headTerminal(new CollinsHeadFinder()) + "," + event.getLeaves().get(0));		
		//features.add("PathEntToEvt=" + Trees.pathNodeToNode(event, entity, root));
		features.add("EntNPAndRelatedToEvt=" + (entity.value().equals("NP") && Utils.isNodesRelated(sentence, entity, event)));
		//features.add("parentrule=" + parentCFGRule);
		//USE LEMMA everywhere.
		//Try before/after

		//features.add("EntPOSEntHeadEvtPOS=" + entity.value() + "," + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());
		//features.add("EntPOSEvtPOSDepRel=" + entity.value() + "," +event.preTerminalYield().get(0).value() + ","  + dependencyExists);
		//features.add("EntPOSEntParentPOSEvtPOS=" + entity.value() + "," + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());
		//features.add("EntLastWordEvtPOS="+leaves.get(leaves.size()-1)+","+event.preTerminalYield().get(0).value());
		//features.add("PathEntToAncestor="+Trees.pathNodeToNode(entity, Trees.getLowestCommonAncestor(entity, event, root), root));
		//features.add("PathEntToRoot="+Trees.pathNodeToNode(entity, root, root));
		//features.add("EntParentPOSEvtPOS=" + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());



		//This feature did not work surprisingly. Maybe because the path from ancestor to event might lead to a lot of different variations.
		//features.add("PathAncestorToEvt="+Trees.pathNodeToNode(Trees.getLowestCommonAncestor(entity, event, root), event, root));
		//This is a bad feature too.
		//features.add("EvtPOSDepRel=" + event.preTerminalYield().get(0).value() + ","  + dependencyExists);
		//Not a good feature too.
		//features.add("EntPOSEvtPOS=" + entity.value() + "," + event.preTerminalYield().get(0).value());
		features.add("bias");
		
		if(Main.printFeature && Example.examplePrint){
			LogInfo.begin_track("Features of %s <- %s", event.toString(), entity.toString());
			for(String f:features){
				LogInfo.logs(f);
			}
			LogInfo.end_track();
		}
		
		FeatureVector fv = new FeatureVector(features);
		return fv;
	}

	public List<BioDatum> setFeaturesTrain(List<Example> data) {
		List<BioDatum> newData = new ArrayList<BioDatum>();

		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
				if(printDebug){
					LogInfo.logs(sentence);
					sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
				}
				if(printAnnotations) {
					LogInfo.logs("---Events--");
					for(EventMention event: sentence.get(EventMentionsAnnotation.class))
						LogInfo.logs(event.getValue());
					LogInfo.logs("---Entities--");
					for(EntityMention entity: sentence.get(EntityMentionsAnnotation.class)) {
						if(entity.getTreeNode() != null)
							LogInfo.logs(entity.getTreeNode() + ":" + entity.getTreeNode().getSpan());
						else
							LogInfo.logs("Couldn't find node:" + entity.getValue());
					}
				}
				//SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
				//LogInfo.logs(dependencies);

				for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
					if(printDebug) LogInfo.logs("-------Event - " + event.getTreeNode()+ "--------");
					for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
						if(node.isLeaf()||node.value().equals("ROOT"))
							continue;

						String type = "O";

						if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
							type = "E";
						}
						if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
						//					if((entityNodes.contains(node))){// || (Utils.isChildOfEntity(entityNodes, node) && node.value().startsWith("NN"))) {
						//						type = "E";
						//					}

						BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, event.getTreeNode(), Utils.getArgumentMentionRelation(event, node).toString());
						newDatum.features = computeFeatures(sentence, node, event.getTreeNode());
						if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features);
						newData.add(newDatum);
					}
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
		}

		return newData;
	}


	public List<BioDatum> setFeaturesTrain(List<Example> data, Params parameters) {

		Counter<String> labelCounts = new ClassicCounter<String>();
		List<BioDatum> newData = new ArrayList<BioDatum>();
		EventFeatureFactory eventFeatureFactory = new EventFeatureFactory(true);
		LinearClassifier<String, String> classifier = new LinearClassifier<String, String>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
		double theta = Main.theta;
		int eventcount = 0;
		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
				IdentityHashMap<Tree, EventMention> goldeventmentions = Utils.getEventNodeAndMentionFromSentence(sentence);

				if(printDebug){
					LogInfo.logs(sentence);
					sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
				}
				if(printAnnotations) {
					LogInfo.logs("---Events--");
					for(EventMention event: sentence.get(EventMentionsAnnotation.class))
						LogInfo.logs(event.getValue());
					LogInfo.logs("---Entities--");
					for(EntityMention entity: sentence.get(EntityMentionsAnnotation.class)) {
						if(entity.getTreeNode() != null)
							LogInfo.logs(entity.getTreeNode() + ":" + entity.getTreeNode().getSpan());
						else
							LogInfo.logs("Couldn't find node:" + entity.getValue());
					}
				}

				//Integer traincounter = 0;
				List<BioDatum> dataforsentence = eventFeatureFactory.setFeaturesTest(sentence, Utils.getEntityNodesFromSentence(sentence), ex.id);
				for(BioDatum d:dataforsentence) {
					Datum<String, String> eventDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
					double scoreE = classifier.scoreOf(eventDatum, "E"), scoreO = classifier.scoreOf(eventDatum, "O");
					scoreE = (Math.exp(scoreE)/(Math.exp(scoreE) + Math.exp(scoreO)));
					if(scoreE < theta)continue;
					//System.out.println(scoreE);
					EventMention event;
					if(goldeventmentions.containsKey(d.eventNode)){
						event = goldeventmentions.get(d.eventNode);
					}else{
						event = new EventMention(globalcounter.toString(), sentence, null);
						event.setTreeNode(d.eventNode);
						globalcounter++;
					}

					eventcount++;

					//SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
					//LogInfo.logs(dependencies);

					/*for(BioDatum d: events){
					if(!d.exampleID.equals(ex.id))continue;
				    Tree eventNode = d.eventNode;
				    EventMention event = new EventMention(globalcounter.toString(), sentence, null);
					globalcounter++;
					event.setTreeNode(eventNode);*/
					//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
					//@heather
					/*for(Tree eventnode: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
					if(eventnode.isLeaf() || eventnode.value().equals("ROOT") || !eventnode.isPreTerminal() || 
							!(eventnode.value().startsWith("JJR") || eventnode.value().startsWith("JJS") ||eventnode.value().startsWith("NN") || eventnode.value().equals("JJ") || eventnode.value().startsWith("VB")))
						continue;
					EventMention event = new EventMention(globalcounter.toString(), sentence, null);
					globalcounter++;
					event.setTreeNode(eventnode);*/
					if(printDebug) LogInfo.logs("-------Event - " + event.getTreeNode()+ "--------");
				    List<Tree> candidates = pruning(d.eventNode, sentence);
					for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
					//for(Tree node: candidates) {	
						if(node.isLeaf()||node.value().equals("ROOT"))
							continue;
						

						String label = "O";

						if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
							label = "E";
							//System.out.println("E");
						}
						if(printDebug) LogInfo.logs(label + " : " + node + ":" + node.getSpan());
						//					if((entityNodes.contains(node))){// || (Utils.isChildOfEntity(entityNodes, node) && node.value().startsWith("NN"))) {
						//						type = "E";
						//					}

						BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), label, node, event.getTreeNode(), Utils.getArgumentMentionRelation(event, node).toString());
						newDatum.features = computeFeatures(sentence, node, event.getTreeNode());
						if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features);
						newData.add(newDatum);
						labelCounts.incrementCount(label);
					}
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
		}
		LogInfo.logs("Event training set label distribution=%s",labelCounts);
		System.out.println("event size for entities:"+eventcount);
		return newData;
	}
	
	public List<Tree> pruning(Tree event, CoreMap sentence){
		List<Tree> sisters = new ArrayList<Tree>();
		Tree current = event;
		if(!event.value().startsWith("VB")){
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)){
				sisters.add(node);
			}
			return sisters;
		}
		
		while(true){
			if(current.value().equals("ROOT"))break;
			//System.out.println("current:"+current.toString());
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)){
				if(node.getChildrenAsList().contains(current)){ //found parent
					//System.out.println("found parent:"+node.toString());
					boolean ccflag = false;
					for(Tree sis:node.getChildrenAsList()){
						if(sis.value().equals("CC")){
							ccflag = true;
						}
					}
					if(!ccflag){
						for(Tree sis:node.getChildrenAsList()){
							if(!sis.equals(current)){
							   
							   sisters.add(sis);
							   if(sis.value().equals("PP")){
								   //System.out.println("PP!!");
								   sisters.addAll(Arrays.asList(sis.children()));	
							   }
							}
						}
					}
					current = node;
					break;
				}
			}
		}
		/*while(true){
			if(current == null || current.value().equals("ROOT"))break;
			System.out.println("\ncurrent:"+current.toString());
			List<Tree> sisters = current.siblings(current);
			if(sisters!=null)
				for(Tree node:sisters){
					System.out.println("sister:"+node.toString());
					if(node.value().equals("CC")){
						sisters = null;
						break;
					}
				}
			if(sisters!=null){
				candidates.addAll(sisters);
				for(Tree node:sisters){
					if(node.value().equals("PP")){
						candidates.addAll(Arrays.asList(node.children()));
					}
				}
			}
			current = current.parent(current);
		}*/
		return sisters;
	}

	public List<BioDatum> setFeaturesTest(CoreMap sentence, Set<Tree> predictedEvents, String exampleID) {
		// this is so that the feature factory code doesn't accidentally use the
		// true label info

		//@heather replace gold predicted Events by all possible even nodes
		/*if(Main.mode.equalsIgnoreCase("allnew") || Main.runModel.equals("ilp")){
    		System.out.println("all event nodes!!");
    		predictedEvents = new HashSet<Tree>();
    		for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
    			if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
    					!(node.value().startsWith("JJR") || node.value().startsWith("JJS") ||node.value().startsWith("NN") || node.value().equals("JJ") || node.value().startsWith("VB")))
    				continue;
    			predictedEvents.add(node);
    		}
    	}
    	EntityPredictionInferer.eventSize+=predictedEvents.size();*/
		//

        test = true;
		List<BioDatum> newData = new ArrayList<BioDatum>();

		IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
		for(Tree eventNode: predictedEvents) {
			//int whichEvent = Main.EventID.get(eventNode);
			List<Tree> candidates = pruning(eventNode, sentence);
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				if(node.isLeaf() || node.value().equals("ROOT"))
					continue;
				String type = (entityNodes.contains(node) && Utils.getArgumentMentionRelation(sentence, eventNode, node) != RelationType.NONE) ? "E" : "O";
				if(type.equals("E")){
					goldEntities++;
				}
				
			}
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
			//for(Tree node:candidates){
				if(node.isLeaf() || node.value().equals("ROOT"))
					continue;
				String type = (entityNodes.contains(node) && Utils.getArgumentMentionRelation(sentence, eventNode, node) != RelationType.NONE) ? "E" : "O";
				if(type.equals("E")){
					coveredEntities++;
				}
				
				BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, eventNode, Utils.getArgumentMentionRelation(sentence, eventNode, node).toString());
				newDatum.features = computeFeatures(sentence, node, eventNode);
				newDatum.setExampleID(exampleID);
				//newDatum.eventId = whichEvent;
				newData.add(newDatum);

			}
		}
		return newData;
	}
}
