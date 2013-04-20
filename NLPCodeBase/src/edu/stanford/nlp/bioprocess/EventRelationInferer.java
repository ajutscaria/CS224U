package edu.stanford.nlp.bioprocess;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import fig.basic.LogInfo;

public class EventRelationInferer {
	List<BioDatum> prediction = null;
	private boolean printDebugInformation = false;
	
	public EventRelationInferer(List<BioDatum> predictions) {
		prediction = predictions;
	}
	
	public EventRelationInferer() {
	}

	public List<BioDatum> BaselineInfer(List<Example> examples, Params parameters, EventRelationFeatureFactory ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		for(Example example:examples) {
			for(EventMention evt:example.gold.get(EventMentionsAnnotation.class)) {
				for(ArgumentRelation rel:evt.getArguments()) {
					  if(rel.mention instanceof EventMention) { 
						  System.out.println("GOLD: " + evt.getTreeNode() + "-" + rel.mention.getTreeNode() + "-->" + rel.type);
					  }
				}
			}
			List<BioDatum> test = ff.setFeaturesTest(example, example.gold.get(EventMentionsAnnotation.class));
			
			for(BioDatum d:test) {
				//System.out.println(d.event1.getTreeNode() + "-" + d.event2.getTreeNode() + "-->" + d.label);
				List<EventMention> mentions = example.gold.get(EventMentionsAnnotation.class);
				if(Utils.isEventNextInOrder(mentions, d.event1, d.event2)) {
					d.guessLabel = "PreviousEvent";
				}
				else {
					d.guessLabel = "NONE";
				}
			}
			
			for(BioDatum d:test) {
				if(d.predictedLabel().equals(d.label()) && d.predictedLabel().equals("NONE"))
					continue;
				if(d.predictedLabel().equals(d.label)) {
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Correct", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else if(d.label().equals("NONE") && !d.predictedLabel().equals("NONE")){
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Extra", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else if(!d.label().equals("NONE") && d.predictedLabel().equals("NONE")){
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Missed", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else {
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Incorrect", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));					
				}
			}
			predicted.addAll(test);
		}
		return predicted;
	}

	
	public List<BioDatum> Infer(List<Example> testData, Params parameters, EventRelationFeatureFactory ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>();
		
		for(Example ex:testData) {
			LogInfo.begin_track("Example %s",ex.id);
				
			LinearClassifier<String, String> classifier = new LinearClassifier<>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
			List<BioDatum> dataset = ff.setFeaturesTest(ex, ex.gold.get(EventMentionsAnnotation.class));

			StringBuilder buffer = new StringBuilder("digraph finite_state_machine { \n\trankdir=LR;\n\tsize=\"50,50\";");
			int count = 0;
			List<EventMention> eventMentions = ex.gold.get(EventMentionsAnnotation.class);
			for(EventMention evtMention:ex.gold.get(EventMentionsAnnotation.class)) {
				buffer.append(String.format("\nnode%s [label = \"%s\"]", count++, Utils.getText(evtMention.getTreeNode())));
			}

			for(BioDatum d:dataset) {
				Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
				d.setPredictedLabel(classifier.classOf(newDatum));
				if(d.predictedLabel().equals(d.label()) && d.predictedLabel().equals("NONE"))
					continue;
				
				if(d.predictedLabel().equals(d.label)) {
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Correct", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else if(d.label().equals("NONE") && !d.predictedLabel().equals("NONE")){
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Extra", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else if(!d.label().equals("NONE") && d.predictedLabel().equals("NONE")){
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Missed", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else {
					LogInfo.logs(String.format("%-10s : %-10s - %-10s Gold:  %s Predicted: %s", "Incorrect", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				
				//dot -o file.png -Tpng file.gv
				if(!d.predictedLabel().equals("NONE")) {
					buffer.append(String.format("\n%s -> %s [ label = \"%s\" fontcolor=\"black\" %s color = \"%s\"];", "node"+eventMentions.indexOf(d.event1), "node"+eventMentions.indexOf(d.event2), d.predictedLabel(),
						//If Cotemporal or same event, put bi-directional edges
						(d.predictedLabel().equals("CotemporalEvent") || d.predictedLabel().equals("SameEvent")) ? "dir = \"both\"" : "", "Black")) ;
				}
				if(!d.label().equals("NONE")) {
					buffer.append(String.format("\n%s -> %s [ label = \"%s\" fontcolor=\"goldenrod3\" %s color = \"%s\"];", "node"+eventMentions.indexOf(d.event1), "node"+eventMentions.indexOf(d.event2), d.label(),
							//If Cotemporal or same event, put bi-directional edges
							(d.label().equals("CotemporalEvent") || d.label().equals("SameEvent")) ? "dir = \"both\"" : "", "goldenrod3")) ;
				}
			}
			buffer.append("\n}");
			predicted.addAll(dataset);
			Utils.writeStringToFile(buffer.toString(), "GraphViz/" + ex.id + ".gv");
		
			try {
	        	Runtime rt = Runtime.getRuntime();
	        	rt.exec("dot -o GraphViz/" + ex.id + ".pdf -Tpdf GraphViz/" + ex.id + ".gv");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.out.println("ERORRRR" + e.getMessage());
				e.printStackTrace();
			}
			
			LogInfo.end_track();
		}
		return predicted;
	}

}
