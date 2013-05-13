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
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;

public class EventRelationInferer {
	List<BioDatum> prediction = null;
	private boolean printDebugInformation = false;
	public IntCounter<String> countGoldTriples = new IntCounter<String>(), countPredictedTriples = new IntCounter<String>(); 
	public int totalEvents = 0;
	IntCounter<Integer> prevEvent = new IntCounter<Integer>(), superEvent = new IntCounter<Integer>(), causeEvent = new IntCounter<Integer>(), degreeDistribution = new IntCounter<Integer>();
	IntCounter<Integer> prevEventPred = new IntCounter<Integer>(), superEventPred = new IntCounter<Integer>(),  causeEventPred = new IntCounter<Integer>(), degreeDistributionPred = new IntCounter<Integer>();
	
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
			HashMap<String, Pair<String,String>> labelings = new HashMap<String, Pair<String,String>>();
			LogInfo.begin_track("Example %s",ex.id);
				
			LinearClassifier<String, String> classifier = new LinearClassifier<String, String>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
			List<BioDatum> dataset = ff.setFeaturesTest(ex, ex.gold.get(EventMentionsAnnotation.class));

			StringBuilder buffer = new StringBuilder("digraph finite_state_machine { \n\trankdir=LR;\n\tsize=\"50,50\";");
			int count = 0;
			List<EventMention> eventMentions = ex.gold.get(EventMentionsAnnotation.class);
			//System.out.println(eventMentions);
			
			for(EventMention evtMention:ex.gold.get(EventMentionsAnnotation.class)) {
				buffer.append(String.format("\nnode%s [label = \"%s\"]", count++, Utils.getText(evtMention.getTreeNode())));
				//System.out.println(evtMention.getTreeNode());
			}

			IntCounter<EventMention> dG = new IntCounter<EventMention>(), sG = new IntCounter<EventMention>(), pG = new IntCounter<EventMention>(), cG = new IntCounter<EventMention>();
			IntCounter<EventMention> dP = new IntCounter<EventMention>(), sP = new IntCounter<EventMention>(), pP = new IntCounter<EventMention>(), cP = new IntCounter<EventMention>();
			
			HashMap<String, Double> weights = new HashMap<String, Double>();
			List<String> labelsInClassifier = (List<String>) classifier.labels();

			System.out.println();
			
			//Ensuring that 'NONE' is always at index 0
			labelsInClassifier.remove("NONE");
			labelsInClassifier.add(0, "NONE");
			
			for(String l:labelsInClassifier)
				LogInfo.logs(l);
			
			for(BioDatum d:dataset) {
				Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
				d.setPredictedLabel(classifier.classOf(newDatum));
				
				for(String possibleLabel:labelsInClassifier)
					weights.put(String.format("%d,%d,%d", eventMentions.indexOf(d.event1), eventMentions.indexOf(d.event2),
											labelsInClassifier.indexOf(possibleLabel)), 
											classifier.logProbabilityOf(newDatum).getCount(possibleLabel));
				
				labelings.put(eventMentions.indexOf(d.event1) + "," + eventMentions.indexOf(d.event2), new Pair<String, String>(d.label, d.predictedLabel()));
				
				
				

				if(!d.label.equals("NONE")) {
					if(d.label == RelationType.PreviousEvent.toString()) 
						pG.incrementCount(d.event1);
					if(d.label == RelationType.NextEvent.toString())
						pG.incrementCount(d.event2);
					if(d.label == RelationType.SuperEvent.toString())
						sG.incrementCount(d.event2);
					if(d.label == RelationType.SubEvent.toString())
						sG.incrementCount(d.event1);
					if(d.label == RelationType.Causes.toString())
						cG.incrementCount(d.event2);
					if(d.label == RelationType.Caused.toString())
						cG.incrementCount(d.event1);
					if(ArgumentRelation.getEventRelations().contains(d.label)) {
						dG.incrementCount(d.event1);
						dG.incrementCount(d.event2);
					}
				}
				if(!d.predictedLabel().equals("NONE")) {
					if(d.predictedLabel() == RelationType.PreviousEvent.toString()) 
						pP.incrementCount(d.event1);
					if(d.predictedLabel() == RelationType.NextEvent.toString())
						pP.incrementCount(d.event2);
					if(d.predictedLabel() == RelationType.SuperEvent.toString())
						sP.incrementCount(d.event2);
					if(d.predictedLabel() == RelationType.SubEvent.toString())
						sP.incrementCount(d.event1);
					if(d.predictedLabel() == RelationType.Causes.toString())
						cP.incrementCount(d.event2);
					if(d.predictedLabel() == RelationType.Caused.toString())
						cP.incrementCount(d.event1);
					if(ArgumentRelation.getEventRelations().contains(d.predictedLabel())) {
						dP.incrementCount(d.event1);
						dP.incrementCount(d.event2);
					}
				}
			}
			
			for(BioDatum d:dataset) {		
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
			/*
			//System.out.println(weights);
			HashMap<Pair<Integer,Integer>, Integer> best = ILPOptimizer.OptimizeEventRelation(weights, eventMentions.size(), labelsInClassifier);
			
			
			for(Pair<Integer,Integer> p:best.keySet()) {
				for(BioDatum d:dataset) {
					if(eventMentions.indexOf(d.event1) == p.first() && 
							eventMentions.indexOf(d.event2) == p.second() && !d.predictedLabel().equals(labelsInClassifier.get(best.get(p)))) {
						d.setPredictedLabel(labelsInClassifier.get(best.get(p)));
						buffer.append(String.format("\n%s -> %s [ label = \"%s\" fontcolor=\"darkgreen\" %s color = \"%s\"];", "node"+p.first(), "node"+p.second(), labelsInClassifier.get(best.get(p)),
								//If Cotemporal or same event, put bi-directional edges
								(labelsInClassifier.get(best.get(p)).equals("CotemporalEvent") || labelsInClassifier.get(best.get(p)).equals("SameEvent")) ? "dir = \"both\"" : "", "darkgreen")) ;
					}
				}
			}
			*/
			for(BioDatum d:dataset) {
				if(d.predictedLabel().equals(d.label()) && d.predictedLabel().equals("NONE"))
					continue;
				
				if(d.predictedLabel().equals(d.label)) {
					LogInfo.logs(String.format("%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", ex.id, "Correct", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else if(d.label().equals("NONE") && !d.predictedLabel().equals("NONE")){
					LogInfo.logs(String.format("%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s",  ex.id, "Extra", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else if(!d.label().equals("NONE") && d.predictedLabel().equals("NONE")){
					LogInfo.logs(String.format("%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", ex.id,  "Missed", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
				else {
					LogInfo.logs(String.format("%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", ex.id,  "Incorrect", Utils.getText(d.event1.getTreeNode()), Utils.getText(d.event2.getTreeNode()), d.label(), d.predictedLabel()));
				}
			}
			
			buffer.append("\n}");
			predicted.addAll(dataset);
			Utils.writeStringToFile(buffer.toString(), "GraphViz/" + ex.id + ".gv");
			
			try {
	        	Runtime rt = Runtime.getRuntime();
	        	rt.exec("dot -o GraphViz/" + ex.id + ".png -Tpng GraphViz/" + ex.id + ".gv");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.out.println("ERORRRR" + e.getMessage());
				e.printStackTrace();
			}

			for(EventMention evt:eventMentions) {
				degreeDistribution.incrementCount((int)dG.getCount(evt));
				prevEvent.incrementCount((int)pG.getCount(evt));
				superEvent.incrementCount((int)sG.getCount(evt));
				causeEvent.incrementCount((int)cG.getCount(evt));
				
				degreeDistributionPred.incrementCount((int)dP.getCount(evt));
				prevEventPred.incrementCount((int)pP.getCount(evt));
				superEventPred.incrementCount((int)sP.getCount(evt));
				causeEventPred.incrementCount((int)cP.getCount(evt));
			}
			
			for(int i = 0; i < eventMentions.size(); i++){
				for(int j = i+1; j < eventMentions.size(); j++){
					for(int k = j+1; k < eventMentions.size(); k++){
						Pair<String, String> rel1 = labelings.get(i + "," + j), rel2 = labelings.get(j + "," + k), rel3 = labelings.get(i + "," + k);
						
						Triple<String, String, String> goldEquivalent = Utils.getEquivalentBaseTriple(new Triple<String, String, String>(rel1.first(), rel2.first(), rel3.first()));
						Triple<String, String, String> predEquivalent = Utils.getEquivalentBaseTriple(new Triple<String, String, String>(rel1.second(), rel2.second(), rel3.second()));
						LogInfo.logs(new Triple<String, String, String>(rel1.second(), rel2.second(), rel3.second()));
						LogInfo.logs(predEquivalent);
						countGoldTriples.incrementCount(goldEquivalent.first()+ "," + goldEquivalent.second() + "," + goldEquivalent.third());
						countPredictedTriples.incrementCount(predEquivalent.first()+ "," + predEquivalent.second() + "," + predEquivalent.third());
						String rel = String.format("%s->%s->%s", rel1.second(), rel2.second(), rel3.second());
						
						if((goldEquivalent.first().equals("NONE") && goldEquivalent.second().equals("SameEvent") && goldEquivalent.third().equals("SameEvent")))
							LogInfo.logs("SAMEEVENTNOClosure" + ex.id + " :" + i + ":" + j + ":" + k);
						
						/*if(rel.equals("Causes->Caused->NONE")) {
							LogInfo.logs("NOGOLD:Causes->Caused->NONE " + ex.id + " " + eventMentions.get(i).getTreeNode()
									+ ":" + eventMentions.get(j).getTreeNode() + ":" + eventMentions.get(k).getTreeNode());
						}
						if(rel.equals("PreviousEvent->Caused->NONE")) {
							LogInfo.logs("NOGOLD:PreviousEvent->Caused->NONE " + ex.id + " " + eventMentions.get(i).getTreeNode()
									+ ":" + eventMentions.get(j).getTreeNode() + ":" + eventMentions.get(k).getTreeNode());
						}
						if(rel.equals("Caused->PreviousEvent->NONE")) {
							LogInfo.logs("NOGOLD:Caused->PreviousEvent->NONE " + ex.id + " " + eventMentions.get(i).getTreeNode()
									+ ":" + eventMentions.get(j).getTreeNode() + ":" + eventMentions.get(k).getTreeNode());
						}
						if(rel.equals("CotemporalEvent->CotemporalEvent->NONE")) {
							LogInfo.logs("NOGOLD:CotemporalEvent->CotemporalEvent->NONE "+ ex.id + " " + eventMentions.get(i).getTreeNode()
									+ ":" + eventMentions.get(j).getTreeNode() + ":" + eventMentions.get(k).getTreeNode());
						}
						if(rel.equals("CotemporalEvent->PreviousEvent->SameEvent")) {
							LogInfo.logs("NOGOLD:CotemporalEvent->PreviousEvent->SameEvent " + ex.id + " " + eventMentions.get(i).getTreeNode() 
									+ ":" + eventMentions.get(j).getTreeNode() + ":" + eventMentions.get(k).getTreeNode());
						}*/
					}
				}
			}
			
			LogInfo.end_track();
		}
		return predicted;
	}

}
