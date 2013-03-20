package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EntityPredictionInferer extends Inferer {
	private boolean printDebugInformation = true;
	List<Datum> prediction = null;
	
	public EntityPredictionInferer() {
		
	}
	
	public EntityPredictionInferer(List<Datum> predictions){
		this.prediction = predictions;
	}

	public List<Datum> BaselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff) {
		List<Datum> predicted = new ArrayList<Datum>();
		//EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example example:examples) {
			LogInfo.begin_track("Example %s",example.id);
			
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				Set<Tree> eventNodes = null;
				if(prediction == null)
					eventNodes = Utils.getEventNodesFromSentence(sentence).keySet();
				else
					eventNodes = Utils.getEventNodesForSentenceFromDatum(prediction, sentence);
				List<Datum> test = ff.setFeaturesTest(sentence, eventNodes);
				for(Tree event:eventNodes) {
					LogInfo.logs("******************Event " + Utils.getText(event)+ 
								"[" + (Utils.getEventNodesFromSentence(sentence).containsKey(event)?"Correct":"Wrong") +"]**********************");
					List<Datum> testDataEvent = new ArrayList<Datum>();
					for(Datum d:test)
						if(d.eventNode == event) {
							testDataEvent.add(d);
						}
					List<Datum> testDataWithLabel = new ArrayList<Datum>();
	
					for (int i = 0; i < testDataEvent.size(); i += parameters.getLabelIndex().size()) {
						testDataWithLabel.add(testDataEvent.get(i));
					}
					
					for(Datum d:testDataWithLabel) {
						if(d.entityNode.value().equals("NP") && Utils.isNodesRelated(sentence, d.entityNode, event))
							d.guessLabel = "E";
						else
							d.guessLabel = "O";
					}
					predicted.addAll(testDataWithLabel);
					
					LogInfo.logs(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
					LogInfo.logs("\n---------GOLD ENTITIES-------------------------");
					for(Datum d:testDataWithLabel) 
						if(d.label.equals("E"))
							LogInfo.logs(d.entityNode + ":" + d.label);
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(Datum d:testDataWithLabel)
						if(d.guessLabel.equals("E") || d.label.equals("E"))
							LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.label, d.guessLabel));
					LogInfo.logs("------------------------------------------\n");
				}
			}
			LogInfo.end_track();
		}
		return predicted;
	}
	
	public List<Datum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<Datum> predicted = new ArrayList<Datum>();
		//EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example ex:testData) {
			LogInfo.begin_track("Example %s",ex.id);
			//IdentityHashSet<Tree> entities = Utils.getEntityNodes(ex);
			
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				if(printDebugInformation ) {
					LogInfo.logs(sentence);
					LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
					LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));
				}
				Set<Tree> eventNodes = null;
				if(prediction == null)
					eventNodes = Utils.getEventNodesFromSentence(sentence).keySet();
				else
					eventNodes = Utils.getEventNodesForSentenceFromDatum(prediction, sentence);
				List<Datum> test = ff.setFeaturesTest(sentence, eventNodes);
				
				for(Tree event:eventNodes) {
					LogInfo.logs("******************Event " + Utils.getText(event)+ 
							"[" + (Utils.getEventNodesFromSentence(sentence).containsKey(event)?"Correct":"Wrong") +"]**********************");
					List<Datum> testDataEvent = new ArrayList<Datum>();
					for(Datum d:test)
						if(d.eventNode == event) {
							//LogInfo.logs(d.entityNode);
							testDataEvent.add(d);
						}
					List<Datum> testDataWithLabel = new ArrayList<Datum>();
	
					for (int i = 0; i < testDataEvent.size(); i += parameters.getLabelIndex().size()) {
						testDataWithLabel.add(testDataEvent.get(i));
					}
					MaxEntModel viterbi = new MaxEntModel(parameters.getLabelIndex(), parameters.getFeatureIndex(), parameters.getWeights());
					viterbi.decodeForEntity(testDataWithLabel, testDataEvent);
					
					IdentityHashMap<Tree, Pair<Double, String>> map = new IdentityHashMap<Tree, Pair<Double, String>>();
	
					for(Datum d:testDataWithLabel) {
						if (Utils.subsumesEvent(d.entityNode, sentence)) {
							map.put(d.entityNode, new Pair<Double, String>(0.0, "O"));
						} else {
							map.put(d.entityNode, new Pair<Double, String>(d.getProbability(), d.guessLabel));
						}
					}
					
					DynamicProgramming dynamicProgrammer = new DynamicProgramming(sentence, map, testDataWithLabel);
					dynamicProgrammer.calculateLabels();
					
					predicted.addAll(testDataWithLabel);
					
					LogInfo.logs(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
					LogInfo.logs("\n---------GOLD ENTITIES-------------------------");
					for(Datum d:testDataWithLabel) 
						if(d.label.equals("E"))
							LogInfo.logs(d.entityNode + ":" + d.label);
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(Datum d:testDataWithLabel)
						if(d.guessLabel.equals("E") || d.label.equals("E"))
							LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.label, d.guessLabel));
					LogInfo.logs("------------------------------------------\n");
				}
				for(EventMention ev:sentence.get(EventMentionsAnnotation.class))
					if(!eventNodes.contains(ev.getTreeNode())){
						LogInfo.logs("||||||||||||||||||||||Event " +ev.getTreeNode()+ "[Missed]||||||||||||||");
						LogInfo.logs("\n---------Missed entities-------------------------");
						for(ArgumentRelation m:ev.getArguments())
							if(ArgumentRelation.getSemanticRoles().contains(m.type))
								LogInfo.logs(m.mention.getTreeNode());
						LogInfo.logs("------------------------------------------\n");
					}
			}
			LogInfo.end_track();

		}
		return predicted;
	}
	
}