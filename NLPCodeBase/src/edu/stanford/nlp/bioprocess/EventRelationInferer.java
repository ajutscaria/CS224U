package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class EventRelationInferer extends Inferer {

	@Override
	public List<BioDatum> BaselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		for(Example example:examples) {
			for(EventMention evt:example.gold.get(EventMentionsAnnotation.class)) {
				for(ArgumentRelation rel:evt.getArguments()) {
					  if(rel.mention instanceof EventMention) { 
						  System.out.println(evt.getTreeNode() + "-" + rel.mention.getTreeNode() + "-->" + rel.type);
					  }
				}
			}
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				List<BioDatum> test = ff.setFeaturesTest(sentence, Utils.getEventNodesFromSentence(sentence).keySet());
				
				for(BioDatum d:test)
					d.guessLabel = "NextEvent";
				predicted.addAll(test);
			}
		}
		return predicted;
	}

	@Override
	public List<BioDatum> Infer(List<Example> testData, Params parameters,
			FeatureExtractor ff) {
		// TODO Auto-generated method stub
		return null;
	}

}
