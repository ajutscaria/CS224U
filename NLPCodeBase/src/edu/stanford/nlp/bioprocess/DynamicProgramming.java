package edu.stanford.nlp.bioprocess;

import java.util.IdentityHashMap;
import java.util.List;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class DynamicProgramming {
	Tree syntacticParse;
	IdentityHashMap<Tree, Pair<Double, String>> tokenMap;
	IdentityHashMap<Tree, BioDatum> nodeDatumMap;
	
	public DynamicProgramming(CoreMap sentence, IdentityHashMap<Tree, Pair<Double, String>> tokenMap, List<BioDatum> data) {
		this.syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		
		this.tokenMap = tokenMap;
		nodeDatumMap = new IdentityHashMap<Tree, BioDatum>();
		for (BioDatum d : data) {
			nodeDatumMap.put(d.entityNode, d);
			//LogInfo.logs(d.node+":"+d.guessLabel+":"+d.getProbability());
		}
//		for (Tree node : syntacticParse.preOrderNodeList()) {
//			LogInfo.logs(node.toString()+":"+this.tokenMap.get(node));
//		}
		//calculateLabels();
		//addTreeNodeAnnotations(this.syntacticParse, sentence.get(TokensAnnotation.class));
//		for (Tree node : syntacticParse.preOrderNodeList()) {
//			LogInfo.logs(node.toString()+":"+this.tokenMap.get(node));
//		}
	}
	
	 
	 
	 

	
	// Assumes the map has probs for E.
	public void calculateLabels() {
		for (Tree node : this.syntacticParse.postOrderNodeList()) {
			if (node.isLeaf() || node.value().equals("ROOT")) {
				continue;
			}
			Pair<Double, String> targetNodePair = this.tokenMap.get(node);
			Double nodeO = Math.log(1-targetNodePair.first);
			Double nodeE = Math.log(targetNodePair.first);
			for (Tree child : node.getChildrenAsList()) {
				if (child.isLeaf()) {
					continue;
				}
				Pair<Double, String> nodeVals = this.tokenMap.get(child);
				nodeE += (Math.log(1-nodeVals.first));
				if (nodeVals.second.equals("O")) {
					nodeO += Math.log((1-nodeVals.first));
				} else {
					nodeO += Math.log(nodeVals.first);
				}
			}
			nodeO = Math.exp(nodeO);
			nodeE = Math.exp(nodeE);
			double sum = nodeO + nodeE;
			nodeO = nodeO/sum;
			nodeE = nodeE/sum;
			
			if (nodeO > nodeE) {// && !allchildrenE(node)) {
				targetNodePair.setFirst(1-nodeO);
				nodeDatumMap.get(node).setProbability(1-nodeO);
				targetNodePair.setSecond("O");
				nodeDatumMap.get(node).guessLabel = "O";
			} else {
				//LogInfo.logs("\n\n-------------------------Predicted Entity: "+node+":" +node.getSpan());
				targetNodePair.setFirst(nodeE);
				nodeDatumMap.get(node).setProbability(nodeE);
				targetNodePair.setSecond("E");
				nodeDatumMap.get(node).guessLabel = "E";
				for (Tree child : node.preOrderNodeList()) {
					if (child.isLeaf() || child.equals(node)) {
						continue;
					}
					//LogInfo.logs("Resetting " + child + " to O");
					this.tokenMap.get(child).setSecond("O");
					nodeDatumMap.get(child).guessLabel = "O";
				}
				//for(String n:nodeDatumMap.keySet())
				//	LogInfo.logs(n + ":" + node.getSpan() +":"+ nodeDatumMap.get(n).guessLabel );
				//LogInfo.logs("============================================================\n\n");
			}
		}
		
		//HACK - remove all DT
		for (Tree node : this.syntacticParse.postOrderNodeList()) {
			if(node.value().equals("DT"))
				nodeDatumMap.get(node).guessLabel = "O";
		}
	}
}
