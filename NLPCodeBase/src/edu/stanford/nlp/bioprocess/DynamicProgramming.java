package edu.stanford.nlp.bioprocess;

import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class DynamicProgramming {
	Tree syntacticParse;
	HashMap<Tree, Pair<Double, String>> tokenMap;
	HashMap<Tree, CoreLabel> treeLabelMap;
	HashMap<Tree, Datum> nodeDatumMap;
	
	public DynamicProgramming(CoreMap sentence, HashMap<Tree, Pair<Double, String>> tokenMap, List<Datum> data) {
		this.syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		this.syntacticParse.pennPrint();
		this.tokenMap = tokenMap;
		nodeDatumMap = new HashMap<Tree, Datum>();
		for (Datum d : data) {
			nodeDatumMap.put(d.node, d);
			System.out.println(d.node+":"+d.guessLabel+":"+d.getProbability());
		}
//		for (Tree node : syntacticParse.preOrderNodeList()) {
//			System.out.println(node.toString()+":"+this.tokenMap.get(node));
//		}
		//calculateLabels();
		//addTreeNodeAnnotations(this.syntacticParse, sentence.get(TokensAnnotation.class));
//		for (Tree node : syntacticParse.preOrderNodeList()) {
//			System.out.println(node.toString()+":"+this.tokenMap.get(node));
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
			if (nodeO > nodeE) {
				targetNodePair.setFirst(1-nodeO);
				nodeDatumMap.get(node).setProbability(1-nodeO);
				targetNodePair.setSecond("O");
				nodeDatumMap.get(node).guessLabel = "O";
			} else {
				System.out.println("Predicted Entity: "+node);
				targetNodePair.setFirst(nodeE);
				nodeDatumMap.get(node).setProbability(nodeE);
				targetNodePair.setSecond("E");
				nodeDatumMap.get(node).guessLabel = "E";
				for (Tree child : node.getChildrenAsList()) {
					if (child.isLeaf()) {
						continue;
					}
					this.tokenMap.get(child).setSecond("O");
					nodeDatumMap.get(child).guessLabel = "O";
				}
			}
		}
	}

	public static void checkTree()
	  {
		  System.out.println("In check tree");
		  String text = "A particular region of each X chromosome contains several genes involved in the inactivation process.";
		  TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		  GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		  LexicalizedParser lp = LexicalizedParser.loadModel();
		  Tree tree = lp.apply(text);
		  tree.pennPrint();
	  }
}
