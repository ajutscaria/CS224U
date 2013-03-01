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
	HashMap<Datum, Pair<Double, String>> tokenMap;
	HashMap<String, Datum> nodeDatumMap;
	
	public DynamicProgramming(CoreMap sentence, HashMap<Datum, Pair<Double, String>> tokenMap, List<Datum> data) {
		this.syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		this.syntacticParse.pennPrint();
		this.tokenMap = tokenMap;
		nodeDatumMap = new HashMap<String, Datum>();
		for (Datum d : data) {
			nodeDatumMap.put(Utils.getKeyFromTree(d.node), d);
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
			Pair<Double, String> targetNodePair = this.tokenMap.get(nodeDatumMap.get(Utils.getKeyFromTree(node)));
			Double nodeO = Math.log(1-targetNodePair.first);
			Double nodeE = Math.log(targetNodePair.first);
			for (Tree child : node.getChildrenAsList()) {
				if (child.isLeaf()) {
					continue;
				}
				Pair<Double, String> nodeVals = this.tokenMap.get(nodeDatumMap.get(Utils.getKeyFromTree(child)));
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
			
			if (nodeO > nodeE && !allchildrenE(node)) {
				targetNodePair.setFirst(1-nodeO);
				nodeDatumMap.get(Utils.getKeyFromTree(node)).setProbability(1-nodeO);
				targetNodePair.setSecond("O");
				nodeDatumMap.get(Utils.getKeyFromTree(node)).guessLabel = "O";
			} else {
				System.out.println("\n\n-------------------------Predicted Entity: "+node+":" +node.getSpan());
				targetNodePair.setFirst(nodeE);
				nodeDatumMap.get(Utils.getKeyFromTree(node)).setProbability(nodeE);
				targetNodePair.setSecond("E");
				nodeDatumMap.get(Utils.getKeyFromTree(node)).guessLabel = "E";
				for (Tree child : node.preOrderNodeList()) {
					if (child.isLeaf() || child.equals(node)) {
						continue;
					}
					//System.out.println("Resetting " + child + " to O");
					this.tokenMap.get(nodeDatumMap.get(Utils.getKeyFromTree(child))).setSecond("O");
					nodeDatumMap.get(Utils.getKeyFromTree(child)).guessLabel = "O";
				}
				//for(String n:nodeDatumMap.keySet())
				//	System.out.println(n + ":" + node.getSpan() +":"+ nodeDatumMap.get(n).guessLabel );
				//System.out.println("============================================================\n\n");
			}
		}
		
		//HACK - remove all DT
		for (Tree node : this.syntacticParse.postOrderNodeList()) {
			if(node.value().equals("DT"))
				nodeDatumMap.get(Utils.getKeyFromTree(node)).guessLabel = "O";
		}
	}

	private boolean allchildrenE(Tree node) {
		for (Tree child : node.getChildrenAsList()) {
			if (child.isLeaf() || this.nodeDatumMap.get(Utils.getKeyFromTree(child)).guessLabel.equals("O"))
				return false;
		}
		return true;
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
