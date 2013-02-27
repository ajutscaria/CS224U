package edu.stanford.nlp.bioprocess;

import java.util.HashMap;

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
	
	public DynamicProgramming(CoreMap sentence, HashMap<Tree, Pair<Double, String>> tokenMap) {
		this.syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		this.syntacticParse.pennPrint();
		this.tokenMap = tokenMap;
		for (Tree node : syntacticParse.preOrderNodeList()) {
			System.out.println(node.toString()+":"+this.tokenMap.get(node));
		}
		calculateLabels();
		for (Tree node : syntacticParse.preOrderNodeList()) {
			System.out.println(node.toString()+":"+this.tokenMap.get(node));
		}
	}
	
	// Assumes the map has probs for E.
	private void calculateLabels() {
		for (Tree node : this.syntacticParse.preOrderNodeList()) {
			if (this.tokenMap.get(node).second == "O" || this.tokenMap.get(node).second == "E") {
				continue;
			}
			Pair<Double, String> targetNodePair = this.tokenMap.get(node);
			Double nodeO = Math.log(1-targetNodePair.first);
			Double nodeE = Math.log(targetNodePair.first);
			for (Tree child : node.getChildrenAsList()) {
				Pair<Double, String> nodeVals = this.tokenMap.get(child);
				nodeO += Math.log(nodeVals.first);
				if (nodeVals.second == "E") {
					nodeE += Math.log((1-nodeVals.first));
				} else {
					nodeE += Math.log(nodeVals.first);
				}
			}
			if (nodeO > nodeE) {
				targetNodePair.setSecond("O");
			} else {
				targetNodePair.setSecond("E");
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
