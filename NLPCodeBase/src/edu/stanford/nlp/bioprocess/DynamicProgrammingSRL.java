package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

public class DynamicProgrammingSRL {
	Tree syntacticParse;
	IdentityHashMap<Tree, List<Pair<String, Double>>> tokenMap;
	IdentityHashMap<Tree, BioDatum> nodeDatumMap;
	IdentityHashMap<Tree, List<Pair<IdentityHashMap<Tree, String>, Double>>> nodeRanks;
	Index labelIndex;
	
	public DynamicProgrammingSRL(CoreMap sentence, IdentityHashMap<Tree, List<Pair<String, Double>>> map, List<BioDatum> data, Index labelIndex) {
		this.syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		
		this.tokenMap = map;
		nodeDatumMap = new IdentityHashMap<Tree, BioDatum>();
		for (BioDatum d : data) {
			nodeDatumMap.put(d.entityNode, d);
		}
//		for (Tree t : this.syntacticParse.postOrderNodeList()) {
//			if (t.isLeaf()) continue;
//			System.out.println("Looking for: "+t.pennString());
//			System.out.println(t.getLeaves()+":"+nodeDatumMap.get(t).getBestRole());
//		}
		System.out.println("Syntactic parse tree: ");
		this.syntacticParse.pennPrint();
		this.nodeRanks = new IdentityHashMap<Tree, List<Pair<IdentityHashMap<Tree, String>, Double>>> ();
		this.labelIndex = labelIndex;
		for (Tree t : this.syntacticParse.postOrderNodeList()) {
//			if (!t.isLeaf() && !t.isPreTerminal() && !t.value().equals("ROOT")) {
			if (t.isPrePreTerminal()) {
				System.out.println("--------------------"+t.pennString()+"----------------------------");
				IdentityHashMap<Tree, String> allottedRoles = new IdentityHashMap<Tree, String>();
				List<Pair<IdentityHashMap<Tree, String>, Double>> withParentallPerms = new ArrayList<Pair<IdentityHashMap<Tree, String>, Double>>();
				List<Pair<IdentityHashMap<Tree, String>, Double>> allPerms = new ArrayList<Pair<IdentityHashMap<Tree, String>, Double>>();
				genPermutations(t.getChildrenAsList(), 0, allottedRoles, allPerms);
				for (int i=0; i<allPerms.size(); i++) {
					Pair<IdentityHashMap<Tree, String>, Double> permPair = allPerms.get(i);
					boolean allNone = true;
					for (Tree child : permPair.first.keySet()) {
						if (!permPair.first.get(child).equals("NONE")) {
							allNone = false;
						}
						System.out.print(child.getLeaves()+":"+permPair.first.get(child)+"---->");
					}
					System.out.println("Prob: "+permPair.second);
					if (allNone) {
						System.out.println("THIS IS ALL NONE");
						double withParentProb = permPair.second;
						String parentRole = null;
						if (nodeDatumMap.get(t) == null) {
							System.out.println("Empty nodeDatumMap");
						}
						for (int cntr=0; cntr < nodeDatumMap.get(t).rankedRoleProbs.size(); cntr++) {
							withParentProb = permPair.second * nodeDatumMap.get(t).rankedRoleProbs.get(cntr).second;
							parentRole = nodeDatumMap.get(t).rankedRoleProbs.get(cntr).first;
							permPair.first.put(t, parentRole);
							permPair.second = withParentProb;
							Pair<IdentityHashMap<Tree, String>, Double> permElem = new Pair<IdentityHashMap<Tree, String>, Double>();
							permElem.first = (IdentityHashMap<Tree, String>) permPair.first.clone();
							permElem.second = permPair.second;
							withParentallPerms.add(permElem);
						}
//						for (int printer = 0; printer < withParentallPerms.size(); printer++) {
//							for (Tree iter : withParentallPerms.get(printer).first.keySet()) {
//								System.out.print(iter.getLeaves()+":"+withParentallPerms.get(printer).first.get(iter)+"---->");
//							}
//							System.out.println("Prob-"+parentRole+": "+withParentallPerms.get(printer).second);
//						}
					} else {
						double withParentProb = permPair.second;
						String parentRole = null;
						withParentProb = permPair.second * nodeDatumMap.get(t).rankedRoleProbs.get(0).second;
						parentRole = "NONE";
						permPair.first.put(t, parentRole);
						permPair.second = withParentProb;
						Pair<IdentityHashMap<Tree, String>, Double> permElem = new Pair<IdentityHashMap<Tree, String>, Double>();
						permElem.first = (IdentityHashMap<Tree, String>) permPair.first.clone();
						permElem.second = permPair.second;
						withParentallPerms.add(permElem);
//						for (int printer = 0; printer < withParentallPerms.size(); printer++) {
//							for (Tree iter : withParentallPerms.get(printer).first.keySet()) {
//								System.out.print(iter.getLeaves()+":"+withParentallPerms.get(printer).first.get(iter)+"---->");
//							}
//							System.out.println("Prob-"+parentRole+": "+withParentallPerms.get(printer).second);
//						}
					}
				}
				//Collections.sort(withParentallPerms, new PairComparatorByDoubleHashMap());

				for (int printer = 0; printer < withParentallPerms.size(); printer++) {
					for (Tree iter : withParentallPerms.get(printer).first.keySet()) {
						System.out.print(iter.getLeaves()+":"+withParentallPerms.get(printer).first.get(iter)+"---->");
					}
					System.out.println("Prob-parent: "+withParentallPerms.get(printer).second);
				}
				System.out.println("--------------------"+t.pennString()+"---------------------------");
//				break;
			}
		}
	}
	
	// Pair<IdentityHashMap<Tree, String>, Double>
	public void genPermutations(List<Tree> children, int k, IdentityHashMap<Tree, String> allottedRoles, List<Pair<IdentityHashMap<Tree, String>, Double>> allPerms) {
		if ( k != children.size() ) {
			Tree child = children.get(k);
//			System.out.println("Looking for: "+child.pennString());
			BioDatum childDatum = nodeDatumMap.get(child);
			for (int i=0; i<childDatum.rankedRoleProbs.size(); i++) {
				String role = childDatum.rankedRoleProbs.get(i).first;
				allottedRoles.put(child, role);
//				System.out.println("Recursive k: "+k);
				genPermutations(children, ++k, allottedRoles, allPerms);
				k--;
			}
		} else {
			double prob = 1.0;
			for (Tree t : allottedRoles.keySet()) {
//				System.out.println(t.getLeaves().toString()+":"+allottedRoles.get(t));
//				for (int i=0; i<nodeDatumMap.get(t).rankedRoleProbs.size(); i++) {
//					System.out.println(nodeDatumMap.get(t).rankedRoleProbs.get(i).first+":"+nodeDatumMap.get(t).rankedRoleProbs.get(i).second);
//				}
				prob *= nodeDatumMap.get(t).getRoleProb(allottedRoles.get(t));
			}
//			System.out.println("Prob: "+prob);
			allPerms.add(new Pair<IdentityHashMap<Tree, String>, Double>((IdentityHashMap<Tree, String>) allottedRoles.clone(), prob));
		}
	}
	
	public void genPermutationsInternal(List<Tree> children, int k, IdentityHashMap<Tree, String> allottedRoles, List<Pair<IdentityHashMap<Tree, String>, Double>> allPerms) {
		if ( k != children.size() ) {
			Tree child = children.get(k);
//			System.out.println("Looking for: "+child.pennString());
			BioDatum childDatum = nodeDatumMap.get(child);
			for (int i=0; i<childDatum.rankedRoleProbs.size(); i++) {
				String role = childDatum.rankedRoleProbs.get(i).first;
				allottedRoles.put(child, role);
//				System.out.println("Recursive k: "+k);
				genPermutationsInternal(children, ++k, allottedRoles, allPerms);
				k--;
			}
		} else {
			double prob = 1.0;
			for (Tree t : allottedRoles.keySet()) {
//				System.out.println(t.getLeaves().toString()+":"+allottedRoles.get(t));
//				for (int i=0; i<nodeDatumMap.get(t).rankedRoleProbs.size(); i++) {
//					System.out.println(nodeDatumMap.get(t).rankedRoleProbs.get(i).first+":"+nodeDatumMap.get(t).rankedRoleProbs.get(i).second);
//				}
				prob *= nodeDatumMap.get(t).getRoleProb(allottedRoles.get(t));
			}
//			System.out.println("Prob: "+prob);
			allPerms.add(new Pair<IdentityHashMap<Tree, String>, Double>((IdentityHashMap<Tree, String>) allottedRoles.clone(), prob));
		}
	}
	
	// Assumes the map has probs for E.
	// Probs of all. ?
	// Iterate over all and make all labels.
	// Take maximum and assign labels
	// Prepare for layers above.
	public void calculateLabels() {
		for (Tree node : this.syntacticParse.postOrderNodeList()) {
			if (node.isLeaf() || node.value().equals("ROOT")) {
				continue;
			}
			
			if (node.isPreTerminal()) {
				List<Pair<String, Double>> targetNodePair = this.tokenMap.get(node);
			}
			
			
			/*
			 * Pair<double[], String> targetNodePair = this.tokenMap.get(node);
			
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
			*/
		}
		
		//HACK - remove all DT
		for (Tree node : this.syntacticParse.postOrderNodeList()) {
			if(node.value().equals("DT"))
				nodeDatumMap.get(node).guessLabel = "O";
		}
	}
}
