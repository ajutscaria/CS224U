package edu.stanford.nlp.bioprocess;

import java.util.*;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import fig.basic.LogInfo;

public class EventFeatureFactory extends FeatureExtractor {
  public EventFeatureFactory(boolean useLexicalFeatures) {
    super(useLexicalFeatures);
  }

  boolean printDebug = false, printAnnotations = false, printFeatures = false;
  public static boolean test = false;
  Set<String> nominalizations = Utils.getNominalizedVerbs();
  HashMap<String, String> verbForms = Utils.getVerbForms();
  List<String> SelectedAdvMods = Arrays.asList(new String[]{"first, then, next, after, later, subsequent, before, previously"});
  HashMap<String, Integer> clusters = Utils.loadClustering();

  public FeatureVector computeFeatures(CoreMap sentence, Tree event) {
    //LogInfo.logs("Current node's text - " + getText(event));

    List<String> features = new ArrayList<String>();
    String currentWord = event.value();
    //List<Tree> leaves = event.getLeaves();
    Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
    CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    int currentTokenIndex = event.getSpan().getSource();

    //if (currentTokenIndex < tokens.size()-1){
    //System.out.println(String.format("current word is %s, next word is %s",token.originalText(), tokens.get(event.getSpan().getSource()+1).originalText()));
    //}
    IndexedWord word = Utils.findDependencyNode(sentence, event);
    Tree parent = event.parent(root);
    //IntPair eventSpan = event.getSpan();

    String parentCFGRule = parent.value() + "->";
    for(Tree n:parent.getChildrenAsList()) {
      parentCFGRule += n.value() + "|";
    }
    parentCFGRule = parentCFGRule.trim();

    //features.add("POS="+currentWord);
    /*if (Utils.findDepthInDependencyTree(sentence, event)==0)
			features.add("root=true,POS="+currentWord);
     */
    if(useLexicalFeatures){
      String text = token.lemma().toLowerCase();
      if(verbForms.containsKey(text)) {
        features.add("lemma="+verbForms.get(text));
      }
      else {
        features.add("lemma="+token.lemma().toLowerCase());
      }
      features.add("word="+token.originalText());
      features.add("POSlemma=" + currentWord+","+token.lemma());

      if(clusters.containsKey(text)) {
        features.add("clusterID=" + clusters.get(text));
        //LogInfo.logs(text + ", clusterID=" + clusters.get(text));
      }
      for(SemanticGraphEdge e: graph.getOutEdgesSorted(word)) {
        if(e.getRelation().toString().equals("advmod") && (currentWord.startsWith("VB") || nominalizations.contains(text)))
          features.add("advmod:" + e.getTarget());
        //LogInfo.logs("TIMEE : " + e.getRelation() + ":" + e.getTarget());
        //features.add("depedgein="+ e.getRelation() + "," + e.getTarget().toString().split("-")[1]);//need to deal with mult children same tag?
        //features.add("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
        //LogInfo.logs("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
      }

      if(nominalizations.contains(token.value())) {
        //LogInfo.logs("Adding nominalization - " + leaves.get(0));
        features.add("nominalization");
      }
    }
    //features.add("POSword=" + currentWord+","+leaves.get(0));
    //features.add("POSparentPOS="+ currentWord + "," + event.parent(root).value());

    //if(currentWord.startsWith("VB"))
    //	features.add("verb");
    features.add("ParentPOS=" + parent.value());
    features.add("path=" + StringUtils.join(Trees.pathNodeToNode(root, event, root), ",").replace("up-ROOT,down-ROOT,", ""));
    features.add("POSparentrule=" + currentWord+","+parentCFGRule);

    for(SemanticGraphEdge e: graph.getIncomingEdgesSorted(word)) {
      //features.add("depedgein="+ e.getRelation());// + "," + e.getSource().toString().split("-")[1]);
      //features.add("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
      //LogInfo.logs("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
    }



    String consecutiveTypes = "";
    if(currentTokenIndex > 0)
      consecutiveTypes += tokens.get(currentTokenIndex-1).get(PartOfSpeechAnnotation.class);
    consecutiveTypes += currentWord;
    if(currentTokenIndex < tokens.size() - 1)
      consecutiveTypes += tokens.get(currentTokenIndex+1).get(PartOfSpeechAnnotation.class);
    features.add("consecutivetypes="+consecutiveTypes);

    /*
		if(nominalizations.contains(leaves.get(0).value())) {
			LogInfo.logs("Adding nominalization - " + leaves.get(0));
			if (currentTokenIndex < tokens.size()-1 && tokens.get(currentTokenIndex+1).originalText().trim().equals("of")){
				features.add("nominalizationWithNextWordOf");
				LogInfo.logs("next word is of - " + leaves.get(0));
			}
			else
				features.add("nominalizationWithoutNextWordOf");
		}
     */
    //features.add("endsining=" + token.lemma() + "," + leaves.get(0).value().endsWith("ing"));
    //Cannot use this feature when looking at all tree nodes as candidates
    //features.add("POSparentPOSgrandparent="+currentWord + "," + event.parent(root).value() + "," + event.parent(root).parent(root).value());
    //Doesn't seem to work as expected even though the event triggers are mostly close to root in dependency tree.
    //features.add("POSdepdepth=" + currentWord + "," + Utils.findDepthInDependencyTree(sentence, event));

    //String classString = "class=" + tokenClass + ",";
    //List<String> updatedFeatures = new ArrayList<String>();
    //for(String feature:features)
    //	updatedFeatures.add(classString + feature);
    features.add("bias");

    if(Main.printFeature && Example.examplePrint){
      LogInfo.begin_track("Features of event %s", event.toString());
      for(String f:features){
        LogInfo.logs(f);
      }
      LogInfo.end_track();
    }

    FeatureVector fv = new FeatureVector(features);
    return fv;
  }

  public List<BioDatum> setFeaturesTrain(List<Example> data, Params parameterss) {
    return null;
  }
  public List<BioDatum> setFeaturesTrain(List<Example> data) {
    List<BioDatum> dataset = new ArrayList<BioDatum>();
    Counter<String> labelCounts = new ClassicCounter<String>();
    for (Example ex : data) {
      if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
      for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
        IdentityHashMap<Tree, EventType> eventNodes = Utils.getEventNodesFromSentence(sentence);
        if(printDebug) LogInfo.logs(sentence);
        if(printAnnotations) {
          LogInfo.logs("---Events--");
          for(EventMention event: sentence.get(EventMentionsAnnotation.class))
            LogInfo.logs(event.getValue());
        }
        for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
          if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
              !(node.value().startsWith("JJR") || node.value().startsWith("JJS") ||node.value().startsWith("NN") || node.value().equals("JJ") || node.value().startsWith("VB")))
            continue; //exclude illegal nodes

          String label = eventNodes.keySet().contains(node) ? "E" : "O";
          BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), label, node, node, ex.id);
          newDatum.features = computeFeatures(sentence, node);
          dataset.add(newDatum);
          labelCounts.incrementCount(label);
          //if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
        }
      }
      if(printDebug) LogInfo.logs("\n------------------------------------------------");
    }
    LogInfo.logs("Event training set label distribution=%s",labelCounts);
    return dataset;
  }

  public List<BioDatum> setFeaturesTest(CoreMap sentence, Set<Tree> selectedEntities, String exampleID) {
    List<BioDatum> dataset = new ArrayList<BioDatum>();
    IdentityHashMap<Tree, EventType> eventNodes = Utils.getEventNodesFromSentence(sentence);
    //int count = 0;
    test = true;
    for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
      String value = node.value();
      if(node.isLeaf() || value.equals("ROOT") || !node.isPreTerminal() || 
          !(value.startsWith("JJR") || value.startsWith("JJS") ||
              value.startsWith("NN") || value.equals("JJ") ||
              value.startsWith("VB")))
        continue;
      //count++;
      String type = eventNodes.keySet().contains(node) ? "E" : "O";
      //System.out.println(node.toString());
      BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, node, exampleID);
      newDatum.features = computeFeatures(sentence, node);
      dataset.add(newDatum);
    }
    //System.out.println(exampleID+":"+sentence);
    //System.out.println("Number of events: "+count + "\n");
    return dataset;
  }
}
