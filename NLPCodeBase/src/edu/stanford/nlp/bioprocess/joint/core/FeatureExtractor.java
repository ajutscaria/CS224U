package edu.stanford.nlp.bioprocess.joint.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.Utils;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import fig.basic.LogInfo;
import fig.basic.Option;

/**
 * Extracts features from input
 * @author heatherchen1003 TODO: Make sure what is trigger token. -> span(token,
 *         token+1)? 
 *         TODO: get features from coref resolution
 * 
 */
public class FeatureExtractor {
  

  public static class Options {
    @Option(gloss = "use lexical feature or not")
    public boolean useLexicalFeatures = true;
    @Option(gloss = "use baseline feature only or not")
    public boolean useBaselineFeaturesOnly = true;
    @Option(gloss = "run global model or not")
    public boolean runGlobalModel = false;
    @Option(gloss = "window for event-event relation")
    public int window = 10;
  }
  
  public static Options opts = new Options();
  private static final  HashMap<String, String> MarkAndPPClusters = new HashMap<String, String>();
  private static final  HashMap<String, String> AdvModClusters = new HashMap<String, String>();

  public FeatureExtractor(boolean useLexicalFeatures, boolean useBaselineFeaturesOnly, boolean runGlobalModel,
      int window){
    opts.useLexicalFeatures = useLexicalFeatures;
    opts.useBaselineFeaturesOnly = useBaselineFeaturesOnly;
    opts.runGlobalModel = runGlobalModel;
    opts.window = window;
  }
  
  
  public static FeatureVector getTriggerFV(Input input, int trigger) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    int tokenId = input.getTriggerTokenId(trigger);
    Pair<CoreMap, Integer> result = AnnotationUtils.getContainingSentenceAndOffset(sentences, tokenId, tokenId);
    CoreMap sentence = result.first;
    int offset = result.second;
    Tree event = AnnotationUtils.getEventNode(sentence, tokenId-offset);

    LogInfo.begin_track("Features for event "+trigger);
    LogInfo.logs("eventNode "+event.toString()+", in sentence "+sentence);
    FeatureVector fv = new FeatureVector();
    String currentPOS = event.value();

    if (opts.useLexicalFeatures) {
      CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
      String text = token.lemma().toLowerCase();
      if (Dictionary.verbForms.containsKey(text)) {
        fv.add("lexical", "lemma="+ Dictionary.verbForms.get(text));
      } else {
        fv.add("lexical", "lemma="+ text);
      }
      fv.add("lexical", "word="+ token.originalText());
      fv.add("lexical", "POSlemma="+ currentPOS + "," + token.lemma());

      if (Dictionary.clusters.containsKey(text)) {
        fv.add("cluster", "clusterID=" + Dictionary.clusters.get(text));
      }
      
      addAdvModFeature(sentence, event, fv, currentPOS, text);

      if (Dictionary.nominalizations.contains(token.value())) {
        fv.add("lexical", "nominalization");
      }
    }

    Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    Tree parent = event.parent(root);
    String parentCFGRule = buildParentCFGRule(parent);
    fv.add("syntactic", "ParentPOS=" + parent.value());
    fv.add("syntactic", "path="
        + StringUtils.join(Trees.pathNodeToNode(root, event, root), ",")
        .replace("up-ROOT,down-ROOT,", ""));
    fv.add("syntactic", "POSparentrule=" + currentPOS + "," + parentCFGRule);
    
    String postagTrigram = getPosTagTrigram(sentence, event, currentPOS);
    fv.add("syntactic", "postag_trigram=" + postagTrigram);
    fv.printFeatures();
    LogInfo.end_track();
    return fv;
  }

  public static FeatureVector getArgumentFV(Input input, int trigger,
      int argument) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    IntPair entityspan = input.getArgumentCandidateSpan(trigger, argument);
    Pair<CoreMap, Integer> result = AnnotationUtils.getContainingSentenceAndOffset(sentences, entityspan.getSource(),
        entityspan.getTarget()-1);
    CoreMap sentence = result.first;
    int offset = result.second;
    
    int eventtokenId = input.getTriggerTokenId(trigger);
    Tree event = AnnotationUtils.getEventNode(sentence, eventtokenId-offset);
    Tree entity = AnnotationUtils.getEntityNode(sentence, 
        new IntPair(entityspan.getSource()-offset, entityspan.getTarget()-offset));
    System.out.println(sentence.toString()+": "+event.toString()+", "+entity.toString()+"\n");
    
    LogInfo.begin_track("Features for entity "+argument+" of event "+trigger);
    LogInfo.logs("entityNode "+entity.toString()+" for event "+event.toString()+" in sentence: "+sentence);
    
    FeatureVector fv = new FeatureVector();
    
    boolean containS = false;
    for (Tree node : entity.getChildrenAsList()) {
      if (node.value().equals("S") || node.value().equals("SBAR")) {
        containS = true;
        break;
      }
    }
    
    fv.add("syntactic", "EntContainsS="+containS);
    fv.add("lexical", "EvtLemma=" + event.getLeaves().get(0).value());
    boolean dependencyExists = Utils.isNodesRelated(sentence, entity, event);
    fv.add("syntactic", "EntCatDepRel=" + entity.value() + "," + dependencyExists);
    fv.add("lexical", "EntHeadEvtPOS="
        + Utils.findCoreLabelFromTree(sentence, entity).lemma() + ","
        + event.preTerminalYield().get(0).value());
    String depPath = Utils.getDependencyPath(sentence, entity, event);
    fv.add("syntactic", "EvtToEntDepPath="
        + ((depPath.equals("") || depPath.equals("[]")) ? 0 : depPath
            .split(",").length));
    fv.add("lexical", "EntHeadEvtHead="
        + entity.headTerminal(new CollinsHeadFinder()) + ","
        + event.getLeaves().get(0));
    fv.add("lexical", "EntNPAndRelatedToEvt="
        + (entity.value().equals("NP") && Utils.isNodesRelated(sentence,
            entity, event)));
    fv.printFeatures();
    LogInfo.end_track();
    return fv;
  }

  public static FeatureVector getRelationFV(Input input, int trig1, int trig2) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    int tokenId1 = input.getTriggerTokenId(trig1);
    Pair<CoreMap,Integer> result1 = AnnotationUtils.getContainingSentenceAndOffset(sentences, tokenId1, tokenId1);
    CoreMap sentence1 = result1.first;
    int offset1 = result1.second;
    Tree event1 = AnnotationUtils.getEventNode(sentence1, tokenId1-offset1);
    int tokenId2 = input.getTriggerTokenId(trig2);
    Pair<CoreMap, Integer> result2 = AnnotationUtils.getContainingSentenceAndOffset(sentences, tokenId2, tokenId2);
    CoreMap sentence2 = result2.first;
    int offset2 = result2.second;
    Tree event2 = AnnotationUtils.getEventNode(sentence2, tokenId2-offset2);
    ClusterSetup();

    LogInfo.begin_track("Features for event-event "+trig1+","+trig2);
    LogInfo.logs("event1 "+event1.toString()+" , event2 "+event2.toString());
    
    FeatureVector fv = new FeatureVector();
    
    // Number of sentences and words between two event mentions. Quantized to
    // 'Low', 'Medium', 'High' etc.
    Pair<Integer, Integer> countsSentenceAndWord = AnnotationUtils.findNumberOfSentencesAndWordsBetween(
        input, event1, event2);
    int sentenceBetweenEvents = countsSentenceAndWord.first();
    int wordsBetweenEvents = countsSentenceAndWord.second(); 
    
    // XXX: This may not work anymore
    //boolean isImmediatelyAfter = (trig2 == trig1 + 1) ? true : false;
    boolean isWithinWindow = (wordsBetweenEvents < opts.window) ? true : false;

    // Lemmas of both events
    lemmaEventPairFeature(sentence1, event1, sentence2, event2, fv);

    if (isWithinWindow) {
      // Add words in between two event mentions if they are adjacent in text.
      addWordsBetweenEventsFeature(input, event1, event2, fv,
          sentenceBetweenEvents);
      
      // If event1 is the first event in the paragraph and is a
      // nominalization, it is likely that others are sub-events
      if (trig1 == 0 && event1.value().startsWith("NN")) {
        fv.add("firstAndNominalization", "firstAndNominalization");
      }
      
      // Extract advmod relation
      // LogInfo.logs("Trying AdvMod: " + example.id + " " + lemma1 + " " +
      // lemma2 + " ");
      extractAdvmodRelationFeature(sentence1, event1, sentence2, event2, fv);
    }

    String pos1 = event1.value(), pos2 = event2.value();
    // If second trigger is noun, the determiner related to it in dependency
    // tree.
    if (pos2.startsWith("NN")) {
      String determiner = Utils.getDeterminer(sentence2, event2);
      if (determiner != null) {
        fv.add("coref", "determinerBefore2:" + determiner);
      }
    }
    
    // POS tags of both events
    fv.add("syntactic", "POS:" + pos1 + "+" + pos2);
    fv.add("distance", "numSentencesInBetween:"
        + quantizedSentenceCount(sentenceBetweenEvents));
    fv.add("distance", "numWordsInBetween:" + quantizedWordCount(wordsBetweenEvents));

    SemanticGraph graph1 = sentence1
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord1 = Utils.findDependencyNode(sentence1, event1);
    SemanticGraph graph2 = sentence2
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord2 = Utils.findDependencyNode(sentence2, event2);
    String advMod = extractAdvModRelation(graph2, indexedWord2);
    if (advMod != null && !advMod.isEmpty()) {
      fv.add("advmod", "advMod:" + advMod);
    }

    // See if the two triggers share a common lemma as child in the dependency
    // graph. Currently not used.
    //shareSameLemmaAsChildFeature(fv, graph1, indexedWord1, graph2, indexedWord2);
    
    // Features if the two triggers are in the same sentence.
    if (sentenceBetweenEvents == 0) {
      // Lowest common ancestor between the two event triggers. Reduces score.
      Tree root = sentence1.get(TreeCoreAnnotations.TreeAnnotation.class);
      Tree lca = Trees.getLowestCommonAncestor(event1, event2, root);
      fv.add("syntactic", "lowestCommonAncestor:" + lca.value());

      // Dependency path if the event triggers are in the same sentence.
      // LogInfo.logs(example.id + " " + lemma1 + " " + lemma2);
      String deppath = Utils.getUndirectedDependencyPath_Events(sentence1,
          event1, event2);
      if (!deppath.isEmpty()) {
        if (!opts.useBaselineFeaturesOnly) {
          fv.add("syntactic", "deppath:" + deppath);
          fv.add("syntactic", "deppathwithword:"
              + Utils.getUndirectedDependencyPath_Events_WithWords(sentence1,
                  event1, event2));
        }
        // Does event1 dominate event2
        if (deppath.contains("->") && !deppath.contains("<-")) {
          fv.add("syntactic", "1dominates2");
        }

        if (deppath.contains("<-") && !deppath.contains("->")) {
          fv.add("syntactic", "2dominates1");
        }
      }

      // Extract mark relation
      // LogInfo.logs("Trying Marker: " + example.id + " " + lemma1 + " " +
      // lemma2 + " ");
      List<Pair<String, String>> markRelations = extractMarkRelation(input,
          sentence1, event1, event2, trig1);
      for (Pair<String, String> markRelation : markRelations) {
        // LogInfo.logs("MARKER ADDED: " + example.id + " " + lemma1 + " " +
        // lemma2 + " " + markRelation);
        if (opts.useBaselineFeaturesOnly) {
          fv.add("connective", "markRelation:" + markRelation.first());
        } else {
          fv.add("connective", "connector:" + markRelation.first());
          // In some cases, we don't have clusters for some relation.
          if (!markRelation.second().isEmpty())
            fv.add("connective", "connectorCluster:" + markRelation.second());
        }
      }

      // Extract PP relation
      // LogInfo.logs("Trying PP: " + example.id + " " + lemma1 + " " + lemma2 +
      // " ");
      List<Pair<String, String>> ppRelations = extractPPRelation(input, trig1,
          sentence1, event1, event2);
      for (Pair<String, String> ppRelation : ppRelations) {
        // LogInfo.logs("PP ADDED: " + example.id + " " + lemma1 + " " + lemma2
        // + " " + ppRelation);
        if (opts.useBaselineFeaturesOnly) {
          fv.add("connective", "PPRelation:" + ppRelation.first());
        } else {
          fv.add("connective", "connector:" + ppRelation.first());
          // In some cases, we don't have clusters (if we haven't included in
          // the list.
          if (!ppRelation.second().isEmpty()) {
            fv.add("connective", "connectorCluster:" + ppRelation.second());
          }
        }
      }
      
      // Do they share a common argument in the dependency tree? (if they are in
      // the same sentence)
      List<SemanticGraphEdge> edges1 = graph1.getOutEdgesSorted(indexedWord1);
      List<SemanticGraphEdge> edges2 = graph2.getOutEdgesSorted(indexedWord2);
      for (SemanticGraphEdge e1 : edges1) {
        for (SemanticGraphEdge e2 : edges2) {
          if (e1.getTarget().equals(e2.getTarget())) {
            fv.add("share", "shareChild:" + e1.getRelation() + "+"
                + e2.getRelation());
            break;
          }
        }
      }
    }
    fv.printFeatures();
    LogInfo.end_track();
    return fv;
  }
  
  public static FeatureVector getTriggerLabelFV(Input input, int trigger,
      String label) {
    FeatureVector fv = getTriggerFV(input, trigger);
    ArrayList<String> indicatorFeatures = fv.getIndicateFeatures();
    ArrayList<fig.basic.Pair<String, Double>> generalFeatures = fv
        .getGeneralFeatures();
    if(indicatorFeatures!=null)
      for (String f : indicatorFeatures) {
        f = label + "&" + f;
      }
    if(generalFeatures!=null)
      for (fig.basic.Pair<String, Double> f : generalFeatures) {
        f.setFirst(label + "&" + f.getFirst());
      }
    return fv;
  }

  public static FeatureVector getArgumentLabelFV(Input input, int trigger,
      int argument, String label) {
    FeatureVector fv = getArgumentFV(input, trigger, argument);
    ArrayList<String> indicateFeatures = fv.getIndicateFeatures();
    ArrayList<fig.basic.Pair<String, Double>> generalFeatures = fv
        .getGeneralFeatures();
    if(indicateFeatures!=null)
      for (String f : indicateFeatures) {
        f = label + "&" + f;
      }
    if(generalFeatures!=null)
      for (fig.basic.Pair<String, Double> f : generalFeatures) {
        f.setFirst(label + "&" + f.getFirst());
      }
    return fv;
  }

  public static FeatureVector getRelationLabelFV(Input input, int trig1,
      int trig2, String label) {
    FeatureVector fv = getRelationFV(input, trig1, trig2);
    ArrayList<String> indicateFeatures = fv.getIndicateFeatures();
    ArrayList<fig.basic.Pair<String, Double>> generalFeatures = fv
        .getGeneralFeatures();
    if(indicateFeatures!=null)
      for (String f : indicateFeatures) {
        f = label + "&" + f;
      }
    if(generalFeatures!=null)
      for (fig.basic.Pair<String, Double> f : generalFeatures) {
        f.setFirst(label + "&" + f.getFirst());
      }
    return fv;
  }
  
  public static FeatureVector getFeatures(Structure s){
    Input input = s.input;
    FeatureVector fv = new FeatureVector();
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      fv.add(getTriggerLabelFV(input, eventId, s.getTriggerLabel(eventId)), new AllFeatureMatcher());
    }
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      for (int entityId = 0; entityId < input
          .getNumberOfArgumentCandidates(eventId); entityId++) {
        fv.add(getArgumentLabelFV(input, eventId, entityId, s.getArgumentCandidateLabel(eventId, entityId)), new AllFeatureMatcher());
      }
    }
    for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = input.getEERelationCandidatePair(Id).getSource(); 
      int event2 = input.getEERelationCandidatePair(Id).getTarget(); 
      fv.add(getRelationLabelFV(input, event1, event2, s.getEERelationLabel(event1, event2)), new AllFeatureMatcher());
    }
    return fv;
  }
  
  private static void addAdvModFeature(CoreMap sentence, Tree event,
      FeatureVector fv, String currentPOS, String text) {
    SemanticGraph depGraph = sentence
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord word = Utils.findDependencyNode(sentence, event);
    for (SemanticGraphEdge e : depGraph.getOutEdgesSorted(word)) {
      if (e.getRelation().toString().equals("advmod")
          && (currentPOS.startsWith("VB") || Dictionary.nominalizations.contains(text)))
        //features.add("advmod:" + e.getTarget());
        fv.add("lexical", "advmod:" + e.getTarget());
    }
  }

  private static String getPosTagTrigram(CoreMap sentence, Tree event,
      String currentPOS) {
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    int currentTokenIndex = event.getSpan().getSource();
    String postagTrigram = "";
    if (currentTokenIndex > 0)
      postagTrigram += tokens.get(currentTokenIndex - 1).get(
          PartOfSpeechAnnotation.class);
    postagTrigram += currentPOS;
    if (currentTokenIndex < tokens.size() - 1)
      postagTrigram += tokens.get(currentTokenIndex + 1).get(
          PartOfSpeechAnnotation.class);
    return postagTrigram;
  }

  private static String buildParentCFGRule(Tree parent) {
    StringBuilder parentCFGRuleBuild = new StringBuilder(parent.value() + "->");
    for (Tree n : parent.getChildrenAsList()) {
      parentCFGRuleBuild.append(n.value() + "|");
    }
    String parentCFGRule = parentCFGRuleBuild.toString().trim();
    return parentCFGRule;
  }

  private static void extractAdvmodRelationFeature(CoreMap sentence1,
      Tree event1, CoreMap sentence2, Tree event2, FeatureVector fv) {
    List<Pair<String, String>> advModRelations = extractAdvModRelation(
        sentence1, sentence2, event1, event2);
    for (Pair<String, String> advModRelation : advModRelations) {
      // LogInfo.logs("ADVMOD ADDED: " + example.id + " " + lemma1 + " " +
      // lemma2 + " " + advModRelation);
      if (opts.useBaselineFeaturesOnly) {
        //features.add("advModRelation:" + advModRelation.first());
        fv.add("advmod", "advModRelation:" + advModRelation.first());
      } else {
        //features.add("connector:" + advModRelation.first());
        fv.add("advmod", "connector:" + advModRelation.first());
        // In some cases, we don't have clusters for some relation.
        if (!advModRelation.second().isEmpty()) {
          //features.add("connectorCluster:" + advModRelation.second());
          fv.add("advmod", "connectorCluster:" + advModRelation.second());
        }
      }
    }
  }

  private static void shareSameLemmaAsChildFeature(FeatureVector fv,
      SemanticGraph graph1, IndexedWord indexedWord1, SemanticGraph graph2,
      IndexedWord indexedWord2) {
    List<Pair<IndexedWord, String>> w1Children = new ArrayList<Pair<IndexedWord, String>>();
    for (SemanticGraphEdge e : graph1.getOutEdgesSorted(indexedWord1)) {
      w1Children.add(new Pair<IndexedWord, String>(e.getTarget(), e
          .getRelation().toString()));
    }

    for (SemanticGraphEdge e : graph2.getOutEdgesSorted(indexedWord2)) {
      for (Pair<IndexedWord, String> pair : w1Children) {
        if (e.getTarget().originalText().equals(pair.first.originalText())) {
          // System.out.println(indexedWord1 + ":" + indexedWord2 +
          // " share children. " + pair.first.originalText()
          // +":" +pair.second+ "+" +e.getRelation().toString());
          //features.add("shareSameLemmaAsChild:"+ pair.second+ "+" +e.getRelation().toString());
          fv.add("share", "shareSameLemmaAsChild:"+ pair.second+ "+" +e.getRelation().toString());
        }
        // System.out.println(indexedWord1 + ":" + indexedWord2 +
        // " share children. " + w.originalText());
        //features.add("shareSameLemmaAsChild");// + w.originalText());
        fv.add("share", "shareSameLemmaAsChild");
      }
    }
  }

  private static void lemmaEventPairFeature(CoreMap sentence1, Tree event1,
      CoreMap sentence2, Tree event2, FeatureVector fv) {
    CoreLabel event1CoreLabel = Utils.findCoreLabelFromTree(sentence1, event1);
    CoreLabel event2CoreLabel = Utils.findCoreLabelFromTree(sentence2, event2);
    String lemma1 = event1CoreLabel.lemma().toLowerCase();
    if (Dictionary.verbForms.containsKey(lemma1)) {
      lemma1 = Dictionary.verbForms.get(lemma1);
    }
    String lemma2 = event2CoreLabel.lemma().toLowerCase();
    if (Dictionary.verbForms.containsKey(lemma2)) {
      lemma2 = Dictionary.verbForms.get(lemma2);
    }
    //features.add("lemmas:" + lemma1 + "+" + lemma2);
    fv.add("lexical", "lemmas:" + lemma1 + "+" + lemma2);
    // Are the lemmas same?
    //features.add("eventLemmasSame:" + lemma1.equals(lemma2));
    fv.add("coref", "eventLemmasSame:" + lemma1.equals(lemma2));
  }

  private static void addWordsBetweenEventsFeature(Input input, Tree event1,
      Tree event2, FeatureVector fv, int sentenceBetweenEvents) {
    List<Pair<String, String>> wordsInBetween = AnnotationUtils.findWordsInBetween(input,
        event1, event2);
    for (int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
      Pair<String, String> pair = wordsInBetween.get(wordCounter);
      String POS = pair.second, word = pair.first;
      String POS2;
      if (wordCounter < wordsInBetween.size() - 1)
        POS2 = wordsInBetween.get(wordCounter + 1).second;
      else
        POS2 = "";

      String word2;
      if (wordCounter < wordsInBetween.size() - 1)
        word2 = wordsInBetween.get(wordCounter + 1).first;
      else
        word2 = "";

      if (!Dictionary.TemporalConnectives.contains(word.toLowerCase())) {
        if (POS.startsWith("VB") && POS2.equals("IN")) {
          //features.add("wordsInBetween:" + word + " " + word2);
          fv.add("lexical", "wordsInBetween:" + word + " " + word2);
          wordCounter++;
        } else{
          //features.add("wordsInBetween:" + word);
          fv.add("lexical", "wordsInBetween:" + word);
        }
      } else {
        if (sentenceBetweenEvents < 2) {
          if (opts.useBaselineFeaturesOnly) {
            //features.add("temporalConnective:" + word.toLowerCase());
            fv.add("connective", "temporalConnective:" + word.toLowerCase());
          } else {
            //features.add("connector:" + word.toLowerCase());// connective
            fv.add("connective", "connector:" + word.toLowerCase());
            if (AdvModClusters.containsKey(word.toLowerCase())) {
              /*features.add("connectorCluster:"
                  + AdvModClusters.get(word.toLowerCase()));*/
              fv.add("connective", "connectorCluster:"
                  + AdvModClusters.get(word.toLowerCase()));
            }
          }
        }
      }
    }
  }


  
  private static void ClusterSetup() {
    MarkAndPPClusters.put("if", RelationType.PreviousEvent.toString());
    MarkAndPPClusters.put("until", RelationType.NextEvent.toString());
    MarkAndPPClusters.put("after", RelationType.PreviousEvent.toString());
    MarkAndPPClusters.put("as", RelationType.CotemporalEvent.toString());
    MarkAndPPClusters.put("because", RelationType.Causes.toString());
    MarkAndPPClusters.put("before", RelationType.NextEvent.toString());
    MarkAndPPClusters.put("since", RelationType.Causes.toString());
    MarkAndPPClusters.put("so", RelationType.Caused.toString());
    MarkAndPPClusters.put("while", RelationType.CotemporalEvent.toString());
    MarkAndPPClusters.put("during", RelationType.SuperEvent.toString());
    MarkAndPPClusters.put("upon", RelationType.PreviousEvent.toString());

    AdvModClusters.put("then", RelationType.PreviousEvent.toString());
    AdvModClusters.put("thus", RelationType.Causes.toString());
    AdvModClusters.put("also", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("eventually", RelationType.PreviousEvent.toString());
    AdvModClusters.put("meanwhile", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("thereby", RelationType.Causes.toString());
    AdvModClusters.put("finally", RelationType.PreviousEvent.toString());
    AdvModClusters.put("first", RelationType.SuperEvent.toString());
    AdvModClusters.put("hence", RelationType.Causes.toString());
    AdvModClusters.put("later", RelationType.PreviousEvent.toString());
    AdvModClusters.put("next", RelationType.PreviousEvent.toString());
    AdvModClusters.put("simultaneously",
        RelationType.CotemporalEvent.toString());
    AdvModClusters.put("subsequently", RelationType.PreviousEvent.toString());
    AdvModClusters.put("if", RelationType.NextEvent.toString());
    AdvModClusters.put("until", RelationType.PreviousEvent.toString());
    AdvModClusters.put("after", RelationType.NextEvent.toString());
    AdvModClusters.put("as", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("because", RelationType.Caused.toString());
    AdvModClusters.put("so", RelationType.Causes.toString());
    AdvModClusters.put("result", RelationType.Causes.toString());
    AdvModClusters.put("results", RelationType.Causes.toString());
    AdvModClusters.put("lead", RelationType.Causes.toString());
    AdvModClusters.put("leads", RelationType.Causes.toString());
    AdvModClusters.put("cause", RelationType.Causes.toString());
    AdvModClusters.put("causes", RelationType.Causes.toString());
    AdvModClusters.put("while", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("during", RelationType.SubEvent.toString());
    AdvModClusters.put("upon", RelationType.NextEvent.toString());
    AdvModClusters.put("include", RelationType.SuperEvent.toString());
    AdvModClusters.put("includes", RelationType.SuperEvent.toString());
    AdvModClusters.put("included", RelationType.SuperEvent.toString());
    AdvModClusters.put("including", RelationType.SuperEvent.toString());
    AdvModClusters.put("begin", RelationType.SuperEvent.toString());
    AdvModClusters.put("begins", RelationType.SuperEvent.toString());
    AdvModClusters.put("began", RelationType.SuperEvent.toString());
    AdvModClusters.put("begun", RelationType.SuperEvent.toString());
    AdvModClusters.put("beginning", RelationType.SuperEvent.toString());
    AdvModClusters.put("start", RelationType.SuperEvent.toString());
    AdvModClusters.put("starts", RelationType.SuperEvent.toString());
    AdvModClusters.put("started", RelationType.SuperEvent.toString());
    AdvModClusters.put("starting", RelationType.SuperEvent.toString());
    AdvModClusters.put("subsequent", RelationType.PreviousEvent.toString());
    AdvModClusters.put("previously", RelationType.NextEvent.toString());
    AdvModClusters.put("previous", RelationType.NextEvent.toString());
  }

  private static String quantizedSentenceCount(int numSentences) {
    if (numSentences == 0) {
      return "None";
    } else if (numSentences == 1) {
      return "Low";
    } else if (numSentences == 2 || numSentences == 3) {
      return "Medium";
    }
    return "High";
  }

  private static String quantizedWordCount(int numWords) {
    if (numWords <= 4) {
      return "Low";
    } else if (numWords <= 8) {
      return "Medium";
    } else if (numWords <= 15) {
      return "High";
    }
    return "VeryHigh";
  }

  private static List<Pair<String, String>> extractAdvModRelation(
      CoreMap sentence1, CoreMap sentence2, Tree event1, Tree event2) {

    SemanticGraph graph1 = sentence1
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    SemanticGraph graph2 = sentence2
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord1 = Utils.findDependencyNode(sentence1, event1), indexedWord2 = Utils
        .findDependencyNode(sentence2, event2);
    List<Pair<String, String>> advModRelations = new ArrayList<Pair<String, String>>();
    if (indexedWord1 == null || indexedWord2 == null) {
      return advModRelations;
    }
    int event1Index = indexedWord1.index(), event2Index = indexedWord2.index();

    Pair<String, String> advModRelation1 = extractAdvModRelation(graph1,
        indexedWord1, indexedWord2, event1Index, event2Index);
    Pair<String, String> advModRelation2 = extractAdvModRelation(graph2,
        indexedWord2, indexedWord1, event1Index, event2Index);
    if (advModRelation1 != null) {
      // advModRelations.add(new Pair<String, String>(advModRelation1.first() +
      // "_1", advModRelation1.second()));
    }

    if (advModRelation2 != null) {
      advModRelations.add(new Pair<String, String>(advModRelation2.first(),
          advModRelation2.second()));
    }

    return advModRelations;
  }

  private static Pair<String, String> extractAdvModRelation(
      SemanticGraph graph, IndexedWord indexedWordThis,
      IndexedWord indexedWordThat, int event1Index, int event2Index) {
    for (SemanticGraphEdge e : graph.getOutEdgesSorted(indexedWordThis)) {
      if (e.getRelation().getShortName().equals("advmod")) {
        String advModName = e.getTarget().lemma().toLowerCase();

        // LogInfo.logs("AdvMod found '" + indexedWordThis + "':" + advModName);
        if (AdvModClusters.containsKey(advModName))
          return new Pair<String, String>(advModName,
              AdvModClusters.get(advModName));
        else
          return new Pair<String, String>(advModName, "");
      }
    }
    return null;
  }

  private static String extractAdvModRelation(SemanticGraph graph,
      IndexedWord indexedWord) {
    for (SemanticGraphEdge e : graph.getOutEdgesSorted(indexedWord)) {
      if (e.getRelation().getShortName().equals("advmod")) {
        return e.getTarget().originalText();
      }
    }
    return null;
  }

  private static List<Pair<String, String>> extractPPRelation(Input input,
      int trig1, CoreMap sentence1, Tree event1, Tree event2) {
    CoreMap sentence = sentence1;
    Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);

    int event1Index = event1.nodeNumber(root), event2Index = event2
        .nodeNumber(root);
    // Are the event triggers part of a Prepositional phrase individually (one
    // feature each)?

    List<Pair<String, String>> ppRelations = new ArrayList<Pair<String, String>>();

    Pair<String, String> ppRelation1 = extractPPRelation(input, trig1,
        sentence1, event1, event2, event1Index, event2Index);
    Pair<String, String> ppRelation2 = extractPPRelation(input, trig1,
        sentence1, event2, event1, event1Index, event2Index);
    if (ppRelation1 != null) {
      ppRelations.add(new Pair<String, String>(ppRelation1.first() + "_1",
          ppRelation1.second()));
    }

    if (ppRelation2 != null) {
      ppRelations.add(new Pair<String, String>(ppRelation2.first() + "_2",
          ppRelation2.second()));
    }

    return ppRelations;
  }

  private static Pair<String, String> extractPPRelation(Input input, int trig1,
      CoreMap sentence1, Tree thisEvent, Tree thatEvent, int event1Index,
      int event2Index) {
    Tree root = sentence1.get(TreeCoreAnnotations.TreeAnnotation.class);
    Tree node = thisEvent;
    // root.pennPrint();
    // Are the event triggers part of a Prepositional phrase individually (one
    // feature each)?
    while (!node.value().equals("ROOT") && !node.value().equals("S")
        && !node.value().equals("SBAR")) {
      if (node.value().equals("PP")) {
        for (Tree ponode : node.postOrderNodeList()) {
          if (ponode.isPreTerminal()
              && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
            // Other event should not be in the same PP
            if (node.dominates(thatEvent)) {
              // LogInfo.logs("PP dominates other event");
              return null;
            }
            Tree lca = Trees
                .getLowestCommonAncestor(thisEvent, thatEvent, root);
            // LogInfo.logs("ANCESTOR " + lca.value());
            List<String> path = Trees.pathNodeToNode(lca, ponode, lca);
            path.remove(path.size() - 1);
            path.remove(0);
            path.remove(0);

            // LogInfo.logs("PPPATH " + thisEvent + " " + path);
            // if(path.contains("down-S") || path.contains("down-SBAR")) {
            if (path.contains("down-S") || path.contains("down-SBAR")) {
              // LogInfo.logs("Contains S or SBAR");
              // return null;
            }

            path = Trees.pathNodeToNode(lca, thatEvent, lca);
            path.remove(path.size() - 1);
            path.remove(0);
            path.remove(0);

            // LogInfo.logs("PPPATH " + thatEvent + " " + path);
            if (path.contains("down-S") || path.contains("down-SBAR")) {
              // LogInfo.logs("Contains S or SBAR");
              // return null;
            }

            String ppName = ponode.firstChild().value().toLowerCase();
            int ppIndex = ponode.firstChild().nodeNumber(root);
            if (AnnotationUtils.isFirstEventInSentence(input, trig1) && ppIndex < event1Index) {
              // LogInfo.logs("PP before :" +ppName);
              if (MarkAndPPClusters.containsKey(ppName))
                return new Pair<String, String>(ppName,
                    MarkAndPPClusters.get(ppName));
              else
                return new Pair<String, String>(ppName, "");
            } else if (ppIndex > event1Index && ppIndex < event2Index) {
              // LogInfo.logs("PP between :" +ppName);
              if (MarkAndPPClusters.containsKey(ppName))
                return new Pair<String, String>(ppName,
                    Utils.getInverseRelation(MarkAndPPClusters.get(ppName)));
              else
                return new Pair<String, String>(ppName, "");
            } else if (ppIndex > event2Index) {
              // LogInfo.logs("PP after");
            }

            break;
          }
        }
      }
      node = node.parent(root);
    }
    return null;
  }

  private static List<Pair<String, String>> extractMarkRelation(Input input,
      CoreMap sentence1, Tree event1, Tree event2, int trig1) {
    CoreMap sentence = sentence1;
    SemanticGraph graph = sentence
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord1 = Utils.findDependencyNode(sentence, event1), indexedWord2 = Utils
        .findDependencyNode(sentence, event2);
    List<Pair<String, String>> markRelations = new ArrayList<Pair<String, String>>();
    if (indexedWord1 == null || indexedWord2 == null) {
      return markRelations;
    }
    int event1Index = indexedWord1.index(), event2Index = indexedWord2.index();
    
    Pair<String, String> markRelation1 = extractMarkRelation(graph, input,
        trig1, indexedWord1, indexedWord2, event1Index, event2Index);
    Pair<String, String> markRelation2 = extractMarkRelation(graph, input,
        trig1, indexedWord2, indexedWord1, event1Index, event2Index);
    if (markRelation1 != null) {
      markRelations.add(new Pair<String, String>(markRelation1.first() + "_1",
          markRelation1.second()));
    }

    if (markRelation2 != null) {
      markRelations.add(new Pair<String, String>(markRelation2.first() + "_2",
          markRelation2.second()));
    }

    return markRelations;
  }

  private static Pair<String, String> extractMarkRelation(SemanticGraph graph,
      Input input, int trig1, IndexedWord indexedWordThis,
      IndexedWord indexedWordThat, int event1Index, int event2Index) {
    for (SemanticGraphEdge e : graph.getOutEdgesSorted(indexedWordThis)) {
      if (e.getRelation().getShortName().equals("mark")) {
        int markIndex = e.getTarget().index();
        String markerName = e.getTarget().lemma().toLowerCase();

        // Check if the source of incoming edge (advcl, ccmod, dep) is same as
        // indexedWordThat or parent of indexedWordThat
        if (graph.getIncomingEdgesSorted(indexedWordThis).size() > 0) {
          SemanticGraphEdge edge = graph
              .getIncomingEdgesSorted(indexedWordThis).get(0);

          IndexedWord parent = edge.getSource();

          if (parent.index() == indexedWordThat.index()
              || isInSameDependencyClauseAndChild(graph, parent,
                  indexedWordThat)) {
            if (AnnotationUtils.isFirstEventInSentence(input, trig1) && markIndex < event1Index) {
              // LogInfo.logs("Marker before :" +markerName);
              if (MarkAndPPClusters.containsKey(markerName))
                return new Pair<String, String>(markerName,
                    MarkAndPPClusters.get(markerName));
              else
                return new Pair<String, String>(markerName, "");
            } else if (markIndex < event2Index) {
              // LogInfo.logs("Marker between :" +markerName);
              if (MarkAndPPClusters.containsKey(markerName))
                return new Pair<String, String>(markerName,
                    Utils.getInverseRelation(MarkAndPPClusters.get(markerName)));
              else
                return new Pair<String, String>(markerName, "");
            } else {
              LogInfo.logs("Marker after");
            }
          }
        }
      }
    }
    return null;
  }

  private static boolean isInSameDependencyClauseAndChild(SemanticGraph graph,
      IndexedWord parent, IndexedWord word) {
    List<SemanticGraphEdge> edges = graph.getShortestDirectedPathEdges(parent,
        word);
    if (edges == null)
      return false;

    for (SemanticGraphEdge edge : edges) {
      if (Dictionary.diffClauseRelations.contains(edge.getRelation().getShortName()))
        return false;
    }
    return true;
  }
}
