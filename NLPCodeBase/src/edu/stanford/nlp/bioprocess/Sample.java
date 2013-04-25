package edu.stanford.nlp.bioprocess;

import java.util.Properties;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefChain.CorefMention;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.logging.Redwood;
import fig.basic.LogInfo;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class Sample {
  
  public static void main(String[] args) {
 // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
    Properties props = new Properties();
    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    
    // read some text in the text variable
    String text = "The virus entered the cell. Then, it attacked the nucleus in New York";
    LogInfo.logs("HELLOO");
    // create an empty Annotation just with the given text
    Annotation document = new Annotation(text);
    
 // run all Annotators on this text
    pipeline.annotate(document);
    
 // these are all the sentences in this document
    // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
    LogInfo.logs("SDF"+sentences);
    for(CoreMap sentence: sentences) {
      // traversing the words in the current sentence
      // a CoreLabel is a CoreMap with additional token-specific methods
      for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
        // this is the text of the token
        String word = token.get(TextAnnotation.class);
        Redwood.log("Word: " + word);
        // this is the POS tag of the token
        String pos = token.get(PartOfSpeechAnnotation.class);
        Redwood.log("POS: " + pos);
        // this is the NER label of the token
        String ne = token.get(NamedEntityTagAnnotation.class);
        Redwood.log("NE: " + ne);
      }

      // this is the parse tree of the current sentence
      Tree tree = sentence.get(TreeAnnotation.class);
      Redwood.log("Tree:");
      Redwood.log(tree);

      // this is the Stanford dependency graph of the current sentence
      SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
      Redwood.log("Semnatic graph:");
      Redwood.log(dependencies);
      IndexedWord root = dependencies.getRoots().iterator().next();
      List<IndexedWord> children = dependencies.getChildList(root);
      Redwood.log("Root: " + root);
      for(IndexedWord child: children)
        Redwood.log(child);
    }

    // This is the coreference link graph
    // Each chain stores a set of mentions that link to each other,
    // along with a method for getting the most representative mention
    // Both sentence and token offsets start at 1!
    Map<Integer, CorefChain> graph = 
      document.get(CorefChainAnnotation.class);
    Redwood.log("Coref:");
    Redwood.log(graph);
    for(CorefChain chain: graph.values()) {
      Map<IntPair, Set<CorefMention>> mentionMap = chain.getMentionMap();
      Redwood.log("Mention map:");
      Redwood.log(mentionMap);
    }
  }

}
