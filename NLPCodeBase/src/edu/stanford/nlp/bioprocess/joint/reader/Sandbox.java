package edu.stanford.nlp.bioprocess.joint.reader;

import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class Sandbox {
  
  public static void main(String[] args) {
    String text = "this is a text. This text is to understand how to convert from characters to words. hello to you all";
    Properties props = new Properties();
    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
    StanfordCoreNLP processor = new StanfordCoreNLP(props, false);
    Annotation annotation = new Annotation(text);
    processor.annotate(annotation);
    List<CoreLabel> tokens = annotation.get(TokensAnnotation.class);
    int i = 0;
    for(CoreLabel token: tokens) {
      Integer begin = token.get(CharacterOffsetBeginAnnotation.class);
      Integer end = token.get(CharacterOffsetEndAnnotation.class);
      System.out.println(token+"\t"+i+"\t"+begin+"\t"+end);
      i++;
    }
    
        
  }

}
