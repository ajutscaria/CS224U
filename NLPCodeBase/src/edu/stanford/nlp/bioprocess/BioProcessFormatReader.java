package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.sun.org.apache.xml.internal.security.keys.content.KeyValue;


import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.ie.machinereading.GenericDataSetReader;

//import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
//import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.EntityMentionsAnnotation;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.TypesafeMap.Key;
import edu.stanford.nlp.util.logging.Redwood;
import edu.stanford.nlp.util.logging.Redwood.RedwoodChannels;

public class BioProcessFormatReader extends GenericDataSetReader {
  protected static final String TEXT_EXTENSION = ".txt";
  protected static final String ANNOTATION_EXTENSION = ".ann";

  protected static final String THEME_TYPE_PREFIX = "T";
  protected static final String EVENT_TYPE = "Event";
  protected static final String ENTITY_TYPE = "Entity", STATIC_ENTITY_TYPE = "Static-Event";
  protected static final String TYPE_NEXT_EVENT = "next-event", TYPE_RESULT = "result", TYPE_AGENT = "agent", 
      TYPE_COTEMPORAL_EVENT = "cotemporal", TYPE_SAME_EVENT = "same-event", TYPE_SUPER_EVENT = "super-event", TYPE_ENABLES = "enables",
      TYPE_DESTINATION = "destination", TYPE_LOCATION = "location", TYPE_THEME = "theme", TYPE_SAME_ENTITY = "same-event";
  
  private Map<String, EntityMention> idsToEntities;
  private Map<String, EventMention> idsToEvents;
 
  //TODO - How is this used?
  /** keeps track which entity is embedded (key) into which entity (value) */
  Map<String, String> inclusionLinks;
  
  public final List<Example> parseFolder(String path) throws IOException {
    List<Example> examples = new ArrayList<Example>();
    File folder = new File(path);
    FilenameFilter textFilter = new FilenameFilter() {
      public boolean accept(File dir, String name) {
        String lowercaseName = name.toLowerCase();
        if (lowercaseName.endsWith(TEXT_EXTENSION)) {
          return true;
        } else {
          return false;
        }
      }
    };
    for(String file:folder.list(textFilter)){
      System.out.println(file);
      String rawText = IOUtils.slurpFile(new File(path + file));
      Example example = new Example();
      example.data = rawText;
      example.id = file.replace(TEXT_EXTENSION, "");
      example.gold = createAnnotation(path + file);
      examples.add(example);
      
      //for(CoreMap sentence: example.gold.get(SentencesAnnotation.class))
      //  System.out.println(sentence.get(CharacterOffsetBeginAnnotation.class));
      
    /*for(CoreMap sentence: document.get(SentencesAnnotation.class)) {
        if(document.get(EntityMentionsAnnotation.class) == null) {
          List<EntityMention> ents = new ArrayList<EntityMention>();
          //ents.add(new EntityMention("idd",sentence, null, null, null, null, null));
          ents.add(new EntityMention("idd",sentence));
          document.set(EntityMentionsAnnotation.class, ents);
        }
        else {
          EntityMention me = new EntityMention("idd",sentence);
          document.get(EntityMentionsAnnotation.class).add(me);
        }
        // traversing the words in the current sentence
        // a CoreLabel is a CoreMap with additional token-specific methods
        for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
          // this is the text of the token
          String word = token.get(TextAnnotation.class);
          Redwood.log("Word: " + word);
          /*
          // this is the POS tag of the token
          String pos = token.get(PartOfSpeechAnnotation.class);
          Redwood.log("POS: " + pos);
          // this is the NER label of the token
          String ne = token.get(NamedEntityTagAnnotation.class);
          Redwood.log("NE: " + ne);
          
        }
      }
        
        // this is the parse tree of the current sentence
        //Tree tree = sentence.get(TreeAnnotation.class);
        //for(CoreLabel label: tree.taggedLabeledYield())
        //    System.out.println(label.lemma());
        //Redwood.log("Tree:");
        //Redwood.log(tree);
        //printYield(tree);
        // this is the Stanford dependency graph of the current sentence
        
      
        SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);

        Redwood.log("Semnatic graph:");
        Redwood.log(dependencies);
        IndexedWord root = dependencies.getRoots().iterator().next();
        List<IndexedWord> children = dependencies.getChildList(root);
        Redwood.log("Root: " + root);
        for(IndexedWord child: children) {
          Redwood.log(child);
          List<EntityMention> ents = new ArrayList<EntityMention>();
          //ents.add(new EntityMention("idd",sentence, null, null, null, null, null));
          ents.add(new EntityMention("idd",sentence));
          child.set(EntityMentionsAnnotation.class, ents);
          //child.set(EntityMentionsAnnotation.class, ents);
        }
      }
      System.out.println(document.get(EntityMentionsAnnotation.class).size());*/
    }
    return examples;
  }
  
  private Annotation createAnnotation(String fileName) {
    String rawText = "";
    try {
      rawText = IOUtils.slurpFile(new File(fileName));
    } catch (IOException e1) {
      // TODO Auto-generated catch block
      e1.printStackTrace();
    }
    Annotation document = new Annotation(rawText);
    processor.annotate(document);
    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
    HashMap<String, ArgumentMention> mentions = new HashMap<String, ArgumentMention>();
    try {
      RandomAccessFile reader = new RandomAccessFile(new File(fileName.replace(TEXT_EXTENSION, ANNOTATION_EXTENSION)), "r");
      String line;
      while((line = reader.readLine())!=null) {
        String[] splits = line.split("\t");
        String desc = splits[0];
        switch(desc.charAt(0)) {
          case 'T':
            String[] argumentDetails = splits[1].split(" ");
            String type = argumentDetails[0];
            ArgumentMention m;
            
            int begin = Integer.parseInt(argumentDetails[1]), end =  Integer.parseInt(argumentDetails[2]);
            //System.out.println(begin + ":" + end + "-" + line);
            CoreMap sentence = getContainingSentence(sentences, begin, end);
            Span span = getSpanFromSentence(sentence, begin, end);
            
            if(type.equals(EVENT_TYPE) || type.equals(STATIC_ENTITY_TYPE)) {
              m = new EventMention(desc, sentence, span);
            }
            else {
              m = new EntityMention(desc, sentence, span);
              addAnnotation(document, (EntityMention)m);
            }
            mentions.put(desc, m);
            break;
        }
      }
      reader.seek(0);
      while((line = reader.readLine())!=null) {
    	//System.out.println(line);
        String[] splits = line.split("\t");
        String desc = splits[0];
        switch(desc.charAt(0)) {
          case 'E':
            String[] parameters = splits[1].split(" ");
            EventMention event = null;
            for(String parameter:parameters) {
              String[] keyValue = parameter.split(":");
              switch(keyValue[0]) {
                case EVENT_TYPE:
                  event = (EventMention)mentions.get(keyValue[1]);
                  break;
                case STATIC_ENTITY_TYPE:
                  event = (EventMention)mentions.get(keyValue[1]);
                  break;
                case TYPE_AGENT:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.Agent);
                  break;
                case TYPE_DESTINATION:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.Destination);
                  break;
                case TYPE_LOCATION:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Location);
                    break;
                case TYPE_RESULT:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.Result);
                  break;
                case TYPE_THEME:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Theme);
                    break;
                case TYPE_COTEMPORAL_EVENT:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.CotemporalEvent);
                  break;
                case TYPE_NEXT_EVENT:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.NextEvent);
                  break;
                case TYPE_SAME_EVENT:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.SameEvent);
                  break;
                case TYPE_SUPER_EVENT:
                  event.addArgument(mentions.get(keyValue[1]), RelationType.SuperEvent);
                  break;
                case TYPE_ENABLES:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Enables);
                    break;
                default:
                  if(keyValue[0].startsWith(TYPE_RESULT))
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Result);
              }
            }
            addAnnotation(document, event);
            break;
          case '*':
        	  String[] params = splits[1].split(" ");
        	  if(params[0].equals(TYPE_SAME_ENTITY)) {
        		  String entity1 = params[1], entity2 = params[2];
        		  ((EntityMention)mentions.get(entity1)).addRelation((EntityMention)mentions.get(entity2), RelationType.SameEntity);
        		  ((EntityMention)mentions.get(entity2)).addRelation((EntityMention)mentions.get(entity1), RelationType.SameEntity);
        	  }
        }
      }
      
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return document;
  }
  
  private CoreMap getContainingSentence(List<CoreMap> sentences, int begin, int end) {
    //System.out.println("----" + begin + ":" + end);
    for(CoreMap sentence:sentences) {
      //System.out.println(sentence.get(CharacterOffsetBeginAnnotation.class) + ":"+sentence.get(CharacterOffsetEndAnnotation.class));
      if(sentence.get(CharacterOffsetBeginAnnotation.class) <= begin && sentence.get(CharacterOffsetEndAnnotation.class) >= end)
        return sentence;
    }
    return null;
  }
  
  private Span getSpanFromSentence(CoreMap sentence, int begin, int end) {
    Span span = new Span();
    //System.out.println(sentence);
    for(CoreLabel label:sentence.get(TokensAnnotation.class)) {
      if(label.beginPosition() == begin)
        span.setStart(label.index() - 1);
      if(label.endPosition() == end)
        span.setEnd(label.index());
    }
    return span;
  }
  
  private void addAnnotation(Annotation document, EntityMention entity) {
    if(document.get(EntityMentionsAnnotation.class) == null) {
      List<EntityMention> mentions = new ArrayList<EntityMention>();
      mentions.add(entity);
      document.set(EntityMentionsAnnotation.class, mentions);
    }
    else
      document.get(EntityMentionsAnnotation.class).add(entity);
  }
  
  private void addAnnotation(Annotation document, EventMention event) {
    if(document.get(EventMentionsAnnotation.class) == null) {
      List<EventMention> mentions = new ArrayList<EventMention>();
      mentions.add(event);
      document.set(EventMentionsAnnotation.class, mentions);
    }
    else
      document.get(EventMentionsAnnotation.class).add(event);
  }
}
