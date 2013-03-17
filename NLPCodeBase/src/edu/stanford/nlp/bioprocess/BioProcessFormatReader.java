package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.ie.machinereading.GenericDataSetReader;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import fig.basic.LogInfo;

public class BioProcessFormatReader extends GenericDataSetReader {
  protected static final String TEXT_EXTENSION = ".txt";
  protected static final String ANNOTATION_EXTENSION = ".ann";

  protected static final String THEME_TYPE_PREFIX = "T";
  protected static final String EVENT_TYPE = "Event";
  protected static final String ENTITY_TYPE = "Entity", STATIC_ENTITY_TYPE = "Static-Event";
  protected static final String TYPE_NEXT_EVENT = "next-event", TYPE_RESULT = "result", TYPE_AGENT = "agent", TYPE_ORIGIN = "origin",
      TYPE_COTEMPORAL_EVENT = "cotemporal", TYPE_SAME_EVENT = "same-event", TYPE_SUPER_EVENT = "super-event", TYPE_ENABLES = "enables",
      TYPE_DESTINATION = "destination", TYPE_LOCATION = "location", TYPE_THEME = "theme", TYPE_SAME_ENTITY = "same-entity",
      TYPE_TIME = "time", TYPE_RAW_MATERIAL = "raw-material", TYPE_CAUSE ="cause";
  
  //static int StaticEventCount =0;
 
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
      LogInfo.logs(file);
      
      String rawText = IOUtils.slurpFile(new File(path + file));
      Example example = new Example();
      example.data = rawText;
      example.id = file.replace(TEXT_EXTENSION, "");
      example.gold = createAnnotation(path + file);
      
      example.prediction = example.gold.copy();
      example.prediction.set(EntityMentionsAnnotation.class, new ArrayList<EntityMention>());
      example.prediction.set(EventMentionsAnnotation.class, new ArrayList<EventMention>());
      examples.add(example);
      //break;
    }
    //LogInfo.logs("Number of static events - " + StaticEventCount);
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
    //Setting spans of the tree nodes of each sentence to avoid running into excpetions where we encounter sentences without any entities later.
    for(CoreMap sentence:sentences) {
    	Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    	
    	List<EventMention> eventMentions = new ArrayList<EventMention>();
        sentence.set(EventMentionsAnnotation.class, eventMentions);
        
        List<EntityMention> entityMentions = new ArrayList<EntityMention>();
        sentence.set(EntityMentionsAnnotation.class, entityMentions);
        
    	syntacticParse.setSpans();
    }
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
            CoreMap sentence = Utils.getContainingSentence(sentences, begin, end);
            Span span = Utils.getSpanFromSentence(sentence, begin, end);
            
            if(type.equals(EVENT_TYPE) || type.equals(STATIC_ENTITY_TYPE)) {
              m = new EventMention(desc, sentence, span);
              //LogInfo.logs("\t\t\t\t" + line);
              Tree eventRoot = Utils.getEventNode(sentence, (EventMention)m);
              IndexedWord head = Utils.findDependencyNode(sentence, eventRoot);
              //Utils.findDepthInDependencyTree(sentence, eventRoot);
              m.setHeadInDependencyTree(head);
              m.setTreeNode(eventRoot);
            }
            else {
              //LogInfo.logs(line);
              m = new EntityMention(desc, sentence, span);
              Tree entityRoot = Utils.getEntityNode(sentence, (EntityMention)m);
              m.setTreeNode(entityRoot);
              Utils.addAnnotation(document, (EntityMention)m);
              IndexedWord head = Utils.findDependencyNode(sentence, entityRoot);
              m.setHeadInDependencyTree(head);
              m.setHeadTokenSpan(Utils.findEntityHeadWord((EntityMention)m));
              //LogInfo.logs(m.getHeadToken().originalText());
            }
            mentions.put(desc, m);
            break;
          case 'E':
        	String[] parameters = splits[1].split(" ");
        	String[] splts = parameters[0].split(":");
        		
   			((EventMention)mentions.get(splts[1])).eventType = splts[0].equals(EVENT_TYPE) ? EventType.Event : EventType.StaticEvent;
   			//if(splts[0].equals(STATIC_ENTITY_TYPE))
   			//	StaticEventCount += 1;
    		mentions.put(desc, mentions.get(splts[1]));
    		mentions.remove(splts[1]);
         } 
      }
      reader.seek(0);
      while((line = reader.readLine())!=null) {
        String[] splits = line.split("\t");
        String desc = splits[0];
        switch(desc.charAt(0)) {
          case 'E':
            String[] parameters = splits[1].split(" ");
            EventMention event = (EventMention)mentions.get(desc);
            for(String parameter:parameters) {
              String[] keyValue = parameter.split(":");
              switch(keyValue[0]) {
                case TYPE_AGENT:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Agent);
                    break;
                case TYPE_ORIGIN:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Origin);
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
                case TYPE_RAW_MATERIAL:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.RawMaterial);
                    break;
                case TYPE_THEME:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Theme);
                    break;
                case TYPE_TIME:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Time);
                    break;
                case TYPE_COTEMPORAL_EVENT:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.CotemporalEvent);
                    ((EventMention)mentions.get(keyValue[1])).addArgument(event, RelationType.CotemporalEvent);
                    break;
                case TYPE_NEXT_EVENT:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.NextEvent);
                    break;
                case TYPE_SAME_EVENT:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.SameEvent);
                    break;
                case TYPE_CAUSE:
                    event.addArgument(mentions.get(keyValue[1]), RelationType.Cause);
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
                    else if(keyValue[0].startsWith(TYPE_LOCATION))
                  	    event.addArgument(mentions.get(keyValue[1]), RelationType.Location);
                    else if(keyValue[0].startsWith(TYPE_DESTINATION))
                    	event.addArgument(mentions.get(keyValue[1]), RelationType.Destination);
                    else if(keyValue[0].startsWith(TYPE_THEME))
                    	event.addArgument(mentions.get(keyValue[1]), RelationType.Theme);
                    else if(keyValue[0].startsWith(TYPE_AGENT))
                    	event.addArgument(mentions.get(keyValue[1]), RelationType.Agent);
                    else if(keyValue[0].startsWith(TYPE_ORIGIN))
                    	event.addArgument(mentions.get(keyValue[1]), RelationType.Origin);
              }
            }
            Utils.addAnnotation(document, event);
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
}
