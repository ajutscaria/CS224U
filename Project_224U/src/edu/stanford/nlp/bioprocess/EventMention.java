package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.util.CoreMap;

/**
 * Each entity mention is described by a span of text
 * 
 * @author Aju
 */
public class EventMention extends ArgumentMention {
 
  private static final long serialVersionUID = -2745903102654191527L;
  private List<ArgumentRelation> arguments;
  public EventType eventType;
 /**
  * 
  */
  public EventMention(String objectId, CoreMap sentence, Span span) {
     super(objectId, sentence, span);
     arguments = new ArrayList<ArgumentRelation>();
  }

  public void addArgument(ArgumentMention mention, RelationType type) {
    arguments.add(new ArgumentRelation(mention, type));
  }
  
  public List<ArgumentRelation> getArguments() {
    return arguments;
  }
  
  public String prettyPrint() {
	  StringBuilder strBld = new StringBuilder();
	  strBld.append(String.format("%-20s", this.getValue()));
	  if(this.arguments.size() > 0) {
		  strBld.append("\tRelations -");
		  for(ArgumentRelation rel:arguments) {
			  strBld.append(String.format("   %s : '%s'", rel.type.toString(), rel.mention.getValue()));
		  }
	  }
	  return strBld.toString();
  }
  
  private static int MENTION_COUNTER = 0;

  /**
   * Creates a new unique id for an entity mention
   * @return the new id
   */
  public static synchronized String makeUniqueId() {
    MENTION_COUNTER ++;
    return "EntityMention-" + MENTION_COUNTER;
  }
}

