package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.util.CoreMap;

/**
 * Each entity mention is described by a type and a span of text
 * 
 * @author Aju
 */
public class EntityMention extends ArgumentMention {
  private static final long serialVersionUID = -2745903102654191527L;
  private List<ArgumentRelation> relations;
  
  public EntityMention(String objectId, CoreMap sentence, Span span) {
   super(objectId, sentence, span);
   relations = new ArrayList<ArgumentRelation>();
  }
  
  public void addRelation(EntityMention mention, RelationType type) {
	  relations.add(new ArgumentRelation(mention, type));
  }
  
  public String prettyPrint() {
	  StringBuilder strBld = new StringBuilder();
	  strBld.append(String.format("%-25s", this.getValue()));
	  if(this.relations.size() > 0) {
		  strBld.append("\tSame entities:");
		  for(ArgumentRelation rel:relations)
			  strBld.append(String.format("  '%s'",rel.mention.getValue()));
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

