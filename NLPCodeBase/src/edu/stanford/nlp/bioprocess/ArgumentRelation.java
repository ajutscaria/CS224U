package edu.stanford.nlp.bioprocess;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class ArgumentRelation implements Serializable {
  /**
	 * 
	 */
	private static final long serialVersionUID = 7366414613762160263L;

public enum EventType {
	  Event,
	  StaticEvent,
	  NONE
  }
  public enum RelationType {
	Origin,
    Agent,
    Location,
    Destination,
    Result,
    RawMaterial,
    Theme,
    Time,
    SameEntity,
    StaticEvent,
    CotemporalEvent,
    NextEvent,
    PreviousEvent,
    SameEvent,
    SuperEvent,
    SubEvent,
    Causes,
    Caused,
    Enables,
    Enabled,
    NONE
  }
  ArgumentMention mention;
  RelationType type;
  
  public ArgumentRelation(ArgumentMention mention, RelationType type) {
    this.mention = mention;
    this.type = type;
  }
  
  public static List<String> getSemanticRoles() {
	  List<String> srlList = new ArrayList<String>();
	  
	  srlList.add(RelationType.NONE.toString());
	  srlList.add(RelationType.Agent.toString());
	  srlList.add(RelationType.Result.toString());
	  srlList.add(RelationType.Origin.toString());
	  srlList.add(RelationType.Location.toString());
	  srlList.add(RelationType.Destination.toString());
	  srlList.add(RelationType.RawMaterial.toString());
	  srlList.add(RelationType.Theme.toString());
	  srlList.add(RelationType.Time.toString());

	  return srlList;  
  }
  
  public static List<String> getEventRelations() {
	  List<String> srlList = new ArrayList<String>();
	  
	  srlList.add(RelationType.NONE.toString());
	  srlList.add(RelationType.CotemporalEvent.toString());
	  srlList.add(RelationType.NextEvent.toString());
	  srlList.add(RelationType.PreviousEvent.toString());
	  srlList.add(RelationType.SameEvent.toString());
	  srlList.add(RelationType.SuperEvent.toString());
	  srlList.add(RelationType.SubEvent.toString());
	  srlList.add(RelationType.Causes.toString());
	  srlList.add(RelationType.Caused.toString());
	  srlList.add(RelationType.Enables.toString());
	  srlList.add(RelationType.Enabled.toString());

	  return srlList;  
  }
}
