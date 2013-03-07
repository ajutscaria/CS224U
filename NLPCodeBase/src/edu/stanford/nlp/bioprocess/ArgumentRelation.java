package edu.stanford.nlp.bioprocess;

import java.io.Serializable;

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
    Theme,
    SameEntity,
    StaticEvent,
    CotemporalEvent,
    NextEvent,
    SameEvent,
    SuperEvent,
    Enables,
    Time,
    NONE
  }
  ArgumentMention mention;
  RelationType type;
  
  public ArgumentRelation(ArgumentMention mention, RelationType type) {
    this.mention = mention;
    this.type = type;
  }
}
