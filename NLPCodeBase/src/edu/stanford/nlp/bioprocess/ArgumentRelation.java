package edu.stanford.nlp.bioprocess;

import java.io.Serializable;

public class ArgumentRelation implements Serializable {
  
  public enum RelationType {
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
    Time
  }
  ArgumentMention mention;
  RelationType type;
  
  public ArgumentRelation(ArgumentMention mention, RelationType type) {
    this.mention = mention;
    this.type = type;
  }
}
