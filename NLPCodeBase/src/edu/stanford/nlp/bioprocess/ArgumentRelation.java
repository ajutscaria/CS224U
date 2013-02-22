package edu.stanford.nlp.bioprocess;

public class ArgumentRelation {
  
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
    Enables
  }
  ArgumentMention mention;
  RelationType type;
  
  public ArgumentRelation(ArgumentMention mention, RelationType type) {
    this.mention = mention;
    this.type = type;
  }
}
