package edu.stanford.nlp.bioprocess.joint.core;

/**
 * Used by feature vector to decide what features to include (good for various ablations)
 * @author jonathanberant
 *
 */
public interface FeatureMatcher {
  public boolean matches(String feature);
}

class AllFeatureMatcher implements FeatureMatcher {
  private AllFeatureMatcher() { }
  @Override
  public boolean matches(String feature) { return true; }
  public static final AllFeatureMatcher matcher = new AllFeatureMatcher();
}

class ExactFeatureMatcher implements FeatureMatcher {
  private String match;
  public ExactFeatureMatcher(String match) { this.match = match; }
  @Override
  public boolean matches(String feature) { return feature.equals(match); }
}
