package edu.stanford.nlp.bioprocess.joint.core;

import com.google.common.base.Splitter;
import com.google.common.collect.Lists;

import fig.basic.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * Params contains the parameters of the model. Currently consists of a map from
 * features to weights.
 *
 * @author Percy Liang and Jonathan Berant
 */
public class Params {
  public static class Options {
    @Option(gloss = "By default, all features have this weight")
    public double defaultWeight = 0;
    @Option(gloss = "Randomly initialize the weights")
    public boolean initWeightsRandomly = false;
    @Option(gloss = "Randomly initialize the weights")
    public Random initRandom = new Random(1);

    @Option(gloss = "Initial step size") public double initStepSize = 1;
    @Option(gloss = "How fast to reduce the step size")
    public double stepSizeReduction = 0;
    @Option(gloss = "Use the AdaGrad algorithm (different step size for each coordinate)")
    public boolean adaptiveStepSize = true;
    @Option(gloss = "Use dual averaging") public boolean dualAveraging = false;
    @Option(gloss = "Whether to do lazy l1 reg updates") public String l1Reg="none";
    @Option(gloss = "L1 reg coefficient") public double l1RegCoeff = 1e-3;
  }
  public static Options opts = new Options();
  public enum L1Reg {
    LAZY,
    NONLAZY,
    NONE;
  }
  private L1Reg parseReg(String l1Reg) {
    if("lazy".equals(l1Reg)) return L1Reg.LAZY;
    if("nonlazy".equals(l1Reg)) return L1Reg.NONLAZY;
    if("none".equals(l1Reg)) return L1Reg.NONE;
    throw new RuntimeException("not legal l1reg");
  }
  private L1Reg l1Reg = parseReg(opts.l1Reg);

  // Discriminative weights
  private HashMap<String, Double> weights = new HashMap<String, Double>();

  // For AdaGrad
  Map<String, Double> sumSquaredGradients = new HashMap<String, Double>();

  // For dual averaging
  Map<String, Double> sumGradients = new HashMap<String, Double>();

  // Number of stochastic updates we've made so far (for determining step size).
  int numUpdates;

  //for lazy l1-reg update
  Map<String,Integer> l1UpdateTimeMap = new HashMap<String, Integer>();



  // Read parameters from |path|.
  public void read(String path) {
    LogInfo.begin_track("Reading parameters from %s", path);
    try {
      BufferedReader in = IOUtils.openIn(path);
      String line;
      while ((line = in.readLine()) != null) {
        String[] pair = Lists.newArrayList(Splitter.on('\t').split(line)).toArray(new String[2]);
        weights.put(pair[0], Double.parseDouble(pair[1]));
      }
      in.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    LogInfo.logs("Read %s weights", weights.size());
    LogInfo.end_track();
  }

  // Update weights by adding |gradient| (modified appropriately with step size).
  public void update(Map<String, Double> gradient) {
    numUpdates++;

    for (Map.Entry<String, Double> entry : gradient.entrySet()) {
      String f = entry.getKey();
      double g = entry.getValue();
      if (g == 0) continue;

      double stepSize = computeStepSize(f, g);

      if (opts.dualAveraging) {
        if (!opts.adaptiveStepSize && opts.stepSizeReduction != 0)
          throw new RuntimeException("Dual averaging not supported when " +
              "step-size changes across iterations for " +
              "features for which the gradient is zero");
        MapUtils.incr(sumGradients, f, g);
        MapUtils.set(weights, f, stepSize * sumGradients.get(f));
      }
      else {
        MapUtils.incr(weights, f, stepSize * g);
      }
    }
    //non lazy implementation goes over all weights 
    if(l1Reg==L1Reg.NONLAZY) {
      Set<String> features = new HashSet<String>(weights.keySet());
      for (String f :features) {
        double stepSize = computeStepSize(f, 0d); //no update for gradient here
        double update = opts.l1RegCoeff* -Math.signum(MapUtils.getDouble(weights, f, opts.defaultWeight));
        clipUpdate(f, stepSize*update);
      }
    }
  }

  private double computeStepSize(String feature, double gradient) {
    if (opts.adaptiveStepSize) {
      MapUtils.incr(sumSquaredGradients, feature, gradient * gradient);
      //ugly - adding one to the denominator when using l1 reg.
      if(l1Reg!=L1Reg.NONE)
        return opts.initStepSize / (Math.sqrt(sumSquaredGradients.get(feature)+1));
      else
        return opts.initStepSize / Math.sqrt(sumSquaredGradients.get(feature));
    } else {
      return opts.initStepSize / Math.pow(numUpdates, opts.stepSizeReduction);
    }
  }

  /*
   * If the update changes the sign, remove the feature
   */
  private void clipUpdate(String f, double update) {
    double currWeight = MapUtils.getDouble(weights, f, 0);
    if(currWeight == 0)
      return;

    if(currWeight*(currWeight+update)<0.0)  {
      weights.remove(f);
    }
    else {
      MapUtils.incr(weights, f, update);
    }
  }

  private void lazyL1Update(String f) {
    if(sumSquaredGradients.get(f)==null) {
      l1UpdateTimeMap.put(f, numUpdates);
      return;
    }
    int numOfIter = numUpdates-l1UpdateTimeMap.get(f);
    if(numOfIter<=0) {
      l1UpdateTimeMap.put(f, numUpdates);
      return;
    }
    
    double stepSize = (numOfIter * opts.initStepSize) / (Math.sqrt(sumSquaredGradients.get(f)+1));
    double update = -opts.l1RegCoeff * Math.signum(MapUtils.getDouble(weights, f, 0.0));
    clipUpdate(f, stepSize*update);
    l1UpdateTimeMap.put(f, numUpdates);
  }

  public double getWeight(String f) {
    if(l1Reg==L1Reg.LAZY)
      lazyL1Update(f);
    if (opts.initWeightsRandomly)
      return MapUtils.getDouble(weights, f, 2 * opts.initRandom.nextDouble() - 1);
    else {
      return MapUtils.getDouble(weights, f, opts.defaultWeight);
    }
  }

  public Map<String, Double> getWeights() { return weights; }

  public void write(PrintWriter out) { write(null, out); }

  public void write(String prefix, PrintWriter out) {
    List<Map.Entry<String, Double>> entries = Lists.newArrayList(weights.entrySet());
    Collections.sort(entries, new ValueComparator<String, Double>(true));
    for (Map.Entry<String, Double> entry : entries) {
      double value = entry.getValue();
      out.println((prefix == null ? "" : prefix + "\t") + entry.getKey() + "\t" + value);
    }
  }

  public void write(String path) {
    LogInfo.begin_track("Params.write(%s)", path);
    PrintWriter out = IOUtils.openOutHard(path);
    write(out);
    out.close();
    LogInfo.end_track();
  }

  public void log() {
    LogInfo.begin_track("Params");
    List<Map.Entry<String, Double>> entries = Lists.newArrayList(weights.entrySet());
    Collections.sort(entries, new ValueComparator<String, Double>(true));
    for (Map.Entry<String, Double> entry : entries) {
      double value = entry.getValue();
      LogInfo.logs("%s\t%s", entry.getKey(), value);
    }
    LogInfo.end_track();
  }

  public void finalizeWeights() {
    if(l1Reg==L1Reg.LAZY) {
      Set<String> features = new HashSet<String>(weights.keySet());
      for(String f: features) 
        lazyL1Update(f);
    }
  }
}
