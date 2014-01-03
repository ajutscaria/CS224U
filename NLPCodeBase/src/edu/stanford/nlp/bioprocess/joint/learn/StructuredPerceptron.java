package edu.stanford.nlp.bioprocess.joint.learn;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory.SolverType;
import edu.stanford.nlp.bioprocess.joint.core.FeatureVector;
import edu.stanford.nlp.bioprocess.joint.core.Input;
import edu.stanford.nlp.bioprocess.joint.core.Params;
import edu.stanford.nlp.bioprocess.joint.core.Structure;
import edu.stanford.nlp.bioprocess.joint.inference.Inference;
import edu.stanford.nlp.bioprocess.joint.reader.DatasetUtils;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;
import fig.basic.Option;

public class StructuredPerceptron {

	public static class Options {
		@Option(gloss = "Number of epochs. Default = 20")
		public int numEpochs = 20;

		@Option(gloss = "Number of examples for status report. Default = 1000")
		public int reportCount = 1000;
		//
		// @Option(gloss = "Initialize randomly? Default = false")
		// public boolean initializeRandom = false;

		// @Option(gloss =
		// "Use loss augmented inference during training? Default = false")
		// public boolean useLossSensitiveInference = false;

	}

	public static Options opts = new Options();
	private final Random random;

	private final ILPSolverFactory solverFactory = new ILPSolverFactory(
			SolverType.CuttingPlaneGurobi);

	public StructuredPerceptron(Random random) {
		this.random = random;
	}

	public Params learn(List<Pair<Input, Structure>> data) throws Exception {
	    LogInfo.begin_track("Perceptron");
		Params w = new Params();
		Params a = new Params();

		// n is a running counter that keeps showing up in many places. We could
		// compute n on the fly, but code-wise it is easier to just remember it.
		int n = 1;
		for (int epoch = 0; epoch < opts.numEpochs; epoch++) {
			n = runEpoch(epoch, data, w, a, n);

		}

		Params avg = average(w, a, n);
        LogInfo.end_track();

		return avg;
	}

	private int runEpoch(int epoch, List<Pair<Input, Structure>> data, Params w,
			Params a, int n) throws Exception {
		
		LogInfo.begin_track("Epoch " + epoch);
		LogInfo.logs("Shuffling examples");
		DatasetUtils.shuffle(data, random);

		int numUpdates = 0;
		int count = 0;

		long inferenceTime = 0;

		for (Pair<Input, Structure> example : data) {
			Structure gold = example.second();

			long start = System.currentTimeMillis();
			Inference inference = new Inference(example.first(), w, solverFactory,
					false);
			long end = System.currentTimeMillis();

			inferenceTime += (end - start);

			Structure prediction = inference.runInference();

			if (!gold.equals(prediction)) {
				// update
				FeatureVector update = getUpdate(gold, prediction);

				// hopefully this takes care of the learning rate
				w.update(update.toMap());

				FeatureVector aUpdate = new FeatureVector();
				aUpdate.increment(n * 1.0, update.toMap());
				a.update(aUpdate.toMap());

				numUpdates++;
			}

			count++;
			n++;
			if (count % opts.reportCount == 0) {
				
				LogInfo.logs(count + " examples seen in epoch, number of updates = "
                        + numUpdates);
				
				LogInfo.logs("Total time spent on inference in this epoch: "
                    + inferenceTime);
			}
		}

		LogInfo.logs("End of epoch summary: " + count
				+ " examples seen in epoch, number of updates = " + numUpdates);
		LogInfo.logs("Total time spent on inference in this epoch: " + inferenceTime);

		LogInfo.end_track();

		return n;
	}

	/**
	 * Calcuate gold features minus predicted features
	 * 
	 * @param gold
	 * @param prediction
	 * @return
	 */
	private FeatureVector getUpdate(Structure gold, Structure prediction) {
		FeatureVector goldFeatures = gold.getFeatures();
		FeatureVector predictedFeatures = prediction.getFeatures();

		FeatureVector update = new FeatureVector();
		update.increment(1.0, goldFeatures.toMap());
		update.increment(-1.0, predictedFeatures.toMap());
		return update;
	}

	/**
	 * Calculate w - a / n
	 * <p>
	 * Shameful code ahead
	 * 
	 * @param w
	 * @param a
	 * @param n
	 * @return
	 */
	private Params average(Params w, Params a, int n) {
		LogInfo.logs("Averaging parameters");

		Params p = new Params();

		// Hmmm... this doesn't look too efficient!
		Map<String, Double> weights = w.getWeights();
		Map<String, Double> aWeights = a.getWeights();

		Map<String, Double> pWeights = p.getWeights();

		for (Entry<String, Double> entry : weights.entrySet()) {
			String name = entry.getKey();
			double value = entry.getValue();

			if (aWeights.containsKey(name))
				value = value - aWeights.get(name) / n;

			pWeights.put(name, value);
		}

		// let's get any weights that may have been left behind
		for (Entry<String, Double> entry : aWeights.entrySet()) {
			String name = entry.getKey();
			double value = entry.getValue();

			if (!weights.containsKey(name))
				pWeights.put(name, -value / n);
		}

		return p;
	}
}
