package edu.stanford.nlp.bioprocess.joint.core;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.stanford.nlp.util.IntPair;

/**
 * 
 * @author svivek
 */
public class Structure implements IStructure {
	public final Input input;
	private final String[] relations;
	private final String[][] arguments;
	private final String[] triggers;

	public Structure(Input input, String[] triggers, String[][] arguments,
			String[] relations) {
		this.input = input;
		this.triggers = triggers;
		this.arguments = arguments;
		this.relations = relations;

		validate();
	}

	private void validate() {
		assert triggers.length == input.getNumberOfTriggers();

		assert arguments.length == input.getNumberOfTriggers();
		for (int triggerId = 0; triggerId < arguments.length; triggerId++) {
			assert arguments[triggerId].length == input
					.getNumberOfArgumentCandidates(triggerId);
		}

		assert relations.length == input.getNumberOfEERelationCandidates();
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return super.toString();
	}

	@Override
	public FeatureVector getFeatureVector() {
		// XXX: This function is not needed for this project
		return null;
	}

	public String getTriggerLabel(int triggerId) {
		assert input.isValidTriggerId(triggerId);
		return triggers[triggerId];
	}

	public String getArgumentCandidateLabel(int triggerId, int argCandidateId) {
		assert input.isValidTriggerId(triggerId);
		assert argCandidateId >= 0 && argCandidateId < arguments[triggerId].length;
		return arguments[triggerId][argCandidateId];
	}

	public String getArgumentCandidateLabel(int triggerId,
			IntPair argCandidateSpan) {
		return getArgumentCandidateLabel(triggerId,
				input.getArgumentSpanIndex(triggerId, argCandidateSpan));
	}

	public String getEERelationLabel(int trigger1, int trigger2) {
		assert input.isValidTriggerId(trigger1);
		assert input.isValidTriggerId(trigger2);

		return relations[input.getEERelationIndex(trigger1, trigger2)];
	}

}
