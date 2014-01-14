package edu.stanford.nlp.bioprocess.joint.core;

import java.io.Serializable;
import java.util.Arrays;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.IntPair;

import edu.stanford.nlp.bioprocess.joint.reader.DatasetUtils;

/**
 * 
 * @author svivek
 */
public class Structure implements IStructure, Serializable {

    private static final long serialVersionUID = -5015837729312326526L;
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

        // some sort of a to string. This will be very verbose, though
        StringBuffer sb = new StringBuffer();

        for (int triggerId = 0; triggerId < input.getNumberOfTriggers(); triggerId++) {
            // print only the triggers that are labeled "E"
            if(!triggers[triggerId].equals(DatasetUtils.OTHER_LABEL)) {
                CoreLabel triggerCoreLabel = input.getTriggerCoreLabel(triggerId);
                sb.append("Event trigger: " + triggerCoreLabel.originalText()
                          + " (token index " + triggerCoreLabel.index() + ")\t"
                          + triggers[triggerId] + "\n");

                //sb.append("\tArguments: \n");
                for (int argId = 0; argId < input
                         .getNumberOfArgumentCandidates(triggerId); argId++) {

                    // print only the arguments not labeled "NONE"
                    if(!arguments[triggerId][argId].equals(DatasetUtils.NONE_LABEL)) {
                        sb.append("\t" + arguments[triggerId][argId] + "\t");
                        for (CoreLabel c : input.getArgumentCandidate(triggerId, argId))
                            sb.append(c.originalText() + " ");
                        sb.append("\n");
                    }

                }
                //sb.append("\n");
            }
        }

        sb.append("\nEvent-event relations:\n");
        for (int rId = 0; rId < input.getNumberOfEERelationCandidates(); rId++) {

            IntPair ee = input.getEERelationCandidatePair(rId);

            // print only the relations that are not labeled "NONE"
            if(!relations[rId].equals(DatasetUtils.NONE_LABEL)) {
                sb.append("\t" + relations[rId] + "\t");

                CoreLabel source = input.getTriggerCoreLabel(ee.getSource());
                CoreLabel target = input.getTriggerCoreLabel(ee.getTarget());

                sb.append("\t" + source.originalText() + "\t" + target.originalText()
                          + "\n");
            }
        }

        return sb.toString();
    }

    @Override
    public edu.illinois.cs.cogcomp.indsup.learning.FeatureVector getFeatureVector() {
        // XXX: This function is not needed for this project
        return null;
    }

    public FeatureVector getFeatures() {
        // TODO Go over all assignments and compute feature vector
        return FeatureExtractor.getFeatures(this);
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

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(arguments);
        result = prime * result + ((input == null) ? 0 : input.hashCode());
        result = prime * result + Arrays.hashCode(relations);
        result = prime * result + Arrays.hashCode(triggers);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Structure other = (Structure) obj;
        if (!Arrays.deepEquals(arguments, other.arguments))
            return false;
        if (input == null) {
            if (other.input != null)
                return false;
        } else if (!input.equals(other.input))
            return false;
        if (!Arrays.equals(relations, other.relations))
            return false;
        if (!Arrays.equals(triggers, other.triggers))
            return false;
        return true;
    }

}
