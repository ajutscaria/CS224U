package edu.stanford.nlp.bioprocess.joint.core;

import edu.stanford.nlp.bioprocess.joint.reader.DatasetUtils;
import fig.basic.LogInfo;

public class Evaluation {

  private int numOfFolds;
  private int[] tpRel, fpRel, fnRel;
  private int[] tpEnt, fpEnt, fnEnt;
  private int[] tpEvt, fpEvt, fnEvt;
  private double[] precisionRel, recallRel, f1Rel;
  private double[] precisionEnt, recallEnt, f1Ent;
  private double[] precisionEvt, recallEvt, f1Evt;

  private class Stat{
    double p;
    double r;
    double f1;
    int tp;
    int fp;
    int fn;
    public Stat(double p, double r, double f1){
      this.f1 = f1;
      this.p = p;
      this.r = r;
    }
    
    public Stat(int tp, int fp, int fn){
      this.tp = tp;
      this.fp = fp;
      this.fn = fn;
    }
  }
  
  public Evaluation(int numOfFold) {
    this.numOfFolds = numOfFold;
    tpRel = new int[numOfFold];
    fpRel = new int[numOfFold];
    fnRel = new int[numOfFold];
    tpEnt = new int[numOfFold];
    fpEnt = new int[numOfFold];
    fnEnt = new int[numOfFold];
    tpEvt = new int[numOfFold];
    fpEvt = new int[numOfFold];
    fnEvt = new int[numOfFold];
    precisionRel = new double[numOfFold];
    recallRel = new double[numOfFold];
    f1Rel = new double[numOfFold];
    precisionEnt = new double[numOfFold];
    recallEnt = new double[numOfFold];
    f1Ent = new double[numOfFold];
    precisionEvt = new double[numOfFold];
    recallEvt = new double[numOfFold];
    f1Evt = new double[numOfFold];
  }

  public void score(Structure gold, Structure predicted, int fold) {
    scoreEvents(gold, predicted, fold);
    scoreArguments(gold, predicted, fold);
    scoreEERelations(gold, predicted, fold);
  }
  
  public Stat getEventMeasure(int fold){
    Stat measures = new Stat(tpEvt[fold], fpEvt[fold], fnEvt[fold]);
    return measures;
  }
  
  public Stat getEntityMeasure(int fold){
    Stat measures = new Stat(tpEnt[fold], fpEnt[fold], fnEnt[fold]);
    return measures;
  }
  
  public Stat getEERelationMeasure(int fold){
    Stat measures = new Stat(tpRel[fold], fpRel[fold], fnRel[fold]);
    return measures;
  }

  private void scoreEvents(Structure gold, Structure predicted, int fold) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    assert gold.input.getNumberOfTriggers() == predicted.input.getNumberOfTriggers();
    for (int eventId = 0; eventId < gold.input.getNumberOfTriggers(); eventId++) {
      //LogInfo.logs("Event "+eventId+", gold:"+gold.getTriggerLabel(eventId)+", predicted:"+predicted.getTriggerLabel(eventId));
      if(gold.getTriggerLabel(eventId).equals(predicted.getTriggerLabel(eventId))){
        if(gold.getTriggerLabel(eventId).equals(DatasetUtils.EVENT_LABEL))
          tp++;
        else
          tn++;
      }else if(gold.getTriggerLabel(eventId).equals(DatasetUtils.EVENT_LABEL) && 
          predicted.getTriggerLabel(eventId).equals(DatasetUtils.OTHER_LABEL)){
        fn++;    
      }else if(gold.getTriggerLabel(eventId).equals(DatasetUtils.OTHER_LABEL) && 
          predicted.getTriggerLabel(eventId).equals(DatasetUtils.EVENT_LABEL)){
        fp++;
      }
        
    }
    //LogInfo.logs("tp:"+tp+", fp:"+fp+", fn:"+fn+", tn:"+tn);
    
    tpEvt[fold] += tp;
    fpEvt[fold] += fp;
    fnEvt[fold] += fn;
    Stat measures = calcStats(tp, fp, fn);
    Stat measuresCul = calcStats(tpEvt[fold], fpEvt[fold], fnEvt[fold]);
    LogInfo.logs("Current Event (P/R/F1):"+ String.format("%.3f ", measures.p)+","+String.format("%.3f ", measures.r)+
        ","+String.format("%.3f ", measures.f1));
    LogInfo.logs("Cumulative Event (P/R/F1):"+ String.format("%.3f ", measuresCul.p)+","
        +String.format("%.3f ", measuresCul.r)+","+String.format("%.3f ", measuresCul.f1));

  }

  private void scoreArguments(Structure gold, Structure predicted, int fold) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    assert gold.input.getNumberOfTriggers() == predicted.input.getNumberOfTriggers();
    for (int eventId = 0; eventId < gold.input.getNumberOfTriggers(); eventId++) {
      //LogInfo.begin_track("For event "+eventId);
      assert gold.input.getNumberOfArgumentCandidates(eventId) == predicted.input.getNumberOfArgumentCandidates(eventId);
      for (int entityId = 0; entityId < gold.input
          .getNumberOfArgumentCandidates(eventId); entityId++) {
        //LogInfo.logs("Entity "+entityId+", gold:"+gold.getArgumentCandidateLabel(eventId, entityId)+", predicted:"+predicted.getArgumentCandidateLabel(eventId, entityId));
        if(gold.getArgumentCandidateLabel(eventId, entityId).equals(predicted.getArgumentCandidateLabel(eventId, entityId))){
          if(!gold.getArgumentCandidateLabel(eventId, entityId).equals(DatasetUtils.NONE_LABEL))
            tp++;
          else
            tn++;
        }else if(!gold.getArgumentCandidateLabel(eventId, entityId).equals(DatasetUtils.NONE_LABEL) && 
            predicted.getArgumentCandidateLabel(eventId, entityId).equals(DatasetUtils.NONE_LABEL)){
          fn++;    
        }else if(gold.getArgumentCandidateLabel(eventId, entityId).equals(DatasetUtils.NONE_LABEL) && 
            !predicted.getArgumentCandidateLabel(eventId, entityId).equals(DatasetUtils.NONE_LABEL)){
          fp++;
        }
      }
      //LogInfo.end_track();
    }
    //LogInfo.logs("tp:"+tp+", fp:"+fp+", fn:"+fn+", tn:"+tn);
    tpEnt[fold] += tp;
    fpEnt[fold] += fp;
    fnEnt[fold] += fn;
    Stat measures = calcStats(tp, fp, fn);
    Stat measuresCul = calcStats(tpEnt[fold], fpEnt[fold], fnEnt[fold]);
    LogInfo.logs("Current Entity (P/R/F1):"+ String.format("%.3f ", measures.p)+","+String.format("%.3f ", measures.r)+
        ","+String.format("%.3f ", measures.f1));
    LogInfo.logs("Cumulative Entity (P/R/F1):"+ String.format("%.3f ", measuresCul.p)+","
        +String.format("%.3f ", measuresCul.r)+","+String.format("%.3f ", measuresCul.f1));
  }

  private void scoreEERelations(Structure gold, Structure predicted, int fold) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    assert gold.input.getNumberOfEERelationCandidates() == predicted.input.getNumberOfEERelationCandidates();
    for (int Id = 0; Id < gold.input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = gold.input.getEERelationCandidatePair(Id).getSource(); 
      int event2 = gold.input.getEERelationCandidatePair(Id).getTarget(); 
      assert gold.input.getEERelationCandidatePair(Id).getSource() == predicted.input.getEERelationCandidatePair(Id).getSource(); 
      assert gold.input.getEERelationCandidatePair(Id).getTarget() == predicted.input.getEERelationCandidatePair(Id).getTarget();
      //LogInfo.logs("Event-event ("+event1+","+event2+")"+", gold:"+gold.getEERelationLabel(event1, event2)+", predicted:"+predicted.getEERelationLabel(event1, event2));
      if(gold.getEERelationLabel(event1, event2).equals(predicted.getEERelationLabel(event1, event2))){
        if(!gold.getEERelationLabel(event1, event2).equals(DatasetUtils.NONE_LABEL))
          tp++;
        else
          tn++;
      }else if(!gold.getEERelationLabel(event1, event2).equals(DatasetUtils.NONE_LABEL) && 
          predicted.getEERelationLabel(event1, event2).equals(DatasetUtils.NONE_LABEL)){
        fn++;    
      }else if(gold.getEERelationLabel(event1, event2).equals(DatasetUtils.NONE_LABEL) && 
          !predicted.getEERelationLabel(event1, event2).equals(DatasetUtils.NONE_LABEL)){
        fp++;
      }
    }
    //LogInfo.logs("tp:"+tp+", fp:"+fp+", fn:"+fn+", tn:"+tn);
    tpRel[fold] += tp;
    fpRel[fold] += fp;
    fnRel[fold] += fn;
    Stat measures = calcStats(tp, fp, fn);
    Stat measuresCul = calcStats(tpRel[fold], fpRel[fold], fnRel[fold]);
    LogInfo.logs("Current EE-Relation (P/R/F1):"+ String.format("%.3f ", measures.p)+","+String.format("%.3f ", measures.r)+
        ","+String.format("%.3f ", measures.f1));
    LogInfo.logs("Cumulative EE-Relation (P/R/F1):"+ String.format("%.3f ", measuresCul.p)+","
        +String.format("%.3f ", measuresCul.r)+","+String.format("%.3f ", measuresCul.f1));
  }
  
  private Stat calcStats(int tp, int fp, int fn){
    double p = (double)tp/(tp+fp);
    double r = (double)tp/(tp+fn);
    double f1 = 2*p*r/(p+r);
    return new Stat(p, r, f1);
  }

  public void calcScore() {
    for (int i = 0; i < numOfFolds; i++) {

      precisionEvt[i] = (double) tpEvt[i] / (tpEvt[i] + fpEvt[i]);
      recallEvt[i] = (double) tpEvt[i] / (tpEvt[i] + fnEvt[i]);
      f1Evt[i] = 2 * precisionEvt[i] * recallEvt[i]
          / (precisionEvt[i] + recallEvt[i]);

      precisionEnt[i] = (double) tpEnt[i] / (tpEnt[i] + fpEnt[i]);
      recallEnt[i] = (double) tpEnt[i] / (tpEnt[i] + fnEnt[i]);
      f1Ent[i] = 2 * precisionEnt[i] * recallEnt[i]
          / (precisionEnt[i] + recallEnt[i]);

      precisionRel[i] = (double) tpRel[i] / (tpRel[i] + fpRel[i]);
      recallRel[i] = (double) tpRel[i] / (tpRel[i] + fnRel[i]);
      f1Rel[i] = 2 * precisionRel[i] * recallRel[i]
          / (precisionRel[i] + recallRel[i]);

    }

    printScores("Event", precisionEvt, recallEvt,
        f1Evt);
    printScores("Entity", precisionEnt, recallEnt,
        f1Ent);
    printScores("Relation", precisionRel, recallRel,
        f1Rel);
  }
  
  public void setEventMeasure(int fold, Stat obj){
    tpEvt[fold] = obj.tp;
    fpEvt[fold] = obj.fp;
    fnEvt[fold] = obj.fn;
  }
  
  public void setEntityMeasure(int fold, Stat obj){
    tpEnt[fold] = obj.tp;
    fpEnt[fold] = obj.fp;
    fnEnt[fold] = obj.fn;
  }
  
  public void setEERelationMeasure(int fold, Stat obj){
    tpRel[fold] = obj.tp;
    fpRel[fold] = obj.fp;
    fnRel[fold] = obj.fn;
  }

  private void printScores(String category, double[] precision, double[] recall,
      double[] f1) {
    LogInfo.logs(String.format("\n------------------------------------------"
        + category + "-------------------------------------------"));
    LogInfo.logs(printScore("Precision", precision));
    LogInfo.logs(printScore("Recall   ", recall));
    LogInfo.logs(printScore("F1       ", f1));
  }

  private String printScore(String scoreType, double[] scores) {
    StringBuilder ret = new StringBuilder(String.format("%s ", scoreType));
    for (double s : scores)
      ret.append(String.format("%.3f ", s));
    ret.append(String.format("%.3f", getAverage(scores)));
    return ret.toString();
  }

  private double getAverage(double[] scores) {
    double sum = 0;
    for (double d : scores)
      sum += d;
    return sum / scores.length;
  }

}
