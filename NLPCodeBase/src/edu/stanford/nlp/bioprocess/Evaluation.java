package edu.stanford.nlp.bioprocess;

import edu.stanford.nlp.util.Triple;

public class Evaluation {
	int fold;
	int[] tpRel, fpRel, fnRel;
	int[] tpEnt, fpEnt, fnEnt;
	int[] tpEvt, fpEvt, fnEvt;
	double[] precisionRel, recallRel, f1Rel;
	double[] precisionEnt, recallEnt, f1Ent;
	double[] precisionEvt, recallEvt, f1Evt;
	
	public Evaluation(int fold){
    	this.fold = fold;
    	tpRel = new int[fold]; fpRel = new int[fold]; fnRel = new int[fold];
    	tpEnt = new int[fold]; fpEnt = new int[fold]; fnEnt = new int[fold];
    	tpEvt = new int[fold]; fpEvt = new int[fold]; fnEvt = new int[fold];
    	precisionRel = new double[fold]; recallRel = new double[fold]; f1Rel = new double[fold];
    	precisionEnt = new double[fold]; recallEnt = new double[fold]; f1Ent = new double[fold];
    	precisionEvt = new double[fold]; recallEvt = new double[fold]; f1Evt = new double[fold];
    }
    
	public void updateStat(int i, String type, Example.Stat stats){
		if(type.equals("event")){
			tpEvt[i-1] += stats.tp; //tp
			fpEvt[i-1] += stats.fp; //fp
			fnEvt[i-1] += stats.fn; //fn
		}else if(type.equals("entity")){
			tpEnt[i-1] += stats.tp; //tp
			fpEnt[i-1] += stats.fp; //fp
			fnEnt[i-1] += stats.fn; //fn
		}else if(type.equals("relation")){
			tpRel[i-1] += stats.tp; //tp
			fpRel[i-1] += stats.fp; //fp
			fnRel[i-1] += stats.fn; //fn
		}
	}
	
	public void calcStat(int i){
		precisionEvt[i-1] = (double)tpEvt[i-1]/(tpEvt[i-1]+fpEvt[i-1]); 
		recallEvt[i-1] = (double)tpEvt[i-1]/(tpEvt[i-1]+fnEvt[i-1]);
		f1Evt[i-1] = 2 * precisionEvt[i-1] * recallEvt[i-1] / (precisionEvt[i-1] + recallEvt[i-1]);
		precisionEnt[i-1] = (double)tpEnt[i-1]/(tpEnt[i-1]+fpEnt[i-1]); 
		recallEnt[i-1] = (double)tpEnt[i-1]/(tpEnt[i-1]+fnEnt[i-1]);
		f1Ent[i-1] = 2 * precisionEnt[i-1] * recallEnt[i-1] / (precisionEnt[i-1] + recallEnt[i-1]);
		precisionRel[i-1] = (double)tpRel[i-1]/(tpRel[i-1]+fpRel[i-1]); 
		recallRel[i-1] = (double)tpRel[i-1]/(tpRel[i-1]+fnRel[i-1]);
		f1Rel[i-1] = 2 * precisionRel[i-1] * recallRel[i-1] / (precisionRel[i-1] + recallRel[i-1]);
	}
}
