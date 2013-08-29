package edu.stanford.nlp.bioprocess.wordcluster;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.util.logging.Redwood;
import fig.basic.LogInfo;

/**
 * Gets a text file in the proper format (one word per line, new line for new sentence) and outputs a clustering by 
 * running Alex Clark code
 * @author jonathanberant
 *
 */
public class ExchangeWordClustering {

	public final static int MIN_WORD_FREQ = 5;
	public final static int TRAIN_ITER_NUM = 20;
	public final static int CLUSTER_NUM = 200;
	public final static String BINARY_PATH = "lib/posinduction/bin/cluster_neyessenmorph";
	public final static String OUTPUT_PREFIX = "lib/data/word_cluster_clark";

	public static void main(String[] args) throws IOException {

		Redwood.startTrack("Clustering");
		String[] cmd = new String[8];
		cmd[0]=BINARY_PATH;
		cmd[1]="-m";
		cmd[2]=""+MIN_WORD_FREQ;
		cmd[3]="-i";
		cmd[4]=""+TRAIN_ITER_NUM;
		cmd[5]=args[0]+"/"+args[1];
		cmd[6]=args[0]+"/"+args[1];
		cmd[7]=""+CLUSTER_NUM;		
		String outFile = args[0]+"/"+OUTPUT_PREFIX+"_m_"+MIN_WORD_FREQ+"_i_"+TRAIN_ITER_NUM+"_cl_"+CLUSTER_NUM;

		PrintWriter writer = IOUtils.getPrintWriter(outFile);
		Process p = Runtime.getRuntime().exec(cmd);
		{
			BufferedReader in =
					new BufferedReader(new InputStreamReader(p.getInputStream()));
			String line;
			while ((line = in.readLine()) != null) {
				writer.println(line);

			}
		}

		{
			BufferedReader err =
					new BufferedReader(new InputStreamReader(p.getErrorStream()));
			String line;
			while ((line = err.readLine()) != null) {
				LogInfo.logs(line);
			}
		}
		writer.close();
		Redwood.endTrack("Clustering");

	}
}
