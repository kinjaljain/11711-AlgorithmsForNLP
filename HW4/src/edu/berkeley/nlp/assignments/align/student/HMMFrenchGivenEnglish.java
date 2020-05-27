package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.*;

public class HMMFrenchGivenEnglish {
    private double epsilon = 0.01;
    private double alpha = 0.4;
    private StringIndexer englishWordIndexer;
    private StringIndexer frenchWordIndexer;
    private HashMap<String, Double> tsIndexer;
    private double[] psi;

    public void train(Iterable<SentencePair> trainingData) {
        System.out.println("Training forward");
        englishWordIndexer = new StringIndexer();
        frenchWordIndexer = new StringIndexer();
        tsIndexer = new HashMap<>(100000);  //theta
        int num_em_iters = 7;

        int maxEnglishSentenceLength = initializeIndexers(trainingData, englishWordIndexer, frenchWordIndexer);
        int totalFrenchWords = frenchWordIndexer.size();
        int totalEnglishWords = englishWordIndexer.size();

        System.out.println("Max English Sentence Length: " + maxEnglishSentenceLength);
        System.out.println("English Words: " + totalEnglishWords);
        System.out.println("French Words: " + totalFrenchWords);

        double uniformTranslationProb = Math.log(1.0/totalEnglishWords); // check once english or french
        initializeTheta(trainingData, totalEnglishWords); // initialise theta

        psi = new double[maxEnglishSentenceLength]; // adding for null as well
        Arrays.fill(psi, Math.log(1.0/maxEnglishSentenceLength)); // initialise psi

        for(int i=0; i< num_em_iters; i++){
            System.out.println("EM Iteration: " + (i+1));
            HashMap<String, Double> newtsIndexer = new HashMap<>(tsIndexer.size());
//            System.out.println("the , le: ");
//            System.out.println(tsIndexer.get("le===the"));
            double[] newpsi = new double[psi.length];
            Arrays.fill(newpsi, Double.NEGATIVE_INFINITY);
            for (SentencePair sentencePair : trainingData) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                frenchWords.replaceAll(String::toLowerCase);
                englishWords.replaceAll(String::toLowerCase);
                englishWords.add("null");
                int numFrench = frenchWords.size();
                int numEnglish = englishWords.size();
                double[][] aTable = new double[numEnglish][numEnglish];
                double[] piTable = new double[numEnglish];

                double temp;
                // fill A table
                for(int j=0; j<numEnglish; j++){
                    aTable[j][numEnglish-1] = Math.log(epsilon);
                }
                for(int j=0; j<numEnglish-1; j++){
                    aTable[numEnglish-1][j] = Math.log((1 - epsilon)/(numEnglish-1));
                }
                for(int j=0; j<numEnglish-1; j++){
                    temp = Double.NEGATIVE_INFINITY;
                    for(int k=0; k<numEnglish-1; k++){
                        temp = SloppyMath.logAdd(temp, psi[Math.abs(k-j)]); // + Math.log(1 - epsilon)
                    }
                    for(int k=0; k<numEnglish-1; k++){
                        aTable[j][k] = psi[Math.abs(k-j)] + Math.log(1 - epsilon)- temp; ///(numEnglish-1))
                    }
                } // including numEnglishWords+1

                // do smoothing
                for(int j=0; j<numEnglish-1; j++){
                    for(int k=0; k<numEnglish-1; k++){
                        aTable[j][k] = SloppyMath.logAdd(Math.log(alpha) + Math.log(1.0/(numEnglish-1)),
                                aTable[j][k] + Math.log(1-alpha));
                    }
                }

                // fill pi table
                System.arraycopy(aTable[0], 0, piTable, 0, numEnglish);

                // create alpha table
                double[][] alphaTable = new double[numEnglish][numFrench];
                for(int j=0; j<numEnglish; j++) {
                    String key = frenchWords.get(0) + "===" + englishWords.get(j);
                    alphaTable[j][0] = tsIndexer.get(key) + piTable[j];
                }
                for(int j=1; j<numFrench; j++){
                    for(int k=0; k<numEnglish; k++) { // check to start from 0 or 1
                        String key = frenchWords.get(j) + "===" + englishWords.get(k);
                        temp = Double.NEGATIVE_INFINITY; // log 0
                        for(int l=0; l<numEnglish; l++){
                            temp = SloppyMath.logAdd(temp, aTable[l][k] + alphaTable[l][j-1]);
                        }
                        alphaTable[k][j] = tsIndexer.get(key) + temp;
                    }
                }

                // create beta table
                double[][] betaTable = new double[numEnglish][numFrench];
                for(int j=0; j<numEnglish; j++) {
                    betaTable[j][numFrench-1] = 0.0; // log 1 = 0
                }
                for(int j=numFrench-2; j>=0; j--){ // k, l= start from 1 not 0 earlier
                    for(int k=0; k<numEnglish; k++) {
                        temp = Double.NEGATIVE_INFINITY;
                        for(int l=0; l<numEnglish; l++){
                            String key = frenchWords.get(j+1) + "===" + englishWords.get(l);
                            temp = SloppyMath.logAdd(temp, tsIndexer.get(key) + aTable[k][l] + betaTable[l][j+1]);
                        }
                        betaTable[k][j] = temp;
                    }
                }
                // p' table
                double[][] changePTable = new double[numFrench][numEnglish];
                double[] z = new double[numFrench];
                Arrays.fill(z, Double.NEGATIVE_INFINITY);
                for(int j=1; j<numFrench; j++){
                    for(int k=0; k< numEnglish; k++){
                        temp = Double.NEGATIVE_INFINITY;
                        for(int l=0; l < numEnglish - k; l++){
                            String key1 = frenchWords.get(j) + "===" + englishWords.get(l+k);
                            String key2 = frenchWords.get(j) + "===" + englishWords.get(l);
                            temp = SloppyMath.logAdd(temp, SloppyMath.logAdd(
                                    alphaTable[l+k][j-1]+ betaTable[l][j] + aTable[l][l+k] + tsIndexer.get(key1),
                                    alphaTable[l][j-1] + betaTable[l+k][j] + aTable[l+k][l] + tsIndexer.get(key2)));
                        } // aTable[l][l+k] + aTable[l+k][l]
                        changePTable[j][k] = temp;
                        z[j] = SloppyMath.logAdd(z[j], temp);
                    }
                }
                // M step
                // get new psi
                for(int j=0; j<numEnglish; j++){ // jump max = numEnglish
                    for(int k=1; k<numFrench; k++){
                        try{
                            newpsi[j] = SloppyMath.logAdd(newpsi[j], (changePTable[k][j] - z[k]));}
                        catch (Exception e){
                            System.out.println(j);
                        }
                    }
                }
                changePTable = null;
                // get new theta
                String key;
                for(int j=0; j<numFrench; j++) {
                    double sum = Double.NEGATIVE_INFINITY;
                    for (int k = 0; k < numEnglish; k++) {
                        sum = SloppyMath.logAdd(sum, alphaTable[k][j] + betaTable[k][j]);
                    }
                    for (int k = 0; k < numEnglish; k++) {
                        key = frenchWords.get(j) + "===" + englishWords.get(k);
                        if (newtsIndexer.containsKey(key))
                            newtsIndexer.put(key, SloppyMath.logAdd(newtsIndexer.get(key),
                                    (alphaTable[k][j] + betaTable[k][j] - sum)));
                        else {
                            newtsIndexer.put(key, (alphaTable[k][j] + betaTable[k][j] - sum));
                        }
                    }
                }
                englishWords.remove("null");
            }
            // normalize new theta
            double[] englishVal = new double[totalEnglishWords];
            Arrays.fill(englishVal, Double.NEGATIVE_INFINITY);
            for(String key: newtsIndexer.keySet()){
                String englishWord = key.split("===")[1];
                englishVal[englishWordIndexer.addAndGetIndex(englishWord)] = SloppyMath.logAdd(
                        englishVal[englishWordIndexer.addAndGetIndex(englishWord)], newtsIndexer.get(key));
            }
            for(String key: newtsIndexer.keySet()){
                String englishWord = key.split("===")[1];
                newtsIndexer.put(key, newtsIndexer.get(key) - englishVal[englishWordIndexer.addAndGetIndex(englishWord)]);
            }
            tsIndexer = newtsIndexer;
             psi = newpsi;
            newtsIndexer = null;
            newpsi = null;
        }
    }

    private double getTheta(int totalEnglishWords, int iter, String key){
        if (iter==1){
            tsIndexer.put(key, Math.log(1.0/totalEnglishWords));
        }
        return tsIndexer.get(key);
    }

    private void initializeTheta(Iterable<SentencePair> trainingData, int totalEnglishWords){
        HashMap<String, Double> tsCounter = new HashMap<>(10000);
        double[] total_s = new double[totalEnglishWords];
        Arrays.fill(total_s, 0);
        for (SentencePair sentencePair : trainingData) {
            List<String> englishWords = sentencePair.getEnglishWords();
            List<String> frenchWords = sentencePair.getFrenchWords();
            frenchWords.replaceAll(String::toLowerCase);
            englishWords.replaceAll(String::toLowerCase);
            englishWords.add("null");
            double[] s_total_t = new double[frenchWords.size()];
            for(int j=0; j< frenchWords.size(); j++){
                s_total_t[j] = 0;
                s_total_t[j] += 1.0 * englishWords.size();
            }
            for(int j=0; j< frenchWords.size(); j++) {
                for (String englishWord : englishWords) {
                    String key = frenchWords.get(j) + "===" + englishWord;
                    if (tsCounter.containsKey(key)) {
                        double val = tsCounter.get(key);
                        tsCounter.put(key, val + (1.0 / s_total_t[j]));
                    }
                    else {
                        tsCounter.put(key, (1.0 / s_total_t[j]));
                    }
                    total_s[englishWordIndexer.addAndGetIndex(englishWord)] += 1.0 / s_total_t[j];
                }
            }
            englishWords.remove("null");
        }
        for(String key: tsCounter.keySet()){
            String englishWord = key.split("===")[1];
//             check log once
            tsIndexer.put(key, Math.log(tsCounter.get(key) / total_s[englishWordIndexer.addAndGetIndex(englishWord)]));
//            tsIndexer.put(key, Math.log(1.0/totalEnglishWords));
        }
    }

    private int initializeIndexers(Iterable<SentencePair> trainingData,
                                   StringIndexer englishWordIndexer, StringIndexer frenchWordIndexer){
        int numPairs = 0;
        int maxEnglishWords = 0;
        for (SentencePair sentencePair : trainingData) {
            numPairs++;
            if(numPairs % 1000 == 0){
                System.out.println("On Sentence Pair: " + numPairs);
            }
            List<String> englishWords = sentencePair.getEnglishWords();
            if (englishWords.size() > maxEnglishWords)
                maxEnglishWords = englishWords.size();
            List<String> frenchWords = sentencePair.getFrenchWords();
            frenchWords.replaceAll(String::toLowerCase);
            englishWords.replaceAll(String::toLowerCase);
            for (String en: englishWords) {
                englishWordIndexer.addAndGetIndex(en);
            }
            for (String fr: frenchWords) {
                frenchWordIndexer.addAndGetIndex(fr);
            }
        }
        int nullWordIndex = englishWordIndexer.addAndGetIndex("null");
        maxEnglishWords += 1;  // check again later once
        System.out.println("Num Pair of Sentences: " + numPairs);
        return maxEnglishWords;
    }


    public Alignment getAlignment(SentencePair sentencePair) {
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();
        englishWords.add("null");
        frenchWords.replaceAll(String::toLowerCase);
        englishWords.replaceAll(String::toLowerCase);
        int numFrench = frenchWords.size();
        int numEnglish = englishWords.size();
        Alignment alignment = new Alignment();
        double[][] aTable = new double[numEnglish][numEnglish];
        double[] piTable = new double[numEnglish];

        double temp;
        // fill A table
        for(int j=0; j<numEnglish; j++){
            aTable[j][numEnglish-1] = Math.log(epsilon);
        }
        for(int j=0; j<numEnglish-1; j++){
            aTable[numEnglish-1][j] = Math.log((1 - epsilon)/(numEnglish-1));
        }
        for(int j=0; j<numEnglish-1; j++){
            temp = Double.NEGATIVE_INFINITY;
            for(int k=0; k<numEnglish-1; k++){
                temp = SloppyMath.logAdd(temp, psi[Math.abs(k-j)]);
            }
            for(int k=0; k<numEnglish-1; k++){
                aTable[j][k] = psi[Math.abs(k-j)] + Math.log(1-epsilon) - temp;
            }
        }

        // do smoothing
        for(int j=0; j<numEnglish-1; j++){
            for(int k=0; k<numEnglish-1; k++){
                aTable[j][k] = SloppyMath.logAdd(Math.log(alpha) + Math.log(1.0/(numEnglish-1)),
                        aTable[j][k] + Math.log(1-alpha));
            }
        }

        // fill pi table
        System.arraycopy(aTable[0], 0, piTable, 0, numEnglish);

        double[][] score = new double[numEnglish][numFrench];
        int[][] backPointers = new int[numEnglish][numFrench];

        // fill score table
        for(int j=0; j<numEnglish; j++) {
            String key = frenchWords.get(0) + "===" + englishWords.get(j);
            score[j][0] = tsIndexer.get(key) + piTable[j];
        }
        double curTemp;
        for(int j=1; j<numFrench; j++){
            for(int k=0; k<numEnglish; k++) { // check to start from 0 or 1
                temp = Double.NEGATIVE_INFINITY; // log 0
                backPointers[k][j] = -1;
                for(int l=0; l<numEnglish; l++){
                    curTemp = score[l][j-1] + aTable[l][k];
                    if (curTemp > temp){
                        temp = curTemp;
                        backPointers[k][j] = l;
                    }
                }
                String key = frenchWords.get(j) + "===" + englishWords.get(k);
                score[k][j] = tsIndexer.get(key) + temp;
            }
        }

        // get max and follow the backpointers in reverse
        curTemp = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for(int i=0; i<numEnglish; i++){
            if (score[i][numFrench-1] > curTemp){
                curTemp = score[i][numFrench-1];
                maxIndex = i;
            }
        }
        if (maxIndex != numEnglish -1)
            alignment.addAlignment(maxIndex, numFrench-1, true);
        int i = numFrench -1;
        while(i>0){
            maxIndex = backPointers[maxIndex][i];
            if (maxIndex != numEnglish -1)
                alignment.addAlignment(maxIndex, i-1, true);
            i -= 1;
        }
        englishWords.remove("null");
        return alignment;

    }
}
