package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.*;

public class HMMEnglishGivenFrench {
    private double epsilon = 0.01;
    private double alpha = 0.4;
    private StringIndexer englishWordIndexer;
    private StringIndexer frenchWordIndexer;
    private HashMap<String, Double> stIndexer;
    private double[] psi;

    public void train(Iterable<SentencePair> trainingData) {
        System.out.println("Training backward");
        englishWordIndexer = new StringIndexer();
        frenchWordIndexer = new StringIndexer();
        stIndexer = new HashMap<>(100000);  //theta
        int num_em_iters = 7;

        // remove +1 if not used anywhere later
        int maxFrenchSentenceLength = initializeIndexers(trainingData, englishWordIndexer, frenchWordIndexer);
        int totalFrenchWords = frenchWordIndexer.size();
        int totalEnglishWords = englishWordIndexer.size();

        System.out.println("Max French Sentence Length: " + maxFrenchSentenceLength);
        System.out.println("English Words: " + totalEnglishWords);
        System.out.println("French Words: " + totalFrenchWords);

//        double uniformTranslationProb = Math.log(1.0/totalEnglishWords); // check once english or french
        initializeTheta(trainingData, totalFrenchWords); // initialise theta

        psi = new double[maxFrenchSentenceLength]; // adding for null as well
        Arrays.fill(psi, Math.log(1.0/maxFrenchSentenceLength)); // initialise psi

        for(int i=0; i< num_em_iters; i++){
            System.out.println("EM Iteration: " + (i+1));
            HashMap<String, Double> newstIndexer = new HashMap<>(stIndexer.size());
//            System.out.println("the , le: ");
//            System.out.println(stIndexer.get("the===le"));
            double[] newpsi = new double[psi.length];
            Arrays.fill(newpsi, Double.NEGATIVE_INFINITY);
            for (SentencePair sentencePair : trainingData) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                frenchWords.add("null");
                englishWords.replaceAll(String::toLowerCase);
                frenchWords.replaceAll(String::toLowerCase);
                int numFrench = frenchWords.size();
                int numEnglish = englishWords.size();
                double[][] aTable = new double[numFrench][numFrench];
                double[] piTable = new double[numFrench];

                double temp;
                // fill A table
                for(int j=0; j<numFrench; j++){
                    aTable[j][numFrench-1] = Math.log(epsilon);
                }
                for(int j=0; j<numFrench-1; j++){
                    aTable[numFrench-1][j] = Math.log((1 - epsilon)/(numFrench-1));
                }
                for(int j=0; j<numFrench-1; j++){
                    temp = Double.NEGATIVE_INFINITY;
                    for(int k=0; k<numFrench-1; k++){
                        temp = SloppyMath.logAdd(temp, psi[Math.abs(k-j)]);
                    }
                    for(int k=0; k<numFrench-1; k++){
                        aTable[j][k] = psi[Math.abs(k-j)] + Math.log(1-epsilon) - temp;
                    }
                }
                // do alpha smoothing
                for(int j=0; j<numFrench-1; j++){
                    for(int k=0; k<numFrench-1; k++){
                        aTable[j][k] = SloppyMath.logAdd(Math.log(alpha) + Math.log(1.0/(numFrench-1)),
                                aTable[j][k] + Math.log(1-alpha));
                    }
                }

                // fill pi table
                System.arraycopy(aTable[0], 0, piTable, 0, numFrench);

                // create alpha table
                double[][] alphaTable = new double[numFrench][numEnglish];
                for(int j=0; j<numFrench; j++) {
                    String key = englishWords.get(0) + "===" + frenchWords.get(j);
                    alphaTable[j][0] = stIndexer.get(key) + piTable[j];
                }
                for(int j=1; j<numEnglish; j++){
                    for(int k=0; k<numFrench; k++) { // check to start from 0 or 1
                        String key = englishWords.get(j) + "===" + frenchWords.get(k);
                        temp = Double.NEGATIVE_INFINITY; // log 0
                        for(int l=0; l<numFrench; l++){
                            temp = SloppyMath.logAdd(temp, aTable[l][k] + alphaTable[l][j-1]);
                        }
                        alphaTable[k][j] = stIndexer.get(key) + temp;
                    }
                }
                // create beta table
                double[][] betaTable = new double[numFrench][numEnglish];
                for(int j=0; j<numFrench; j++) {
                    betaTable[j][numEnglish-1] = 0.0; // log 1 = 0
                }
                for(int j=numEnglish-2; j>=0; j--){ // k, l= start from 1 not 0 earlier
                    for(int k=0; k<numFrench; k++) {
                        temp = Double.NEGATIVE_INFINITY;
                        for(int l=0; l<numFrench; l++){
                            String key = englishWords.get(j+1) + "===" + frenchWords.get(l);
                            temp = SloppyMath.logAdd(temp, stIndexer.get(key) + aTable[k][l] + betaTable[l][j+1]);
                        }
                        betaTable[k][j] = temp;
                    }
                }
                // p' table
                double[][] changePTable = new double[numEnglish][numFrench];
                double[] z = new double[numEnglish];
                Arrays.fill(z, Double.NEGATIVE_INFINITY);
                for(int j=1; j<numEnglish; j++){
                    for(int k=0; k< numFrench; k++){
                        temp = Double.NEGATIVE_INFINITY;
                        for(int l=0; l < numFrench - k; l++){
                            String key1 = englishWords.get(j) + "===" + frenchWords.get(l+k);
                            String key2 = englishWords.get(j) + "===" + frenchWords.get(l);
                            temp = SloppyMath.logAdd(temp, SloppyMath.logAdd(
                                    alphaTable[l+k][j-1]+ betaTable[l][j] + aTable[l][l+k] + stIndexer.get(key1),
                                    alphaTable[l][j-1] + betaTable[l+k][j] + aTable[l+k][l] + stIndexer.get(key2)));
                        } // aTable[l][l+k] + aTable[l+k][l]
                        changePTable[j][k] = temp;
                        z[j] = SloppyMath.logAdd(z[j], temp);
                    }
                }
                // M step
                // get new psi
                for(int j=0; j<numFrench; j++){ // jump max = numEnglish
                    for(int k=1; k<numEnglish; k++){
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
                for(int j=0; j<numEnglish; j++) {
                    double sum = Double.NEGATIVE_INFINITY;
                    for (int k = 0; k < numFrench; k++) {
                        sum = SloppyMath.logAdd(sum, alphaTable[k][j] + betaTable[k][j]);
                    }
                    for (int k = 0; k < numFrench; k++) {
                        key = englishWords.get(j) + "===" + frenchWords.get(k);
                        if (newstIndexer.containsKey(key))
                            newstIndexer.put(key, SloppyMath.logAdd(newstIndexer.get(key),
                                    (alphaTable[k][j] + betaTable[k][j] - sum)));
                        else {
                            newstIndexer.put(key, (alphaTable[k][j] + betaTable[k][j] - sum));
                        }
                    }
                }
                frenchWords.remove("null");
            }
            // normalize new theta
            double[] frenchVal = new double[totalFrenchWords];
            Arrays.fill(frenchVal, Double.NEGATIVE_INFINITY);
            for(String key: newstIndexer.keySet()){
                String frenchWord = key.split("===")[1];
                frenchVal[frenchWordIndexer.addAndGetIndex(frenchWord)] = SloppyMath.logAdd(
                        frenchVal[frenchWordIndexer.addAndGetIndex(frenchWord)], newstIndexer.get(key));
            }
            for(String key: newstIndexer.keySet()){
                String frenchWord = key.split("===")[1];
                newstIndexer.put(key, newstIndexer.get(key) - frenchVal[frenchWordIndexer.addAndGetIndex(frenchWord)]);
            }
            stIndexer = newstIndexer;
            psi = newpsi;
            newstIndexer = null;
            newpsi = null;
        }
    }

    private void initializeTheta(Iterable<SentencePair> trainingData, int totalFrenchWords){
        HashMap<String, Double> tsCounter = new HashMap<>(10000);
        double[] total_s = new double[totalFrenchWords];
        Arrays.fill(total_s, 0);
        for (SentencePair sentencePair : trainingData) {
            List<String> englishWords = sentencePair.getEnglishWords();
            List<String> frenchWords = sentencePair.getFrenchWords();
            frenchWords.add("null");
            englishWords.replaceAll(String::toLowerCase);
            frenchWords.replaceAll(String::toLowerCase);
            double[] s_total_t = new double[englishWords.size()];
            for(int j=0; j< englishWords.size(); j++){
                s_total_t[j] = 0;
                s_total_t[j] += 1.0 * frenchWords.size();
            }
            for(int j=0; j< englishWords.size(); j++) {
                for (String frenchWord : frenchWords) {
                    String key = englishWords.get(j) + "===" + frenchWord;
                    if (tsCounter.containsKey(key)) {
                        double val = tsCounter.get(key);
                        tsCounter.put(key, val + (1.0 / s_total_t[j]));
                    }
                    else {
                        tsCounter.put(key, (1.0 / s_total_t[j]));
                    }
                    total_s[frenchWordIndexer.addAndGetIndex(frenchWord)] += 1.0 / s_total_t[j];
                }
            }
            frenchWords.remove("null");
        }
        for(String key: tsCounter.keySet()){
            String frenchWord = key.split("===")[1];
            // check log once
            stIndexer.put(key, Math.log(tsCounter.get(key) / total_s[frenchWordIndexer.addAndGetIndex(frenchWord)]));
//            stIndexer.put(key, Math.log(1.0/totalFrenchWords));
        }
    }

    private int initializeIndexers(Iterable<SentencePair> trainingData,
                                   StringIndexer englishWordIndexer, StringIndexer frenchWordIndexer){
        int numPairs = 0;
        int maxFrenchWords = 0;
        for (SentencePair sentencePair : trainingData) {
            numPairs++;
            if(numPairs % 1000 == 0){
                System.out.println("On Sentence Pair: " + numPairs);
            }
            List<String> frenchWords = sentencePair.getFrenchWords();
            if (frenchWords.size() > maxFrenchWords)
                maxFrenchWords = frenchWords.size();
            List<String> englishWords = sentencePair.getEnglishWords();
            for (String en: englishWords) {
                englishWordIndexer.addAndGetIndex(en.toLowerCase());
            }
            for (String fr: frenchWords) {
                frenchWordIndexer.addAndGetIndex(fr.toLowerCase());
            }
        }
        int nullWordIndex = frenchWordIndexer.addAndGetIndex("null");
        maxFrenchWords += 1;  // check again later once
        System.out.println("Num Pair of Sentences: " + numPairs);
        return maxFrenchWords;
    }


    public Alignment getAlignment(SentencePair sentencePair) {
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();
        englishWords.replaceAll(String::toLowerCase);
        frenchWords.replaceAll(String::toLowerCase);
        frenchWords.add("null");
        int numFrench = frenchWords.size();
        int numEnglish = englishWords.size();
        Alignment alignment = new Alignment();
        double[][] aTable = new double[numFrench][numFrench];
        double[] piTable = new double[numFrench];

        double temp;
        // fill A table
        for(int j=0; j<numFrench; j++){
            aTable[j][numFrench-1] = Math.log(epsilon);
        }
        for(int j=0; j<numFrench-1; j++){
            aTable[numFrench-1][j] = Math.log((1 - epsilon)/(double)(numFrench-1));
        }
        for(int j=0; j<numFrench-1; j++){
            temp = Double.NEGATIVE_INFINITY;
            for(int k=0; k<numFrench-1; k++){
                temp = SloppyMath.logAdd(temp, psi[Math.abs(k-j)]);
            }
            for(int k=0; k<numFrench-1; k++){
                aTable[j][k] = psi[Math.abs(k-j)] + Math.log(1-epsilon) - temp;
            }
        }

        // do smoothing
        for(int j=0; j<numFrench-1; j++){
            for(int k=0; k<numFrench-1; k++){
                aTable[j][k] = SloppyMath.logAdd(Math.log(alpha) + Math.log(1.0/(numFrench-1)),
                        aTable[j][k] + Math.log(1-alpha));
            }
        }

        // fill pi table
        System.arraycopy(aTable[0], 0, piTable, 0, numFrench);

        double[][] score = new double[numFrench][numEnglish];
        int[][] backPointers = new int[numFrench][numEnglish];

        // fill score table
        for(int j=0; j<numFrench; j++) {
            String key = englishWords.get(0) + "===" + frenchWords.get(j);
            score[j][0] = stIndexer.get(key) + piTable[j];
        }
        double curTemp;
        for(int j=1; j<numEnglish; j++){
            for(int k=0; k<numFrench; k++) { // check to start from 0 or 1
                temp = Double.NEGATIVE_INFINITY; // log 0
                backPointers[k][j] = -1;
                for(int l=0; l<numFrench; l++){
                    curTemp = score[l][j-1] + aTable[l][k];
                    if (curTemp > temp){
                        temp = curTemp;
                        backPointers[k][j] = l;
                    }
                }
                String key = englishWords.get(j) + "===" + frenchWords.get(k);
                score[k][j] = stIndexer.get(key) + temp;
            }
        }

        // get max and follow the backpointers in reverse
        curTemp = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for(int i=0; i<numFrench; i++){
            if (score[i][numEnglish-1] > curTemp){
                curTemp = score[i][numEnglish-1];
                maxIndex = i;
            }
        }
        if (maxIndex != numFrench -1)
            alignment.addAlignment(numEnglish-1, maxIndex, true);
        int i = numEnglish -1;
        while(i>0){
            maxIndex = backPointers[maxIndex][i];
            if (maxIndex != numFrench -1)
                alignment.addAlignment(i-1, maxIndex, true);
            i -= 1;
        }
        frenchWords.remove("null");
        return alignment;
    }
}
