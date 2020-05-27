package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class IBM1 implements WordAligner {

    private StringIndexer englishWordIndexer = new StringIndexer();
    private StringIndexer frenchWordIndexer = new StringIndexer();
    private int num_em_iters = 5;
    HashMap<String, Double> tsIndexer = new HashMap<>(10000);
    HashMap<String, Double> stIndexer = new HashMap<>(10000);

    IBM1(Iterable<SentencePair> trainingData) {
        int numPairs = 0;
        for (SentencePair sentencePair : trainingData) {
            numPairs++;
            if(numPairs % 1000 == 0){
                System.out.println("On Sentence Pair: " + numPairs);
            }
            List<String> englishWords = sentencePair.getEnglishWords();
            List<String> frenchWords = sentencePair.getFrenchWords();
            for (String en: englishWords) {
                englishWordIndexer.addAndGetIndex(en.toLowerCase());
            }
            for (String fr: frenchWords) {
                frenchWordIndexer.addAndGetIndex(fr.toLowerCase());
            }
        }
//        trainEnglishGivenFrench(trainingData);
        trainFrenchGivenEnglish(trainingData);
    }

    private void trainEnglishGivenFrench(Iterable<SentencePair> trainingData){
        System.out.println("Training French Given English...");
        frenchWordIndexer.addAndGetIndex("null");
        int totalFrenchWords = frenchWordIndexer.size();
        int totalEnglishWords = englishWordIndexer.size();
        System.out.println("English Words: " + totalEnglishWords);
        System.out.println("French Words: " + totalFrenchWords);
        for(int i=0; i< num_em_iters; i++){
            System.out.println("EM Iteration: " + (i+1));
            HashMap<String, Double> tsCounter = new HashMap<>(10000);
            double[] total_s = new double[totalFrenchWords];
            Arrays.fill(total_s, 0);
            // E step
            for (SentencePair sentencePair : trainingData) {
                List<String> englishWords = sentencePair.getEnglishWords();
                List<String> frenchWords = sentencePair.getFrenchWords();
                frenchWords.add(0, "null");
                double[] s_total_t = new double[englishWords.size()];
                for(int j=0; j< englishWords.size(); j++){
                    s_total_t[j] = 0;
                    for (String frenchWord : frenchWords) {
                        if (i == 0) {
                            s_total_t[j] += 1.0;
                        } else {
                            s_total_t[j] += stIndexer.get(englishWords.get(j).toLowerCase() + "===" + frenchWord.toLowerCase());
                        }
                    }
                }
                for(int j=0; j< englishWords.size(); j++) {
                    for (String frenchWord : frenchWords) {
                        String key = englishWords.get(j).toLowerCase() + "===" + frenchWord.toLowerCase();
                        if (i == 0) {
                            if (tsCounter.containsKey(key)) {
                                double val = tsCounter.get(key);
                                tsCounter.put(key, val + (1.0 / s_total_t[j]));
                            }
                            else {
                                tsCounter.put(key, (1.0 / s_total_t[j]));
                            }
                            total_s[frenchWordIndexer.addAndGetIndex(frenchWord.toLowerCase())] += 1.0 / s_total_t[j];
                        }
                        else {
                            double t_val = stIndexer.get(key);
                            if (tsCounter.containsKey(key)) {
                                double val = tsCounter.get(key);
                                tsCounter.put(key, val + (t_val/ s_total_t[j]));
                            }
                            else {
                                tsCounter.put(key, (t_val / s_total_t[j]));
                            }
                            total_s[frenchWordIndexer.addAndGetIndex(frenchWord.toLowerCase())] += t_val / s_total_t[j];
                        }
                    }
                }
                frenchWords.remove("null");
            }

            // traverse only for present pairs to speed up
            // M step
            for(String key: tsCounter.keySet()){
                String frenchWord = key.split("===")[1];
                stIndexer.put(key, tsCounter.get(key) / total_s[frenchWordIndexer.addAndGetIndex(frenchWord.toLowerCase())]);
            }
        }

    }
    private void trainFrenchGivenEnglish(Iterable<SentencePair> trainingData){
        System.out.println("Training French Given English...");
        englishWordIndexer.addAndGetIndex("null");
        int totalFrenchWords = frenchWordIndexer.size();
        int totalEnglishWords = englishWordIndexer.size();
        System.out.println("English Words: " + totalEnglishWords);
        System.out.println("French Words: " + totalFrenchWords);
        for(int i=0; i< num_em_iters; i++){
            System.out.println("EM Iteration: " + (i+1));
            HashMap<String, Double> tsCounter = new HashMap<>(10000);
            double[] total_s = new double[totalEnglishWords];
            Arrays.fill(total_s, 0);
            // E step
            for (SentencePair sentencePair : trainingData) {
                List<String> englishWords = sentencePair.getEnglishWords();
                List<String> frenchWords = sentencePair.getFrenchWords();
                englishWords.add(0, "null");
                double[] s_total_t = new double[frenchWords.size()];
                for(int j=0; j< frenchWords.size(); j++){
                    s_total_t[j] = 0;
                    for (String englishWord : englishWords) {
                        if (i == 0) {
                            s_total_t[j] += 1.0;
                        } else {
                            s_total_t[j] += tsIndexer.get(frenchWords.get(j).toLowerCase() + "===" + englishWord.toLowerCase());
                        }
                    }
                }
                for(int j=0; j< frenchWords.size(); j++) {
                    for (String englishWord : englishWords) {
                        String key = frenchWords.get(j).toLowerCase() + "===" + englishWord.toLowerCase();
                        if (i == 0) {
                            if (tsCounter.containsKey(key)) {
                                double val = tsCounter.get(key);
                                tsCounter.put(key, val + (1.0 / s_total_t[j]));
                            }
                            else {
                                tsCounter.put(key, (1.0 / s_total_t[j]));
                            }
                            total_s[englishWordIndexer.addAndGetIndex(englishWord.toLowerCase())] += 1.0 / s_total_t[j];
                        }
                        else {
                            double t_val = tsIndexer.get(key);
                            if (tsCounter.containsKey(key)) {
                                double val = tsCounter.get(key);
                                tsCounter.put(key, val + (t_val/ s_total_t[j]));
                            }
                            else {
                                tsCounter.put(key, (t_val / s_total_t[j]));
                            }
                            total_s[englishWordIndexer.addAndGetIndex(englishWord.toLowerCase())] += t_val / s_total_t[j];
                        }
                    }
                }
                englishWords.remove("null");
            }

            // traverse only for present pairs to speed up
            // M step
            for(String key: tsCounter.keySet()){
                String englishWord = key.split("===")[1];
                tsIndexer.put(key, tsCounter.get(key) / total_s[englishWordIndexer.addAndGetIndex(englishWord.toLowerCase())]);
            }
        }
    }

    // without intersected..
    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();
        int alignIndex;

        for (int i = 0; i < frenchWords.size(); i++) {
//            double alignProb = stIndexer.get(englishWords.get(i).toLowerCase() + "===" + "null");
            double alignProb = tsIndexer.get(frenchWords.get(i).toLowerCase() + "===" + "null");
            alignIndex = -1;
            for (int j = 0; j < englishWords.size(); j++) {
//                double currentProb = stIndexer.get(englishWords.get(i).toLowerCase() + "===" + frenchWords.get(j).toLowerCase());

                double currentProb = tsIndexer.get(frenchWords.get(i).toLowerCase() + "===" + englishWords.get(j).toLowerCase());
                if (currentProb > alignProb) {
                    alignProb = currentProb;
                    alignIndex = j;
                }
            }
            if(alignIndex != -1) {
                alignment.addAlignment(alignIndex, i, true);
            }
        }
        return alignment;
    }
}




