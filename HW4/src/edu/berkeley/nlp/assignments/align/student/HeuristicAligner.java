package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.List;

public class HeuristicAligner implements WordAligner {
    private StringIndexer englishWordIndexer = new StringIndexer();
    private StringIndexer frenchWordIndexer = new StringIndexer();
    private IntCounter englishCounter = new IntCounter();
    private IntCounter frenchCounter = new IntCounter();
    private CounterMap<Integer, Integer> frenchToEnglishPair = new CounterMap<Integer, Integer>();

    HeuristicAligner(Iterable<SentencePair> trainingData) {
        int numPairs = 0;
        for(SentencePair sentencePair: trainingData){
            numPairs++;
            if(numPairs % 1000 == 0){
                System.out.println("On Sentence pair: " + numPairs);
            }
            List<String> englishWords = sentencePair.getEnglishWords();
            List<String> frenchWords = sentencePair.getFrenchWords();
            for (String en: englishWords) {
                int index = englishWordIndexer.addAndGetIndex(en.toLowerCase());
                englishCounter.incrementCount(index, 1);
            }
            for (String fr: frenchWords) {
                int index = frenchWordIndexer.addAndGetIndex(fr.toLowerCase());
                frenchCounter.incrementCount(index, 1);
                for (String englishWord : englishWords) {
                    int englishWordIndex = englishWordIndexer.addAndGetIndex(englishWord.toLowerCase());
                    frenchToEnglishPair.incrementCount(englishWordIndex, index, 1);
                }
            }
        }
        System.out.println("NumEnglishWords: " + englishWordIndexer.size());
        System.out.println("NumFrenchWords: " + frenchWordIndexer.size());
        System.out.println("NumFrenchToEnglishPair: " + frenchToEnglishPair.size());
    }

    public Alignment alignSentencePair(SentencePair sentencePair){
        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();

        for (int i = 0; i < frenchWords.size(); i++) {
            double alignRatio = Double.NEGATIVE_INFINITY;
            int alignIndex = -1;
            int frenchWordIndex = frenchWordIndexer.addAndGetIndex(frenchWords.get(i).toLowerCase());
            for (int j = 0; j < englishWords.size(); j++) {
                int englishWordIndex = englishWordIndexer.addAndGetIndex(englishWords.get(j).toLowerCase());

                // pmi
//                double currentRatio = frenchToEnglishPair.getCount(englishWordIndex, frenchWordIndex)/
//                        (englishCounter.getCount(englishWordIndex) * frenchCounter.getCount(frenchWordIndex));

                // dice
                double currentRatio = 2 * frenchToEnglishPair.getCount(englishWordIndex, frenchWordIndex)/
                        (englishCounter.getCount(englishWordIndex) + frenchCounter.getCount(frenchWordIndex));

                // ochichai
//                double currentRatio = frenchToEnglishPair.getCount(englishWordIndex, frenchWordIndex)/
//                        Math.sqrt(englishCounter.getCount(englishWordIndex) + frenchCounter.getCount(frenchWordIndex));

                if (currentRatio > alignRatio) {
                    alignRatio = currentRatio;
                    alignIndex = j;
                }
            }
            if(alignIndex != -1){
                alignment.addAlignment(alignIndex, i, true);
            }
        }
        return alignment;
    }
}
