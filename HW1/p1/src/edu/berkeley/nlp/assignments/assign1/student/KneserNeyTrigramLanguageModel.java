package edu.berkeley.nlp.assignments.assign1.student;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.StringIndexer;
import edu.berkeley.nlp.util.MemoryUsageUtils;

public class KneserNeyTrigramLanguageModel implements NgramLanguageModel {
    private static final String START = NgramLanguageModel.START;
    private static final String STOP = NgramLanguageModel.STOP;
    static final float loadFactor = 0.70f;
    static final float discount = 0.90f;
    private int total_bigrams;
    private int total_trigrams;
    private int total_unigrams;
    private int[] wordCounter = new int[200000];
    private int[] fertility_x_unigram = new int[200000];
    private int[] fertility_unigram_x = new int[200000];
    private int[] fertility_x_unigram_x = new int[200000];
    private int[] fertility_x_bigram = new int[200000];
    private int[] fertility_bigram_x = new int[200000];
    private StringIndexer wordIndexer = EnglishWordIndexer.getIndexer();
    private BigramIndexer bigramIndexer = new BigramIndexer(1500000, loadFactor);
    private TrigramIndexer trigramIndexer;
//    char log_flag = 'y';

    // Trigram Language Model with Kneser-Ney Smoothing Constructor
    // params: List of Sentences
    public KneserNeyTrigramLanguageModel(Iterable<List<String>> sentenceCollection) {
        long startTime = System.nanoTime();
        System.out.println("Building TrigramLanguageModel . . .");
        //  First create UnigramMap and BigramMap simultaneously
        for (List<String> sentence : sentenceCollection) {
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            stoppedSentence.add(0, START);
            stoppedSentence.add(STOP);
            int word1_index = wordIndexer.addAndGetIndex(stoppedSentence.get(0));
            if (word1_index >= wordCounter.length) {
                wordCounter = CollectionUtils.copyOf(wordCounter, wordCounter.length * 2);
                fertility_x_unigram_x = CollectionUtils.copyOf(fertility_x_unigram_x, wordCounter.length * 2);
            }
            wordCounter[word1_index]++;
            int numWords = stoppedSentence.size();
            for (int i = 1; i < numWords; i++) {
                int word2_index = wordIndexer.addAndGetIndex(stoppedSentence.get(i));
                if (word2_index >= wordCounter.length) {
                    wordCounter = CollectionUtils.copyOf(wordCounter, wordCounter.length * 2);
                    fertility_x_unigram_x = CollectionUtils.copyOf(fertility_x_unigram_x, wordCounter.length * 2);
                }
                wordCounter[word2_index]++;
                bigramIndexer.addAndGetIndex(word1_index, word2_index);
                if (bigramIndexer.getValue(word1_index, word2_index) == 1) {
                    if (word1_index >= fertility_unigram_x.length)
                        fertility_unigram_x = CollectionUtils.copyOf(fertility_unigram_x, bigramIndexer.data.length);
                    fertility_unigram_x[word1_index]++;
                    if (word2_index >= fertility_x_unigram.length)
                        fertility_x_unigram = CollectionUtils.copyOf(fertility_x_unigram, bigramIndexer.data.length);
                    fertility_x_unigram[word2_index]++;
                }
                word1_index = word2_index;
            }
        }
        total_bigrams = bigramIndexer.size;
        // Now create TrigramMap with enough size to start with
        trigramIndexer = new TrigramIndexer(bigramIndexer.size, loadFactor);
        int sent = 0;
        char isNewTrigram = 'n';
        for (List<String> sentence : sentenceCollection) {
            sent++;
            if (sent % 1000000 == 0) {
                System.out.println("On sentence " + sent);
            }
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            stoppedSentence.add(0, NgramLanguageModel.START);
            stoppedSentence.add(STOP);
            int word1_index = wordIndexer.addAndGetIndex(stoppedSentence.get(0));
            int word2_index = wordIndexer.addAndGetIndex(stoppedSentence.get(1));
            int word1word2_index = bigramIndexer.getIndex(bigramIndexer.getKey(word1_index, word2_index));
            int numWords = stoppedSentence.size();
            for (int i = 2; i < numWords; i++) {
                int word3_index = wordIndexer.addAndGetIndex(stoppedSentence.get(i));
                trigramIndexer.addAndGetIndex(word1word2_index, word3_index);
                if (trigramIndexer.getValue(word1word2_index, word3_index) == 1) {
                    if (word1word2_index >= fertility_bigram_x.length)
                        fertility_bigram_x = CollectionUtils.copyOf(fertility_bigram_x, bigramIndexer.data.length);
                    fertility_bigram_x[word1word2_index]++;
                    if (word2_index >= fertility_x_unigram_x.length)
                        fertility_x_unigram_x = CollectionUtils.copyOf(fertility_x_unigram_x, bigramIndexer.data.length);
                    fertility_x_unigram_x[word2_index]++;
                    isNewTrigram = 'y';

                }
                int word2word3_index = bigramIndexer.getIndex(bigramIndexer.getKey(word2_index, word3_index));
                if (isNewTrigram == 'y') {
                    if (word2word3_index >= fertility_x_bigram.length)
                        fertility_x_bigram = CollectionUtils.copyOf(fertility_x_bigram, bigramIndexer.data.length);
                    fertility_x_bigram[word2word3_index]++;
                    isNewTrigram = 'n';
                }
                word2_index = word3_index;
                word1word2_index = word2word3_index;
            }
        }
//        total_unigrams = num_unigrams(wordCounter);
//        total_trigrams = trigramIndexer.size;
//        System.out.println("Number of Unigrams: " + total_unigrams);
//        System.out.println("Number of Bigrams: " + total_bigrams);
//        System.out.println("Number of Trigrams: " + total_trigrams);
//        findMaxUnigramCount();
//        findMaxBigramCount();
//        findMaxTrigramCount();
//        findMaxFertilityCount(fertility_unigram_x);
//        findMaxFertilityCount(fertility_x_unigram);
//        findMaxFertilityCount(fertility_x_unigram_x);
//        findMaxFertilityCount(fertility_bigram_x);
//        findMaxFertilityCount(fertility_x_bigram);
        System.out.println("Done Building TrigramLanguageModel.");
        long endTime = System.nanoTime();
        System.out.println("Total Time: " + (double)(endTime - startTime)/(1e9));
//        System.out.println("Perplexity: " + calculatePerplexity(sentenceCollection));
//        log_flag = 'n';
    }

    public static int num_unigrams(int[] a) {
        if (a == null) { return 0; }
        int result = 0;
        for (int i = 0; i < a.length; i++) {
            if(a[i]>0)
                result += 1;
        }
        return result;
    }

    public int getOrder() {
        return 3;
    }

    public double getNgramLogProbability(int[] ngram, int from, int to) {
        int dif = to - from;
        if (dif == 1) {
            int location_word = ngram[from];
            return Math.log((location_word < 0 || location_word >= wordCounter.length || fertility_x_unigram[location_word] == 0) ?
                    1.0/(double) total_bigrams : fertility_x_unigram[location_word] /(double) total_bigrams);
        }
        else if (dif == 2) {
            int location_word1 = ngram[from];
            if (location_word1 >= 0 && location_word1 < fertility_x_unigram_x.length) {
                long fertility_x_word1_x = Math.max(fertility_x_unigram_x[location_word1], 0);
                if (fertility_x_word1_x > 0) {
                    int location_word1word2 = bigramIndexer.getIndex(
                            bigramIndexer.getKey(ngram[from], ngram[from + 1]));
                    double term2 = 0.0;
                    if (location_word1word2 >= 0 && location_word1word2 < fertility_x_bigram.length) {
                        term2 = Math.max(term2, (fertility_x_bigram[location_word1word2]) - discount);
                    }
                    int location_word2 = ngram[from + 1];
                    double term3 = 0.0;
                    if (location_word2 >= 0 && location_word2 < fertility_x_unigram.length) {
                        term3 = discount * Math.max(fertility_x_unigram[ngram[from + 1]], 0) *
                                Math.max(fertility_unigram_x[ngram[from]], 0)/ (double) total_bigrams;
                    }
                    double x = (term2 + term3) / fertility_x_word1_x;
                    if (x != 0) {
                        return Math.log(x);
                    }
                }
            }
            // back off to lower level ngram probability
            return getNgramLogProbability(ngram, from+1, from+2);
        }
        else if (dif == 3) {
            int location_word1 = ngram[from];
            int location_word2 = ngram[from + 1];
            int location_word3 = ngram[from + 2];
            if (location_word3 >= 0 && location_word3 < fertility_x_unigram_x.length) {
                if (location_word1 >= 0 && location_word2 >= 0 && location_word1 < fertility_x_unigram_x.length && location_word2 < fertility_x_unigram_x.length) {
                    int location_word1word2 = bigramIndexer.getIndex(
                            bigramIndexer.getKey(ngram[from], ngram[from + 1]));
                    if (location_word1word2 >= 0 && location_word1word2 < bigramIndexer.counter.length) {
                        double count_word1word2 = Math.max(bigramIndexer.counter[location_word1word2], 0);
                        if (count_word1word2 > 0) {
                            int location_word1word2word3 = trigramIndexer.getIndex(
                                    trigramIndexer.getKey(location_word1word2, location_word3));
                            if (location_word1word2word3 >= 0 && location_word1word2word3 < trigramIndexer.counter.length) {
                                long count_word1word2word3 = trigramIndexer.counter[location_word1word2word3];
                                double term1 = Math.max(count_word1word2word3 - discount, 0.0);
                                long fertility_bigram_word1word2_x = Math.max(fertility_bigram_x[location_word1word2], 0);
                                long fertility_x_word2_x = 0;
                                if (fertility_bigram_word1word2_x > 0) {
                                    fertility_x_word2_x = Math.max(fertility_x_unigram_x[location_word2], fertility_x_word2_x);
                                }
                                double term2 = 0.0;
                                double term3 = 0.0;
                                if (fertility_x_word2_x > 0) {
                                    int location_word2word3 = bigramIndexer.getIndex(
                                            bigramIndexer.getKey(location_word2, location_word3));
                                    if (location_word2word3 >= 0 && location_word2word3 < fertility_x_bigram.length) {
                                        term2 = Math.max(term2, (fertility_x_bigram[location_word2word3]) - discount);
                                    }
                                    term3 = discount * Math.max(fertility_x_unigram[location_word3], 0) *
                                            Math.max(fertility_unigram_x[location_word2], 0)
                                            / (double) total_bigrams;
                                }
                                double x = (term1 + (discount * fertility_bigram_word1word2_x) *
                                        ((term2 + term3) / fertility_x_word2_x)) / count_word1word2;
                                if (x != 0) {
                                    return Math.log(x);
                                }
                            }
                        }
                    }
                }
                return getNgramLogProbability(ngram, from+1, from+3);
            }
            // back off to lower level ngram probability
            return getNgramLogProbability(ngram, from+2, from+3);
        }
        else
            return 0.0;
    }

    public long getCount(int[] ngram) {
        long count = 0;
        if (ngram.length == 3) {
            long val = bigramIndexer.getValue(ngram[0], ngram[1]);
            if (val > 0)
                count = trigramIndexer.getValue(
                        bigramIndexer.getIndex(bigramIndexer.getKey(ngram[0], ngram[1])), ngram[2]);
        }
        else if (ngram.length == 2) {
            long val = bigramIndexer.getValue(ngram[0], ngram[1]);
            if (val > 0)
                count = val;
        }
        else if (ngram.length == 1) {
            if (ngram[0] > 0 && ngram[0] < wordCounter.length)
                count = wordCounter[ngram[0]];
        }
        return Math.max(0, count);
    }

    private void findMaxUnigramCount() {
        int maxCount = 0;
        for (int i = 0; i < wordCounter.length; i++) {
            int value = wordCounter[i];
            if (value > maxCount){
                maxCount = value;
            }
        }
        System.out.println("Max Unigram count: " + maxCount);
    }

    private void findMaxBigramCount() {
        int maxCount = 0;
        for (int i = 0; i < bigramIndexer.data.length; i++) {
            int value = bigramIndexer.counter[i];
            if (value > maxCount){
                maxCount = value;
            }
        }
        System.out.println("Max Bigram count: " + maxCount);
    }

    private void findMaxTrigramCount() {
        int maxCount = 0;
        for (int i = 0; i < trigramIndexer.data.length; i++) {
            int value = (int) trigramIndexer.data[i] & 0xFFFFF;
            if (value > maxCount) {
                maxCount = value;
            }
        }
        System.out.println("Max Trigram count: " + maxCount);
    }

//    private void findMaxFertilityCount(int[] a) {
//        int maxCount = 0;
//        for (int i = 0; i < a.length; i++) {
//            int value = (int) a[i] & 0xFFFFF;
//            if (value > maxCount) {
//                maxCount = value;
//            }
//        }
//        System.out.println("Max fertility count: " + maxCount);
//    }

    public double calculatePerplexity(Iterable<List<String>> sentenceCollection){
        double sum = 0.0;
        int limit = 9000000;
        total_unigrams = num_unigrams(wordCounter);
        for (List<String> sentence : sentenceCollection) {
            if(limit <=0)
                break;
            limit--;
            double probability_sentence = 0.0;
            List<String> s = new ArrayList<String>(sentence);
            s.add(0, START);
            s.add(STOP);
            int[] ngram_array = new int[sentence.size()];
            for (int i = 0; i < sentence.size(); i++){
                ngram_array[i] = wordIndexer.addAndGetIndex(s.get(i));
            }
            probability_sentence = getNgramLogProbability(ngram_array, 0,1);
            if(sentence.size() > 2) {
                probability_sentence *= getNgramLogProbability(ngram_array, 0, 2);
            }
            for (int i = 2; i < sentence.size(); i++){
                probability_sentence *= getNgramLogProbability(ngram_array, i-2, i+1);
            }
            sum += Math.log(probability_sentence);
            System.out.println(sum);
        }
        return Math.pow(2, - (sum/limit));
    }
}
