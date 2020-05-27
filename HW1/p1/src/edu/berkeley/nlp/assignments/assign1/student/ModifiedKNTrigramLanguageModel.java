package edu.berkeley.nlp.assignments.assign1.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.MemoryUsageUtils;
import edu.berkeley.nlp.util.StringIndexer;
import java.util.ArrayList;
import java.util.List;

// Implements Modified Kneser-Ney Smoothing (Chen & Goodman) on Trigram Language Model
public class ModifiedKNTrigramLanguageModel implements NgramLanguageModel {
        static final String START = NgramLanguageModel.START;
        static final String STOP = NgramLanguageModel.STOP;
        static final float loadFactor = 0.70f;
        static final float discount = 0.75f;
        private int[] wordCounter = new int[200000];
        private int total_bigrams;
        private int total_trigrams;
        private int[] fertility_x_unigram = new int[200000];
        private int[] fertility_unigram_x = new int[200000];
        private int[] fertility_x_bigram = new int[200000];
        private int[] fertility_x_unigram_x = new int[200000];
        private int[] fertility_bigram_x = new int[200000];
        public StringIndexer wordIndexer = EnglishWordIndexer.getIndexer();
        public BigramIndexer bigramIndexer = new BigramIndexer(1800000, loadFactor);
        public TrigramIndexer trigramIndexer;
        public int[] n1_fertility_unigram_x;
        public int[] n2_fertility_unigram_x;
        public int[] n3_fertility_unigram_x;
        public int[] n1_fertility_bigram_x;
        public int[] n2_fertility_bigram_x;
        public int[] n3_fertility_bigram_x;
        public double[] discounts;


        public ModifiedKNTrigramLanguageModel(Iterable<List<String>> sentenceCollection) {
            System.out.println("Building TrigramLanguageModel . . .");
            int sent = 0;
            for (List<String> sentence : sentenceCollection) {
                sent++;
                if (sent % 100000 == 0) {
                    System.out.println("On sentence " + sent);
                    MemoryUsageUtils.printMemoryUsage();
                }
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
                for(int i=1; i<numWords; i++) {
                    int word2_index = wordIndexer.addAndGetIndex(stoppedSentence.get(i));
                    if (word2_index >= wordCounter.length) {
                        wordCounter = CollectionUtils.copyOf(wordCounter, wordCounter.length * 2);
                        fertility_x_unigram_x = CollectionUtils.copyOf(fertility_x_unigram_x, wordCounter.length * 2);
                    }
                    wordCounter[word2_index]++;
                    bigramIndexer.addAndGetIndex(word1_index, word2_index);
                    if(bigramIndexer.getValue(word1_index, word2_index)== 1){
                        if(word1_index >= fertility_unigram_x.length)
                            fertility_unigram_x = CollectionUtils.copyOf(fertility_unigram_x, bigramIndexer.data.length);
                        fertility_unigram_x[word1_index]++;
                        if(word2_index >= fertility_x_unigram.length)
                            fertility_x_unigram = CollectionUtils.copyOf(fertility_x_unigram, bigramIndexer.data.length);
                        fertility_x_unigram[word2_index]++;
                    }
                    word1_index = word2_index;
                }
            }
            total_bigrams = bigramIndexer.size;
            System.out.println("Number of bigrams: "+ total_bigrams);
            trigramIndexer =  new TrigramIndexer(bigramIndexer.size, loadFactor);
            sent = 0;
            char c = 'n';
            for (List<String> sentence : sentenceCollection){
                sent++;
                if (sent % 100000 == 0) {
                    System.out.println("On sentence " + sent);
                    MemoryUsageUtils.printMemoryUsage();
                }
                List<String> stoppedSentence = new ArrayList<String>(sentence);
                stoppedSentence.add(0, NgramLanguageModel.START);
                stoppedSentence.add(STOP);
                int word1_index = wordIndexer.addAndGetIndex(stoppedSentence.get(0));
                int word2_index = wordIndexer.addAndGetIndex(stoppedSentence.get(1));
                int word1word2_index = bigramIndexer.getIndex(bigramIndexer.getKey(word1_index, word2_index));
                int numWords = stoppedSentence.size();
                for(int i=2; i<numWords; i++) {
                    int word3_index = wordIndexer.addAndGetIndex(stoppedSentence.get(i));
                    trigramIndexer.addAndGetIndex(word1word2_index, word3_index);
                    if(trigramIndexer.getValue(word1word2_index, word3_index)== 1){
                        if(word1word2_index >= fertility_bigram_x.length)
                            fertility_bigram_x = CollectionUtils.copyOf(fertility_bigram_x, bigramIndexer.data.length);
                        fertility_bigram_x[word1word2_index]++;
                        if(word2_index >= fertility_x_unigram_x.length)
                            fertility_x_unigram_x = CollectionUtils.copyOf(fertility_x_unigram_x, bigramIndexer.data.length);
                        fertility_x_unigram_x[word2_index]++;
                        c = 'y';

                    }
                    int word2word3_index = bigramIndexer.getIndex(bigramIndexer.getKey(word2_index, word3_index));
                    if(c=='y'){
                        if(word2word3_index >= fertility_x_bigram.length)
                            fertility_x_bigram = CollectionUtils.copyOf(fertility_x_bigram, bigramIndexer.data.length);
                        fertility_x_bigram[word2word3_index]++;
                        c = 'n';
                    }
                    word2_index = word3_index;
                    word1word2_index = word2word3_index;
                }
            }
            total_trigrams = trigramIndexer.size;
            System.out.println("Number of trigrams: "+ total_trigrams);
            n1_fertility_unigram_x = new int[fertility_x_unigram.length];
            n2_fertility_unigram_x = new int[fertility_x_unigram.length];
            n3_fertility_unigram_x = new int[fertility_x_unigram.length];
            putNUnigram(bigramIndexer, n1_fertility_unigram_x, n2_fertility_unigram_x, n3_fertility_unigram_x);
            n1_fertility_bigram_x = new int[fertility_bigram_x.length];
            n2_fertility_bigram_x = new int[fertility_bigram_x.length];
            n3_fertility_bigram_x = new int[fertility_bigram_x.length];
            putNBigram(trigramIndexer, n1_fertility_bigram_x, n2_fertility_bigram_x, n3_fertility_bigram_x);
            discounts = getDiscounts(wordCounter, bigramIndexer.counter, trigramIndexer.counter);
            System.out.println("Done building TrigramLanguageModel.");
            MemoryUsageUtils.printMemoryUsage();

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
                    long fertility_x_word1_x = Math. max(fertility_x_unigram_x[location_word1],0);
                    if (fertility_x_word1_x > 0) {
                        int location_word1word2 = bigramIndexer.getIndex(
                                bigramIndexer.getKey(ngram[from], ngram[from + 1]));
                        if (location_word1word2 >= 0 && location_word1word2 < fertility_x_bigram.length) {
                            int count_word1word2 = Math.max(bigramIndexer.counter[location_word1word2], 0);
                            double d;
                            if(count_word1word2 == 1)
                                d = discounts[0];
                            else if(count_word1word2 == 2)
                                d = discounts[1];
                            else if(count_word1word2 >=3)
                                d = discounts[2];
                            else
                                d = 0;
                            double term1 = Math.max(0.0, (fertility_x_bigram[location_word1word2]) - d);
                            double term2 = discounts[0]*n1_fertility_unigram_x[ngram[from]] +
                                    discounts[1]*n2_fertility_unigram_x[ngram[from]] +
                                    discounts[2]*n3_fertility_unigram_x[ngram[from]];
                            int location_word2 = ngram[from + 1];
                            if (location_word2 >= 0 && location_word2 < fertility_x_unigram.length) {
                                double term3 = Math.max(fertility_x_unigram[ngram[from + 1]], 0) * term2/
                                        (double) total_bigrams;
                                double x = (term1 + term3) / fertility_x_word1_x;
                                if(x!=0)
                                    return x;
                            }
                        }
                    }
                }
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
                                    double d;
                                    if (count_word1word2 == 1)
                                        d = discounts[0];
                                    else if (count_word1word2 == 2)
                                        d = discounts[1];
                                    else if (count_word1word2 >= 3)
                                        d = discounts[2];
                                    else
                                        d = 0;
                                    double term1 = Math.max(count_word1word2word3 - d, 0);
                                    double term2 = discounts[0] * n1_fertility_bigram_x[location_word1word2] +
                                            discounts[1] * n2_fertility_bigram_x[location_word1word2] +
                                            discounts[2] * n3_fertility_bigram_x[location_word1word2];
                                    long fertility_x_word2_x = Math.max(fertility_x_unigram_x[location_word2], 0);
                                    if (fertility_x_word2_x > 0) {
                                        int location_word2word3 = bigramIndexer.getIndex(
                                                bigramIndexer.getKey(ngram[from + 1], ngram[from + 2]));
                                        if (location_word2word3 >= 0 && location_word2word3 < fertility_x_bigram.length) {
                                            long count_word2word3 = Math.max(bigramIndexer.counter[location_word2word3], 0);
                                            double d2;
                                            if (count_word2word3 == 1)
                                                d2 = discounts[0];
                                            else if (count_word2word3 == 2)
                                                d2 = discounts[1];
                                            else if (count_word2word3 >= 3)
                                                d2 = discounts[2];
                                            else
                                                d2 = 0;
                                            double term3_1 = Math.max(0.0, (fertility_x_bigram[location_word2word3]) - d2);
                                            double term3_2 = discounts[0] * n1_fertility_unigram_x[ngram[from + 1]] +
                                                    discounts[1] * n2_fertility_unigram_x[ngram[from + 1]] +
                                                    discounts[2] * n3_fertility_unigram_x[ngram[from + 1]];
                                            double term3_3 = Math.max(fertility_x_unigram[ngram[from + 2]], 0) * term3_2 /
                                                    (double) total_bigrams;
                                            double term3 = (term3_1 + term3_3) / fertility_x_word2_x;
                                            double x = (term1 + (term2 * term3)) / count_word1word2;
                                            if (x != 0) {
                                                return Math.log(x);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return getNgramLogProbability(ngram, from + 1, from + 3);
                }
                return getNgramLogProbability(ngram, from + 2, from + 3);
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
            } else if (ngram.length == 2) {
                long val = bigramIndexer.getValue(ngram[0], ngram[1]);
                if (val > 0)
                    count = val;
            } else if (ngram.length == 1) {
                if (ngram[0] > 0 && ngram[0] < wordCounter.length)
                    count = wordCounter[ngram[0]];
            }
            return Math.max(0, count);
        }

        public double[] getDiscounts(int[] wordcounter, int[] bigramcounter, int[] trigramcounter){
            int n1=0, n2=0, n3=0, n4=0;
            double d1=1, d2=2, d3=3;
            double[] d =  new double[3];
            for(int i=0; i<wordcounter.length; i++){
                if(wordcounter[i] == 1)
                    n1++;
                else if(wordcounter[i] == 2)
                    n2++;
                else if(wordcounter[i] == 3)
                    n3++;
                else if(wordcounter[i] == 4)
                    n4++;
            }
            for(int i=0; i<bigramcounter.length; i++){
                if(bigramcounter[i] == 1)
                    n1++;
                else if(bigramcounter[i] == 2)
                    n2++;
                else if(bigramcounter[i] == 3)
                    n3++;
                else if(bigramcounter[i] == 4)
                    n4++;
            }
            for(int i=0; i<trigramcounter.length; i++){
                if(trigramcounter[i] == 1)
                    n1++;
                else if(trigramcounter[i] == 2)
                    n2++;
                else if(trigramcounter[i] == 3)
                    n3++;
                else if(trigramcounter[i] == 4)
                    n4++;
            }
            System.out.println("n1: " + n1 + ", n2: " + n2 + ", n3: " + n3 + ", n4: " + n4);
            d1 -= (2*discount*n2)/n1;
            d2 -= (3*discount*n3)/n2;
            d3 -= (4*discount*n4)/n3;
            d[0] = d1;
            d[1] = d2;
            d[2] = d3;
            return d;
        }

        public void putNUnigram(BigramIndexer bigramIndexer, int[] n1_fertility_unigram_x,
                                int[] n2_fertility_unigram_x, int[] n3_fertility_unigram_x){
            long[] data = bigramIndexer.data;
            for(int i=0; i<data.length; i++){
                long key = data[i];
                if(key != -1) {
                    key = key >> 20;
                    int word_index = (int) (key & 0xFFFFF);
                    if (bigramIndexer.counter[i] == 1) {
                        n1_fertility_unigram_x[word_index]++;
                    } else if (bigramIndexer.counter[i] == 2) {
                        n2_fertility_unigram_x[word_index]++;
                    } else if (bigramIndexer.counter[i] >= 3) {
                        n3_fertility_unigram_x[word_index]++;
                    }
                }
            }
        }
        public void putNBigram(TrigramIndexer trigramIndexer, int[] n1_fertility_bigram_x,
                               int[] n2_fertility_bigram_x, int[] n3_fertility_bigram_x){
            long[] data = trigramIndexer.data;
            for(int i=0; i<data.length; i++){
                long key = data[i];
                key = key >> 20;
                int word1word2_index = (int) (key & 0xFFFFF);
                if(trigramIndexer.counter[i] == 1){
                    n1_fertility_bigram_x[word1word2_index]++;
                }
                else if(trigramIndexer.counter[i] == 2){
                    n2_fertility_bigram_x[word1word2_index]++;
                }
                else if(trigramIndexer.counter[i] >= 3){
                    n3_fertility_bigram_x[word1word2_index]++;
                }
            }
        }
}
