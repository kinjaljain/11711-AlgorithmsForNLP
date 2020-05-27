package edu.berkeley.nlp.assignments.rerank.student;
import edu.berkeley.nlp.util.IntCounter;

public class PerceptronDatum {
    int goldTreeIndex;
    IntCounter goldFeatures;
    int[][] kBestListfeatures;

    public PerceptronDatum(int goldTreeIndex, IntCounter goldFeatures, int[][] kBestListfeatures){
        this.goldTreeIndex = goldTreeIndex;
        this.goldFeatures = goldFeatures;
        this.kBestListfeatures = kBestListfeatures;
    }
    public int getGoldIndex(){
        return goldTreeIndex;
    }
    public int[][] getKBestListfeatures() {
        return kBestListfeatures;
    }
    public IntCounter getGoldFeatures() {
        return goldFeatures;
    }
}