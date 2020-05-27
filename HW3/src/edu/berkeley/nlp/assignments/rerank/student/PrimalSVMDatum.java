package edu.berkeley.nlp.assignments.rerank.student;
import edu.berkeley.nlp.util.IntCounter;

public class PrimalSVMDatum {
    IntCounter goldFeatures;
    double[] kBestListLoss;
    int[][] kBestListfeatures;

    public PrimalSVMDatum(IntCounter goldFeatures, double[] kBestListLoss, int[][] kBestListfeatures){
        this.goldFeatures = goldFeatures;
        this.kBestListLoss = kBestListLoss;
        this.kBestListfeatures = kBestListfeatures;
    }
    public int[][] getKBestListfeatures() {
        return kBestListfeatures;
    }
    public IntCounter getGoldFeatures() {
        return goldFeatures;
    }
    public double[] getKBestListLoss() {
        return kBestListLoss;
    }
}
