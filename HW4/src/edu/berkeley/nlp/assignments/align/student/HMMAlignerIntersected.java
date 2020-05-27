package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.Pair;

import java.util.Set;

public class HMMAlignerIntersected implements WordAligner {
    HMMEnglishGivenFrench hmme2f;
    HMMFrenchGivenEnglish hmmf2e;
    boolean intersected = false;

    HMMAlignerIntersected(Iterable<SentencePair> trainingData, boolean intersect) {
        if (intersect){
            intersected = true;
            hmmf2e = new HMMFrenchGivenEnglish();
            hmmf2e.train(trainingData);
        }

        hmme2f = new HMMEnglishGivenFrench();
        hmme2f.train(trainingData);
    }

    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignmente2f = hmme2f.getAlignment(sentencePair);
        Alignment alignment = new Alignment();
        if (intersected){
            Alignment alignmentf2e = hmmf2e.getAlignment(sentencePair);
            Set<Pair<Integer, Integer>> a = alignmente2f.getSureAlignments();
            for (Pair<Integer, Integer> val : a) {
                int e = val.getFirst();
                int f = val.getSecond();
                if (alignmentf2e.containsSureAlignment(e, f)) {
                    alignment.addAlignment(e, f, true);
                }
            } 
        }
        else{
            alignment = alignmente2f;
        }
        return alignment;
    }
}

