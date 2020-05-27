package edu.berkeley.nlp.assignments.assign1.student;

public class TrigramIndexer extends OpenAddressHashMap{
    public TrigramIndexer(int capacity, float loadFactor) {
        super(capacity, loadFactor, 1.5f);
    }

    public int addAndGetIndex(int w1, int w2) {
        long key = getKey(w1, w2);
        int index = getIndex(key);
        boolean is_resized = updateValue(index, key);
        if (is_resized) {
            key = getKey(w1, w2);
            getIndex(key);
        }
        return index;
    }
}