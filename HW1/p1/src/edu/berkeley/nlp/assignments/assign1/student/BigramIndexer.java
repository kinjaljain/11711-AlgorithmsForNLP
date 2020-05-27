package edu.berkeley.nlp.assignments.assign1.student;

public class BigramIndexer extends OpenAddressHashMap {

    public BigramIndexer(int capacity, float loadFactor) {
        super(capacity, loadFactor, 2f);
    }

    public int addAndGetIndex(int w1, int w2) {
        long key = getKey(w1, w2);
        int index = getIndex(key);
        boolean is_resized = updateValue(index, key);
        // If resizing happened, find the index with new hash
        if (is_resized) {
            key = getKey(w1, w2);
            getIndex(key);
        }
        return index;
    }
}