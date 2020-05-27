package edu.berkeley.nlp.assignments.assign1.student;

import java.nio.ByteBuffer;
import java.util.Arrays;

// Long to Int HashMap with Linear Probing
public class OpenAddressHashMap {

    private float loadFactor;
    private float resizeFactor;
    private int maxSize;
    public int size = 0;
    public long[] data;
    public int[] counter;
    private static final int FNV_32_INIT = 0x811c9dc5;
    private static final int FNV_32_PRIME = 0x01000193;

    public OpenAddressHashMap(int capacity, float loadFactor, float resizeFactor) {
//        capacity = PrimeFinder.nextPrime(capacity);
        this.data = new long[capacity];
        this.counter = new int[capacity];
        Arrays.fill(data, -1);
        Arrays.fill(counter, -1);
        this.loadFactor = loadFactor;
        this.resizeFactor = resizeFactor;
        this.maxSize = (int) (data.length * loadFactor);
    }

    public int getValue(int key1, int key2) {
        long key = getKey(key1, key2);
        int index = getIndex(key);
        return counter[index];
    }

    public long getKey(int key1, int key2) {
        long key = key1;
        key = key << 20 | key2;
        return key;
    }

    public long sum() {
        long sum = 0;
        for (int i = 0; i < counter.length; i++) {
            long value = counter[i];
            if(value != -1)
                sum += value;
        }
        return sum;
    }

    private int getArrayHash(long val) {
        return cern_hash32(val);
//        return fnv1_hash32(longToBytes(val));
//        return murmur_hash32(longToBytes(val));
    }

    public static int cern_hash32(long val){
        return ((int) (val ^ (val >>> 32))) * 1070981;
    }

    public byte[] longToBytes(long x) {
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
        buffer.putLong(x);
        return buffer.array();
    }

    public static int fnv1_hash32(final byte[] data) {
        int rv = FNV_32_INIT;
        final int len = data.length;
        for(int i = 0; i < len; i++) {
            rv ^= data[i];
            rv *= FNV_32_PRIME;
        }
        return rv;
    }

    public static int murmur_hash32(final byte[] data) {
        final int m = 0x5bd1e995;
        int seed = 0x9747b28c;
        int length = 8;
        int h = seed^length;
        int length4 = length/4;

        for (int i=0; i<length4; i++) {
            final int i4 = i*4;
            int k = (data[i4]&0xff) +((data[i4+1]&0xff)<<8)
                    +((data[i4+2]&0xff)<<16) +((data[i4+3]&0xff)<<24);
            k *= m;
            k ^= k >>> 24;
            k *= m;
            h *= m;
            h ^= k;
        }
        h *= m;
        h ^= h >>> 13;
        h *= m;
        h ^= h >>> 15;

        return h;
    }

    public int getIndex(long key) {
        int index = getArrayHash(key) % data.length;
        if (index < 0) index = -index;
        while (counter[index]!= -1 && data[index]!= key) {
            index = (index+1) % data.length;
        }
        return index;
    }

    private void resizeAndRehash() {
        int prevSize = data.length;
        int dataSize = (int) (prevSize * resizeFactor);
//        dataSize = PrimeFinder.nextPrime(dataSize);
        long[] newData = new long[dataSize];
        int[] newCounter = new int[dataSize];
        Arrays.fill(newData, -1);
        Arrays.fill(newCounter, -1);
        maxSize = (int) (dataSize * loadFactor);
        for(int i=0; i<prevSize; i++) {
            if(counter[i] != -1) {
                long key = data[i];
                int index = getArrayHash(key) % dataSize;
                if (index < 0) index = -index;
                while (newCounter[index] != -1) {
                    index = (index+1) % dataSize;
                }
                newData[index] = data[i];
                newCounter[index] = counter[i];
            }
        }
        data = newData;
        counter = newCounter;
        newData = null;
        newCounter = null;
    }

    public boolean updateValue(int location, long key) {
        long value = counter[location];
        if (value == -1) {
            size++;
            data[location] = key;
            counter[location] = 1;
            if(size >= maxSize){
                resizeAndRehash();
                return true;
            }
        }
        else {
            counter[location] += 1;
        }
        return false;
    }
}
