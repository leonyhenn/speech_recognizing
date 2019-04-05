import os, fnmatch
import numpy as np
import string
import re

dataDir = '/u/cs401/A3/data/'
# dataDir = '../data/'

def preprocess(text):
    text = re.sub(r'[^\w\s\d\[\]]',' ',text)
    text = text.lower()
    return text

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    
    n = len(r)
    m = len(h)    
    R = np.zeros((n+1,m+1))
    B = np.zeros((n+1,m+1))
    R[:,0] = np.arange(n+1)
    R[0,:] = np.arange(m+1)
    R[0,0] = 0
    B[:,0] = 1
    B[0,:] = 2
    B[0,0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            delete = R[i - 1, j] + 1
            sub = R[i - 1, j - 1] + (not (r[i - 1] == h[j - 1]))
            ins = R[i, j - 1] + 1
            R[i, j] = min(delete, sub, ins)
            if (R[i, j] == delete):
                B[i, j] = 1 #"up"
            elif (R[i, j] == ins):
                B[i, j] = 2 #"left"
            elif (R[i, j] == sub):
                B[i, j] = 3 #"up-left"

    num_del = 0
    num_sub = 0
    num_ins = 0
    
    i = n
    j = m
    while i != 0 or j != 0:
        
        if B[i,j] == 1:#"up"
            num_del += 1
            i = i - 1
        elif B[i,j] == 2:#"left"
            num_ins += 1
            j = j - 1
        elif B[i,j] == 3:#"up-left"
            if(R[i, j] == R[i - 1, j - 1] + 1):
                num_sub += 1
            i = i - 1
            j = j - 1
        else:
            i = i - 1
            j = j - 1
        
    return R[n,m] / n, num_sub, num_ins, num_del

if __name__ == "__main__":
    # print( 'TODO' ) 
    wer_kaldi = []
    wer_google = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # print( speaker )
            
            google = "transcripts.Google.txt"
            kaldi = "transcripts.Kaldi.txt"
            tr = "transcripts.txt"
            
            google = os.path.join(dataDir, speaker, google)
            kaldi = os.path.join(dataDir, speaker, kaldi)
            tr = os.path.join(dataDir, speaker, tr)

            tr_file = open(tr,'r')
            tr_content = tr_file.readlines()
            if (len(tr_content)==0):
                continue

            kaldi_empty = False
            kaldi_file = open(kaldi,'r')
            kaldi_content = kaldi_file.readlines()
            if (len(kaldi_content)==0):
                kaldi_empty = True

            google_empty = False
            google_file = open(google,'r')
            google_content = google_file.readlines()
            if (len(google_content)==0):
                google_empty = True

            output_file = open("asrDiscussion.txt",'a')

            if kaldi_empty + google_empty < 2:
                for i in range(len(tr_content)):
                    if not kaldi_empty:
                        # print(preprocess(kaldi_content[i]))
                        # print(preprocess(google_content[i]))
                        # print(preprocess(tr_content[i]))
                        kaldi_result = Levenshtein(preprocess(kaldi_content[i]).split(),preprocess(tr_content[i]).split())
                        print("[{}] [{}] [{}] [{}] S:[{}], I:[{}], D:[{}]".format(speaker,"kaldi",i,kaldi_result[0],kaldi_result[1],kaldi_result[2],kaldi_result[3]))
                        output_file.write("[{}] [{}] [{}] [{}] S:[{}], I:[{}], D:[{}] \n".format(speaker,"kaldi",i,kaldi_result[0],kaldi_result[1],kaldi_result[2],kaldi_result[3]))
                        wer_kaldi.append(kaldi_result[0])
                    if not google_empty:
                        google_result = Levenshtein(preprocess(google_content[i]).split(),preprocess(tr_content[i]).split())
                        print("[{}] [{}] [{}] [{}] S:[{}], I:[{}], D:[{}]".format(speaker,"google",i,google_result[0],google_result[1],google_result[2],google_result[3]))
                        output_file.write("[{}] [{}] [{}] [{}] S:[{}], I:[{}], D:[{}] \n".format(speaker,"google",i,google_result[0],google_result[1],google_result[2],google_result[3]))
                        wer_google.append(google_result[0])

    wer_kaldi = np.array(wer_kaldi)
    wer_google = np.array(wer_google)
    kaldi_mean = np.mean(wer_kaldi)
    kaldi_std = np.std(wer_kaldi)
    google_mean = np.mean(wer_google)
    google_std = np.std(wer_google)
    output_file.write("kaldi mean: {} \n".format(kaldi_mean))
    output_file.write("kaldi std: {} \n".format(kaldi_std))
    output_file.write("google mean: {} \n".format(google_mean))
    output_file.write("google std: {} \n".format(google_std))
    output_file.write("From looking at three transcripts I can see that: google removes all content-irrelavant stuff, like 'um'; also it misinterpret vocal sounds as a part of word (were/um) as (where); it also has trouble recognizing repitive or incomplete sentences where it merges those sentences into something else. Kaldi did a good job on recognizing content-irrelavant sounds, such as 'um' and [laughter]. Lso it recognize incomplete or repitive sentence really well.")
    output_file.close()
